import numpy as np
from numba import float64, jit, vectorize


# Following Roland Seferian, equation 3
# https://www.geosci-model-dev.net/11/321/2018/gmd-11-321-2018.pdf
@jit(nopython=True)
def calculate_alpha_dir(n_lambda, µ):
    a = np.sqrt(1.0 - (1.0 - µ**2) / n_lambda**2)
    b = ((a - n_lambda * µ) / (a + n_lambda * µ)) ** 2
    c = ((µ - n_lambda * a) / (µ + n_lambda * a)) ** 2

    return 0.5 * (b + c)


@jit(nopython=True)
def calculate_diffuse_reflection(n_λ, σ):
    # Diffuse albedo from Jin et al., 2006 (Eq 5b)
    return -0.1479 + 0.1502 * n_λ - 0.0176 * n_λ * σ


@jit(nopython=True)
def surface_roughness(µ, σ):
    # Surface roughness following Jin et al. 2014 equation 4
    # This roughness parameter determines the Fresnel refraction
    # index from flat surface
    return (
        0.0152
        - 1.7873 * µ
        + 6.8972 * (µ**2)
        - 8.5778 * (µ**3)
        + 4.071 * σ
        - 7.6446 * µ * σ
    ) * np.exp(0.1643 - 7.8409 * µ - 3.5639 * µ**2 - 2.3588 * σ + 10.054 * µ * σ)


@jit(nopython=True)
def calculate_direct_reflection(n_λ, µ, σ):
    # Direct reflection following Jin et al. 2014 equation 1
    f_0 = calculate_alpha_dir(1.34, µ)
    f_λ = calculate_alpha_dir(n_λ, µ)

    return f_λ - (surface_roughness(µ, σ) * f_λ / f_0)


@jit(nopython=True)
def calculate_direct_reflection_from_chl(
    λ, chl, alpha_chl, alpha_w, beta_w, σ, µ, alpha_direct
):
    rw = 0.48168549 - 0.014894708 * σ - 0.20703885 * σ**2

    # Determine absorption and backscattering
    # coefficients to determine reflectance below the surface (Ro) once for all
    # Backscattering by chlorophyll:

    if float(chl) == 0.0 or np.isnan(chl):
        a_bp = 0.0 * λ
        b_chl = 0.0 * λ
    else:
        a_bp = 0.06 * alpha_chl * np.exp(np.log(chl) * 0.65) + 0.2 * (
            0.00635 + 0.06 * (np.exp(np.log(chl) * 0.65)) * np.exp(0.014 * (440.0 - λ))
        )

        # Backscattering of biological pigment (b_chl) with λ expressed here in nm and [Chl] in mg m−3. This
        # formulation is valid for [Chl] ranging between 0.02 and 2 mg m−3 (Morel and Maritorena (2001))
        # Equation 12 Roland Seferian, 2018
        b_chl = (0.416 * np.exp(0.766 * np.log(chl))) * (
            0.002
            + 0.01
            * (0.5 - 0.25 * np.log(chl))
            * np.exp((0.5 * (np.log(chl) - 0.3)) * np.log(λ / 550.0))
        )

    # # Use Morel 91 formula to compute the direct reflectance below the surface (Morel-Gentili(1991), Eq (12))
    n = 0.5 * beta_w / (0.5 * beta_w + b_chl)

    # Equation 11 Roland Seferian, 2018
    beta = 0.6279 - 0.2227 * n - 0.0513 * n**2 + (0.2465 * n - 0.3119) * µ

    # Equation 10 Roland Seferian, 2018
    R0 = beta * (0.5 * beta_w + b_chl) / (alpha_w + a_bp)

    # Water leaving albedo, equation 8 Roland Seferian, 2018
    return R0 * (1.0 - rw) / (1 - rw * R0)


@jit(nopython=True)
def calculate_diffuse_reflection_from_chl(
    λ, chl, alpha_chl, alpha_w, beta_w, σ, alpha_direct
):
    #  In the case of ocean interior reflectance for direct incoming radiation it depends on µ = cos(θ) whereas in the
    # case of ocean interior reflectance for diffuse µ = 0.676. This value is considered an effective angle of incoming radiation of 47.47◦
    # according to Morel and Gentili (1991). Hence
    return calculate_direct_reflection_from_chl(
        λ, chl, alpha_chl, alpha_w, beta_w, σ, np.arccos(0.676), alpha_direct
    )


@vectorize([float64(float64)])
def whitecap(wind):
    # Whitecap effect as defined by Salisbury et al. 2014. NOTE that the value in paper is in percent
    # so we use the ratio instead (/100.)
    # Salisbury, D. J., Anguelova, M. D., and Brooks, I. M.: Global Distribution and Seasonal
    # Dependence of Satellite-based Whitecap Fraction
    #
    # Whitecaps are the surface manifestation of bubble plumes, created when
    # surface gravity waves break and entrain air into the water column.
    # They enhance air-sea exchange, introducing physical processes different from
    # those operating at the bubble-free water surface.

    return 0.000397 * (np.exp(1.59 * np.log(wind)))


@jit(nopython=True)
def calculate_spectral_and_broadband_OSA(
    wind,
    alpha_wc,
    alpha_direct,
    alpha_diffuse,
    alpha_direct_chl,
    alpha_diffuse_chl,
    solar_energy,
):
    wc = whitecap(wind)
    # OSA is the result array containing info on total diffuse and direct broadband ocean surface albedo
    # but also separated into uv, visible, and near-infrared components
    # 0=direct broadband, 1=diffuse broadband, 2=direct visible, 3=diffuse visible, 4=direct iv, 5=diffuse uv,
    # 6=direct near infrared, 7=diffuse near infrared
    OSA = np.zeros((1, 2))
    OSA_UV = np.zeros((1, 2))
    OSA_VIS = np.zeros((1, 2))

    OSA_direct = (alpha_direct + alpha_direct_chl) * (1 - wc) + wc * alpha_wc
    OSA_diffuse = (alpha_diffuse + alpha_diffuse_chl) * (1 - wc) + wc * alpha_wc

    # Integrate across all wavelengths 200-4000nm at 10 nm wavelength bands and then
    # weight by the solar energy at each band to get total broadband.
    # The solar energy is dimensionless with sum equal to 1 and therefore already weighted.
    OSA[0, 0] = np.sum(OSA_direct * solar_energy)
    OSA[0, 1] = np.sum(OSA_diffuse * solar_energy)

    # Calculate the visible direct OSA
    start_index_uv = len(np.arange(200, 200, 10))
    end_index_uv = len(np.arange(200, 410, 10))
    start_index_visible = len(np.arange(200, 400, 10))
    end_index_visible = len(np.arange(200, 710, 10))

    # PAR/VIS
    OSA_VIS[0, 0] = np.sum(
        OSA_direct[start_index_visible:end_index_visible]
        * solar_energy[start_index_visible:end_index_visible]
    )
    OSA_VIS[0, 1] = np.sum(
        OSA_diffuse[start_index_visible:end_index_visible]
        * solar_energy[start_index_visible:end_index_visible]
    )
    # UV
    OSA_UV[0, 0] = np.sum(
        OSA_direct[start_index_uv:end_index_uv]
        * solar_energy[start_index_uv:end_index_uv]
    )
    OSA_UV[0, 1] = np.sum(
        OSA_diffuse[start_index_uv:end_index_uv]
        * solar_energy[start_index_uv:end_index_uv]
    )

    return OSA, OSA_UV, OSA_VIS


@jit(nopython=True)
def calculate_OSA(
    µ_deg,
    uv,
    chl,
    wavelengths,
    refractive_indexes,
    alpha_chl,
    alpha_w,
    beta_w,
    alpha_wc,
    solar_energy,
):
    if µ_deg < 0 or µ_deg > 100:
        µ_deg = 0

    µ = np.cos(np.radians(µ_deg))

    # Solar zenith angle
    # wind is wind at 10 m height (m/s)
    σ = np.sqrt(0.003 + 0.00512 * uv)

    # Direct reflection
    alpha_direct = calculate_direct_reflection(refractive_indexes, µ, σ)

    # Diffuse reflection
    alpha_diffuse = calculate_diffuse_reflection(refractive_indexes, σ)
    # Reflection from chlorophyll and biological pigments

    alpha_direct_chl = calculate_direct_reflection_from_chl(
        wavelengths, chl, alpha_chl, alpha_w, beta_w, σ, µ, alpha_direct
    )
    # Diffuse reflection interior of water from chlorophyll
    alpha_diffuse_chl = calculate_diffuse_reflection_from_chl(
        wavelengths, chl, alpha_chl, alpha_w, beta_w, σ, alpha_diffuse
    )
    # OSA
    return calculate_spectral_and_broadband_OSA(
        uv,
        alpha_wc,
        alpha_direct,
        alpha_diffuse,
        alpha_direct_chl,
        alpha_diffuse_chl,
        solar_energy,
    )
