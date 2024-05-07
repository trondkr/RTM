import logging
from typing import Tuple

import numpy as np
import xarray as xr

from CMIP6_config import Config_albedo


# Class for calculating albedo of sea-ice, snow and snow-ponds
# The albedo and absorbed/transmitted flux parameterizations for
# snow over ice, bare ice and ponded ice.
# Methods applied from cesm3 online code :
# http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cice/ice_shortwave.F90.html#COMPUTE_ALBEDOS
class CMIP6_cesm3():

    def __init__(self) -> None:

        self.config = Config_albedo()
        self.config.setup_parameters()
        self.chl_abs_A, self.chl_abs_B, self.chl_abs_wavelength = self.config.setup_absorption_chl()
        self.o3_abs, self.o3_wavelength = self.config.setup_ozone_uv_spectrum()

        # Input parameter is ocean_albedo with the same size as the global/full grid (360x180).
        # This could be the ocean albedo assuming no ice and can be the output
        # from the OSA (ocean surface albedo) calculations.
        # In addition, snow and ice parameters needed are:
        # ice_thickness, snow_thickness,sea_ice_concentration
        #
        self.shortwave = 'cesm3'  # shortwave
        self.albedo_type = 'cesm3'  # albedo parameterization, 'default'('cesm3')

    def calculate_bare_ice_albedo(self, osa, fh, albicev, ice_thickness, snow_concentration, puny):
        """
        Calculate the bare ice albedo based on the given parameters.

        Args:
            osa (float): The optical surface albedo.
            fh (float): The fraction of fresh snow.
            albicev (float): The albedo of the ice surface.
            ice_thickness (float): The thickness of the ice.
            snow_concentration (float): The concentration of snow on the ice surface.
            puny (float): The threshold for snow concentration.

        Returns:
            xr.DataArray: The calculated bare ice albedo.

        """
        return xr.where((ice_thickness >= 0) & (snow_concentration < puny), osa * (1 - fh) + albicev * fh, osa)

    def calculate_temperature_dependence(self, air_temp, Td, Tp, dalb_mlt):
        """
        Args:
            air_temp: The air temperature.
            Td: The upper threshold temperature for air temperature. If the air temperature is above this threshold,
            it will be set to the threshold.
            Tp: The lower threshold temperature for air temperature. If the air temperature is below this threshold,
            it will be set to the threshold.
            dalb_mlt: The temperature dependence coefficient.

        Returns:
            The calculated temperature dependence.

        Example:
            >>> calculate_temperature_dependence(30, 25, 20, 0.5)
            2.5

        """
        air_temp = xr.where(air_temp > Td, Td, air_temp)
        air_temp = xr.where(air_temp < Tp, Tp, air_temp)
        return dalb_mlt * (air_temp - Tp)

    def apply_melt_conditions(self, ice_concentration, ice_thickness, snow_concentration, snow_thickness, albo_dr,
                              albo_melt):
        """
        Applies melt conditions based on the given parameters.

        Args:
            ice_concentration (float): The concentration of ice.
            ice_thickness (float): The thickness of ice.
            snow_concentration (float): The concentration of snow.
            snow_thickness (float): The thickness of snow.
            albo_dr (xarray.DataArray): The albedo for non-melting conditions.
            albo_melt (xarray.DataArray): The albedo for melting conditions.

        Returns:
            xarray.DataArray: The resulting albedo based on the melt conditions.
        """
        return xr.where(
            (ice_concentration > 0) & (ice_thickness > 0.1) & (snow_concentration < 0.01) & (snow_thickness < 0.01),
            albo_melt, albo_dr)

    # http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cice/ice_shortwave.F90.html#COMPUTE_ALBEDOS
    def direct_and_diffuse_albedo_from_snow_and_ice(self,
                                                    osa: np.ndarray,
                                                    snow_concentration: np.ndarray,
                                                    snow_thickness: np.ndarray,
                                                    ice_concentration: np.ndarray,
                                                    ice_thickness: np.ndarray,
                                                    air_temp: np.ndarray) -> np.ndarray:
        """
        Args:
            osa: An ndarray representing the open ocean surface albedo.
            snow_concentration: An ndarray representing the concentration of snow on the ice.
            snow_thickness: An ndarray representing the thickness of the snow on the ice.
            ice_concentration: An ndarray representing the concentration of ice in the ice-ocean system.
            ice_thickness: An ndarray representing the thickness of the ice.
            air_temp: An ndarray representing the air temperature.

        Returns:
            An ndarray representing the direct and diffuse albedo from the snow and ice.

        """
        ahmax = 0.5
        dalb_mlt = -0.075

        # http://www.cesm.ucar.edu/models/cesm3.0/csim/UsersGuide/ice_usrdoc/node22.html
        # Ebert, Elizabeth E., and Judith A. Curry. 1993. “An Intermediate One‐dimensional Thermodynamic Sea Ice
        # Model for Investigating Ice‐atmosphere Interactions.” Journal of Geophysical Research 98 (C6): 10085–109.
        albsnowv = 0.98  # Visible snow albedo (cesm3) https://www2.cesm.ucar.edu/models/ccsm2.0/csim/SciGuide/ice_scidoc.pdf
        albicev = 0.78  # Visible ice albedo (cesm3)

        snowpatch = 0.02  # https://journals.ametsoc.org/jcli/article/26/4/1355/34253/The-Sensitivity-of-the-Arctic-Ocean-Sea-Ice
        puny = 1.0e-11

        Td = 0.15
        Tp = -1.0  # Melting of ice range goes from -1 to 0
        dalb_mltv = -0.1

        # snow albedo - al(albedo)v/i(visual/near-infrared)dr/df(direct/diffuse)ni/ns(ice/snow)
        fhtan = np.arctan(5 * ahmax)

        # Bare ice, thickness dependence. When values are less than  0.5 then this function (fh)
        # takes effect. This gives an asymptotic function that is 0 for 0 ice and 1 for full ice thickness
        fh = np.where((np.arctan(ice_thickness * 5.0) / fhtan) >= 1.0, 1.0, np.arctan(ice_thickness * 5.0) / fhtan)

        # Set the ice albedo where we don't have open ocean albedo. This is for thick ice
        # and the values are modified according to thickness and melting ponds further down.
        # Björk, Göran, Christian Stranne, and Karin Borenäs. 2013.
        # “The Sensitivity of the Arctic Ocean Sea Ice Thickness and Its Dependence on the Surface
        # Albedo Parameterization.” Journal of Climate 26 (4): 1355–70.

        # Fraction of pure snow on ice and the albedo
        fs = snow_concentration / (snow_concentration + snowpatch)
        albo_dr = xr.where(snow_concentration > 0, osa * (1 - fs) + fs * albsnowv, osa)

        # Fraction of bare ice and the albedo
        albo_dr = self.calculate_bare_ice_albedo(albo_dr, fh, albicev, ice_thickness, snow_concentration, puny)

        # Calculate the relative melting conditions
        albo_melt = albo_dr + self.calculate_temperature_dependence(air_temp, Td, Tp, dalb_mlt)

        # Calculate the impact of meltponds on snow and ice and impacts on albedo
        albo_dr = self.apply_melt_conditions(ice_concentration, ice_thickness, snow_concentration, snow_thickness,
                                             albo_dr, albo_melt)
        # Sanity check
        albo_dr = xr.where(albo_dr < 0.05, osa, albo_dr)

        # Return the albedo
        return albo_dr


    def calc_snow_attenuation(self, dr, snow_thickness: np.ndarray, snow_attenuation: float):
        """
        Calculate attenuation from snow assuming a constant attenuation coefficient of 20 m-1
        :param dr: Incoming solar radiation after accounting for atmospheric and surface albedo
        """
        # snow_attenuation unit : m-1

        #total_snow = np.count_nonzero(np.where(snow_thickness > 0))
        #per = (total_snow / snow_thickness.size) * 100.
      
        #logging.info(f"[CMIP6_cesm3] Percentage of grid point snow cover {per}")
        #logging.info(f"[CMIP6_cesm3] Mean ice thickness {np.nanmean(snow_thickness):3.2f}")
        
        return dr * np.exp(snow_attenuation * (-snow_thickness))

    def calc_ice_attenuation(self, spectrum: str, dr: np.ndarray, ice_thickness: np.ndarray):
        """
        This method splits the total incoming UV or VIS light into its wavelength components defined by the
        relative energy within each wavelength this is done by calculating the total sum of all wavelengths within
        the spectrum bands and then dividing the wavelength fractions to the total.

        :param spectrum: UV or VIS
        :param dr: The UV or VIS fraction of total incoming solar radiation
        :param ice_thickness: ice thickness on 2D array
        :return: Total irradiance after absorption through ice has been removed
        """
        logging.debug(f"[CMIP6_cesm3] calc_ice_attenuation started for spectrum {spectrum}")

        if spectrum == "uv":
            start_index = self.config.start_index_uv
            end_index = self.config.end_index_uv

        elif spectrum == "vis":
            start_index = self.config.start_index_visible
            end_index = self.config.end_index_visible

        else:
            raise Exception("f[CMIP6_cesm3] No valid spectrum defined ({spectrum})")

        # Calculate the effect for individual wavelength bands
        attenuation = self.config.absorption_ice_pg[start_index:end_index]
        dr_final = np.empty(np.shape(dr))

        for i in range(len(dr_final[:, 0, 0])):
            dr_final[i, :, :] = np.squeeze(dr[i, :, :]) * np.exp(attenuation[i] * (-ice_thickness))

      #  total_ice = np.count_nonzero(np.where(ice_thickness > 0))
      #  per = (total_ice / ice_thickness.size) * 100.

     #   logging.info("[CMIP6_cesm3] Sea-ice attenuation ranges from {:3.3f} to {:3.3f}".format(np.nanmin(attenuation),
                                                                                       np.nanmax(attenuation)))
     #   logging.info("[CMIP6_cesm3] Mean {} SW {:3.2f} in ice covered cells".format(spectrum, np.nanmean(dr_final)))
     #   logging.info("[CMIP6_cesm3] Percentage of grid point ice cover {}".format(per))
     #   logging.info("[CMIP6_cesm3] Mean ice thickness {:3.2f}".format(np.nanmean(ice_thickness)))

        return dr_final

    def effect_of_ozone_on_uv_at_wavelength(self, dr, ozone, wavelength_i):

        dr_ozone = dr * np.exp(-self.o3_abs[wavelength_i] * ozone)
        return dr_ozone

    def calculate_uvi(self, direct_sw_uv, ozone, wavelengths):
        """
        https://www.uio.no/studier/emner/matnat/fys/nedlagte-emner/FYS3610/h08/undervisningsmateriale/compendium/Ozone_and_UV_2008.pdf

        :param direct_sw_albedo_ice_snow_corrected_uv: irradiance in the upper part of the water column after adjusting for ice, snow, waves, chl etc.
        :return: UVI index between 0-11
        """
        # http://uv.biospherical.com/Solar_Index_Guide.pdf
        # https://stackoverflow.com/questions/65111670/use-total-irradiance-to-calculate-uv-index/65112111?noredirect=1#comment115117591_65112111

        uvi_wave = np.zeros((len(wavelengths),
                             len(direct_sw_uv[0, :, 0]),
                             len(direct_sw_uv[0, 0, :])))

        # Calculate the shortwave radiation per wavelength for UV
        scale_factor = 40.  # (/Wm-2)
        for wavelength_i, d in enumerate(wavelengths):
            # Calculate the effect of ozone on attenuation
            uvi_ozone = self.effect_of_ozone_on_uv_at_wavelength(direct_sw_uv[wavelength_i, :, :],
                                                                 ozone, wavelength_i)
            # Weight per wavelength based on the erythema spectrum
            uvi_wave[wavelength_i, :, :] = uvi_ozone * self.config.erythema_spectrum[wavelength_i]

        return np.squeeze(np.trapz(y=uvi_wave,
                                   x=wavelengths, axis=0) * scale_factor)

    #   return np.sum(uvi_wave, axis=0) * scale_factor

    # absorbed_solar - shortwave radiation absorbed by ice, ocean
    # Compute solar radiation absorbed in ice and penetrating to ocean
    def compute_surface_solar_for_specific_wavelength_band(self,
                                                           osa: np.ndarray,
                                                           direct_sw: np.ndarray,
                                                           diffuse_sw: np.ndarray,
                                                           chl: np.ndarray,
                                                           snow_thickness: np.ndarray,
                                                           ice_thickness: np.ndarray,
                                                           snow_attenuation: float,
                                                           spectrum: str) -> np.ndarray:

        # Before calling this method you need to initialize CMIP6_cesm3 with the OSA albedo array from the
        # wavelength band of interest:
        # wavelength_band_name:
        # OSA_uv, OSA_vis, OSA_nir, OSA_full
        # For OSA_uv and OSA_vis we just use the output of alvdfn and alvdrn as identical  but with different fraction
        # of energy component in total energy.
        # is_ = ice-snow

        # Effect of snow and ice
        # Albedo from snow and ice - direct where sea ice concentration is above zero
        sw_ice_ocean = (direct_sw + diffuse_sw) * (1.0 - osa)

        #  The effect of snow on attenuation
        sw_attenuation_corrected_for_snow = np.where(snow_thickness > 0,
                                                     self.calc_snow_attenuation(sw_ice_ocean, snow_thickness,
                                                                                snow_attenuation),
                                                     sw_ice_ocean)  
        
        # The wavelength dependent effect of ice on attenuation
        sw_attenuation_corrected_for_snow_and_ice = np.where(ice_thickness > 0,
                                                             self.calc_ice_attenuation(spectrum,
                                                                                       sw_attenuation_corrected_for_snow,
                                                                                       ice_thickness),
                                                             sw_attenuation_corrected_for_snow)

        if spectrum == "uv":
            return sw_attenuation_corrected_for_snow_and_ice

        # Account for the chlorophyll abundance and effect on attenuation of visible light
        return self.calculate_chl_attenuated_shortwave(sw_attenuation_corrected_for_snow_and_ice, chl)

    def calculate_chl_attenuated_shortwave(self, dr: np.ndarray, chl: np.ndarray, depth: float = 0.1):
        """
        Following Matsuoka et al. 2007 as defined in Table 3.

        This method splits the total incoming VIS light into its wavelength components defined by the
        relative energy within each wavelength this is done by calculating the total sum of all wavelengths within
        the VIS band (400-700) and then dividing the wavelength fractions by the total.

        chlorophyll values in kg/m-3 but need to be converted to mg/m-3

        :param dr: Incoming solar radiation after accounting for atmospheric and surface albedo, attenuation
         and albedo from ice and snow.
        :param chl: chlorophyll values in kg/m-3 as 2d array
        :param depth: we use a constant depth of 0.1 m unless otherwise stated
        :return: Visible solar radiation reaching 0.1 m into the water column past ice, snow and chlorophyll
        """
        kg2mg = 1.e6
        # Divide total incoming irradiance on number of wavelength segments,
        # then iterate the absorption effect for each wavelength and calculate total
        # irradiance absorbed by chl.
        #  dr_wave = np.zeros((len(self.config.fractions_shortwave_vis), len(dr[:, 0]), len(dr[0, :])))
        #  for i, d in enumerate(dr_wave):
        #      dr_wave[i, :, :] = dr[:, :] * (
        #                  self.config.fractions_shortwave_vis[i] / np.sum(self.config.fractions_shortwave_vis))

        logging.debug(
            f"[CMIP6_cesm3] {len(self.config.fractions_shortwave_vis)} segments to integrate for effect of wavelength on attenuation by chl")

        # Convert the units of chlorophyll to mgm-3
        chl = chl * kg2mg
        dr_chl_integrated = np.zeros(np.shape(dr))

        # Integrate over all wavelengths and calculate total absorption and
        # return the final light values
        assert self.chl_abs_wavelength[1] - self.chl_abs_wavelength[
            0] == 10, "The wavelengths need to be split into 10 nm consistently"
        for i_wave, x in enumerate(self.chl_abs_wavelength):
            dr_chl_integrated[i_wave, :, :] = np.nan_to_num(dr[i_wave, :, :]) * np.exp(
                -depth * self.chl_abs_A[i_wave] * chl ** self.chl_abs_B[i_wave])
            dr_chl_integrated[i_wave, :, :] = np.where(np.isnan(dr_chl_integrated[i_wave, :, :]), dr[i_wave, :, :],
                                                       dr_chl_integrated[i_wave, :, :])

        return dr_chl_integrated

    def calculate_albedo_in_mixed_snow_ice_grid_cell(self, sisnconc, siconc, albicev, albsnowv):
        return (1. - siconc) * 0.06 + siconc * ((1. - sisnconc) * albicev + sisnconc * albsnowv)

    def calculate_diffuse_albedo_per_grid_point(self, sisnconc: np.ndarray,
                                                siconc: np.ndarray) -> np.ndarray:
        """
        Routine for  getting a crude estimate of the albedo based on ocean, snow, and ice values.
        The result is used by pvlib to calculate the  initial diffuse irradiance.
        :param sisnconc:
        :param siconc:
        :return: albedo (preliminary version used for pvlib)
        """
        albicev = 0.78  # Visible ice albedo (cesm3)
        albsnowv = 0.98  # Visible snow albedo (cesm3)

        albedo = np.zeros(np.shape(sisnconc)) + 0.06
        ice_alb = np.where(siconc > 0,
                           self.calculate_albedo_in_mixed_snow_ice_grid_cell(sisnconc, siconc, albicev, albsnowv),
                           albedo)
        albedo[~np.isnan(ice_alb)] = ice_alb[~np.isnan(ice_alb)]
        return albedo
