import logging
import sys
from typing import Tuple
import numpy as np
import xarray as xr
from CMIP6_config import Config_albedo

# Class for calculating albedo of sea-ice, snow and snow-ponds
# The albedo and absorbed/transmitted flux parameterizations for
# snow over ice, bare ice and ponded ice.
# Methods applied from CCSM3 online code :
# http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cice/ice_shortwave.F90.html#COMPUTE_ALBEDOS
class CMIP6_CCSM3():

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
        self.shortwave = 'ccsm3'  # shortwave
        self.albedo_type = 'ccsm3'  # albedo parameterization, 'default'('ccsm3')

    # http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cice/ice_shortwave.F90.html#COMPUTE_ALBEDOS
    def direct_and_diffuse_albedo_from_snow_and_ice(self,
                                                    osa: np.ndarray,
                                                    snow_concentration: np.ndarray,
                                                    snow_thickness: np.ndarray,
                                                    ice_concentration: np.ndarray,
                                                    ice_thickness: np.ndarray,
                                                    air_temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Calculate albedo for each grid point taking into account snow and ice thickness. Also,
        accounts for meltponds and their effect on albedo. Based on CICE5 shortwave
        albedo calculations:
        https://github.com/CICE-Consortium/CICE-svn-trunk/blob/7d9cff9d8dbabf6d4947e388a6d98c870c808536/cice/source/ice_shortwave.F90
        Units:
        air_temp # Celsius
        ice_thickness  # meter
        snow_thickness # meter
        snow_concentration  # fraction

        Returns:
        albo_dr = albedo visual direct
        :return: alvdfn, alvdrn
        """
        ahmax = 0.5
        dT_mlt = 1.0
        dalb_mlt = -0.075
        dalb_mltv = -0.1
        dalb_mlti = -0.15
        # http://www.cesm.ucar.edu/models/ccsm3.0/csim/UsersGuide/ice_usrdoc/node22.html
        albicev = 0.73  # Visible ice albedo (CCSM3)
        albsnowv = 0.96  # Visible snow albedo (CCSM3)

        snowpatch = 0.02  # https://journals.ametsoc.org/jcli/article/26/4/1355/34253/The-Sensitivity-of-the-Arctic-Ocean-Sea-Ice
        puny = 1.0e-11
        Timelt = 0.0  # melting temp.ice top surface(C)

        # snow albedo - al(albedo)v/i(visual/near-infrared)dr/df(direct/diffuse)ni/ns(ice/snow)
        # hi = ice height
        fhtan = np.arctan(ahmax * 4.0)

        # bare ice, thickness dependence
        fh = np.where((np.arctan(ice_thickness * 4.0) / fhtan) >= 1.0, 1.0, np.arctan(ice_thickness * 4.0) / fhtan)
     
        albo_dr = np.where(np.isnan(osa), albicev, osa)
        albo = albo_dr * (1.0 - fh)
       
        albo_i = albicev * fh + albo
        albo_dr = np.where(ice_thickness > 0, albo_i, albo_dr)

        # bare ice, temperature dependence (where snow is zero). Affects albedo when
        # temperature is between -1 and 0.
        dTs = xr.where(air_temp > -1, 1.0, 0.0)
        fT = -np.min(dTs/dT_mlt, 0)

        # Account for melting ponds (only if ice thicker than 0.1 m) and
        # snow patches. Scale by ice and snow concentrations.
        albo_dr = np.where((ice_thickness >= 0.1) & (snow_thickness < puny), (albo_i - dalb_mlt * fT)*ice_concentration, albo_dr)
        albo_dr = np.where(snow_thickness > snowpatch, albsnowv* snow_concentration, albo_dr)
        
        # Avoid very low values where melting too strong
        albo_dr = np.where(albo_dr < snowpatch, osa, albo_dr)

        # avoid negative albedo for thin, bare, melting ice
        albo_dr = np.where(albo_dr > 0, albo_dr, osa)

        return np.where(np.isnan(albo_dr), np.nanmean(osa), albo_dr)


    def calc_snow_attenuation(self, dr, snow_thickness: np.ndarray):
        """
        Calculate attenuation from snow assuming a constant attenuation coefficient of 20 m-1
        :param dr: Incoming solar radiation after accounting for atmospheric and surface albedo
        """
        attenuation_snow = 20  # unit : m-1

        total_snow = np.count_nonzero(np.where(snow_thickness > 0))
        per = (total_snow / snow_thickness.size) * 100.

        return dr * np.exp(attenuation_snow * (-snow_thickness))

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
        logging.debug(f"[CMIP6_ccsm3] calc_ice_attenuation started for spectrum {spectrum}")

        if spectrum == "uv":
            start_index = self.config.start_index_uv
            end_index = self.config.end_index_uv

        elif spectrum == "vis":
            start_index = self.config.start_index_visible
            end_index = self.config.end_index_visible

        else:
            raise Exception("f[CMIP6_ccsm3] No valid spectrum defined ({spectrum})")

        #   dr = dr[start_index:end_index, :, :]
        # Calculate the effect for individual wavelength bands
        attenuation = self.config.absorption_ice_pg[start_index:end_index]
        dr_final = np.empty(np.shape(dr))

        for i in range(len(dr_final[:, 0, 0])):
            dr_final[i, :, :] = np.squeeze(dr[i, :, :]) * np.exp(attenuation[i] * (-ice_thickness))

        total_ice = np.count_nonzero(np.where(ice_thickness > 0))
      #  per = (total_ice / ice_thickness.size) * 100.

     #   logging.info("[CMIP6_ccsm3] Sea-ice attenuation ranges from {:3.3f} to {:3.3f}".format(np.nanmin(attenuation),
      #                                                                               np.nanmax(attenuation)))
      #  logging.info("[CMIP6_ccsm3] Mean {} SW {:3.2f} in ice covered cells".format(spectrum, np.nanmean(dr_final)))
      #  logging.info("[CMIP6_ccsm3] Percentage of grid point ice cover {}".format(per))
      #  logging.info("[CMIP6_ccsm3] Mean ice thickness {:3.2f}".format(np.nanmean(ice_thickness)))

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
            uvi_ozone = self.effect_of_ozone_on_uv_at_wavelength(direct_sw_uv[wavelength_i,:,:],
                                                          ozone, wavelength_i)
            # Weight per wavelength based on the erythema spectrum
            uvi_wave[wavelength_i,:,:] = uvi_ozone * self.config.erythema_spectrum[wavelength_i]

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
                                                           sea_ice_concentration: np.ndarray,
                                                           snow_concentration: np.ndarray,
                                                           air_temp: np.ndarray,
                                                           lon: np.ndarray,
                                                           lat: np.ndarray,
                                                           model_object,
                                                           spectrum: str) -> np.ndarray:

        # Before calling this method you need to initialize CMIP6_CCSM3 with the OSA albedo array from the
        # wavelength band of interest:
        # wavelength_band_name:
        # OSA_uv, OSA_vis, OSA_nir, OSA_full
        # For OSA_uv and OSA_vis we just use the output of alvdfn and alvdrn as identical  but with different fraction
        # of energy component in total energy.
        # is_ = ice-snow

        # Effect of snow and ice
        # Albedo from snow and ice - direct where sea ice concentration is above zero
        sw_ice_ocean = (direct_sw + diffuse_sw)* (1.0 - osa)

        #  The effect of snow on attenuation
        sw_attenuation_corrected_for_snow = np.where(snow_thickness > 0,
                                            self.calc_snow_attenuation(sw_ice_ocean, snow_thickness),
                                            sw_ice_ocean)

        # The wavelength dependent effect of ice on attenuation
      
        sw_attenuation_corrected_for_snow_and_ice = np.where(ice_thickness > 0,
                                           self.calc_ice_attenuation(spectrum, sw_attenuation_corrected_for_snow,
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
            f"[CMIP6_ccsm3] {len(self.config.fractions_shortwave_vis)} segments to integrate for effect of wavelength on attenuation by chl")

        # Convert the units of chlorophyll to mgm-3
        chl = chl * kg2mg
        dr_chl_integrated = np.zeros(np.shape(dr))

        # Integrate over all wavelengths and calculate total absorption and
        # return the final light values
        assert self.chl_abs_wavelength[1]-self.chl_abs_wavelength[0]==10, "The wavelengths need to be split into 10 nm consistently"
        for i_wave, x in enumerate(self.chl_abs_wavelength):
            dr_chl_integrated[i_wave,:,:] = np.nan_to_num(dr[i_wave, :, :]) * np.exp(
                -depth * self.chl_abs_A[i_wave] * chl ** self.chl_abs_B[i_wave])
            dr_chl_integrated[i_wave,:,:] = np.where(np.isnan(dr_chl_integrated[i_wave,:,:]), dr[i_wave,:,:], dr_chl_integrated[i_wave,:,:])

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
        albicev = 0.73  # Visible ice albedo (CCSM3)
        albsnowv = 0.96  # Visible snow albedo (CCSM3)

        albedo = np.zeros(np.shape(sisnconc)) + 0.06
        ice_alb = np.where(siconc > 0,
                           self.calculate_albedo_in_mixed_snow_ice_grid_cell(sisnconc, siconc, albicev, albsnowv),
                           albedo)
        albedo[~np.isnan(ice_alb)] = ice_alb[~np.isnan(ice_alb)]
        return albedo
