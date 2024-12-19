import datetime
import logging
import os
from calendar import monthrange
from re import T
from typing import List, Any
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dask.distributed import Client
from pathlib import Path

# Computational modules
import cftime
import dask
import netCDF4
import numpy as np
import pandas as pd
from platformdirs import user_cache_dir
import pvlib
#import pvlib.forecast
import texttable
import xarray as xr
import xesmf as xe
from google.cloud import storage
import logging
import CMIP6_IO
import CMIP6_albedo_plot
import CMIP6_albedo_utils
import CMIP6_cesm3
import CMIP6_config
import CMIP6_date_tools
import CMIP6_regrid

from numpy.typing import NDArray 

class CMIP6_light:
    def __init__(self, args):
        np.seterr(divide="ignore")
        self.config = CMIP6_config.Config_albedo()
        self.config.source_id = args.source_id
        self.config.member_id = args.member_id
        self.show_table_of_info=False
        
        if not self.config.use_local_CMIP6_files:
            self.config.read_cmip6_repository()
        self.cmip6_models: List[Any] = []

    # Required setup for doing light calculations, but only required once per timestep.
    def setup_pv_system(self, current_time):
        offset = 0  # int(lon_180/15.)
        when = [
            datetime.datetime(
                current_time.year,
                current_time.month,
                current_time.day,
                current_time.hour,
                0,
                0,
                tzinfo=datetime.timezone(datetime.timedelta(hours=offset)),
            )
        ]
        time = pd.DatetimeIndex(when)

        sandia_modules = pvlib.pvsystem.retrieve_sam("SandiaMod")
        sapm_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

        module = sandia_modules["Canadian_Solar_CS5P_220M___2009_"]
        inverter = sapm_inverters["ABB__MICRO_0_25_I_OUTD_US_208__208V_"]
        pv_system = {"module": module, "inverter": inverter, "surface_azimuth": 180}

        return time, pv_system

    def calculate_zenith(self, latitude, ctime):
        longitude = 0.0
        # get_solar-position returns a Pandas dataframe with index so we convert the value
        # to numpy after calculating
        solpos = pvlib.solarposition.get_solarposition(ctime, latitude, longitude)
        return np.squeeze(solpos["zenith"])

    def cloud_opacity_factor(self, I_diff_clouds, I_dir_clouds, I_ghi_clouds, spectra):
        # First we calculate the rho fraction based on campbell_norman irradiance
        # with clouds converted to POA irradiance. In the paper these
        # values are obtained from observations. The equations used for calculating cloud opacity factor
        # to scale the clear sky spectral estimates using spectrl2. Results can be compared with sun calculator:
        # https://www2.pvlighthouse.com.au/calculators/solar%20spectrum%20calculator/solar%20spectrum%20calculator.aspx
        #
        # Ref: Marco Ernst, Hendrik Holst, Matthias Winter, Pietro P. Altermatt,
        # SunCalculator: A program to calculate the angular and spectral distribution of direct and diffuse solar radiation,
        # Solar Energy Materials and Solar Cells, Volume 157, 2016, Pages 913-922,
        np.seterr(divide="ignore", invalid="ignore")
        rho = I_diff_clouds / I_ghi_clouds
     
        I_diff_s = np.trapezoid(
            y=spectra["poa_sky_diffuse"], x=spectra["wavelength"], axis=0
        ) + np.trapezoid(y=spectra["poa_ground_diffuse"], x=spectra["wavelength"], axis=0)
        
        I_dir_s = np.trapezoid(y=spectra["poa_direct"], x=spectra["wavelength"], axis=0)
        I_glob_s = np.trapezoid(y=spectra["poa_global"], x=spectra["wavelength"], axis=0)

        rho_spectra = I_diff_s / I_glob_s

        N_rho = (rho - rho_spectra) / (1 - rho_spectra)
  
        # Direct light. Equation 6 Ernst et al. 2016
        F_diff_s = (
            spectra["poa_sky_diffuse"][:, :] + spectra["poa_ground_diffuse"][:, :]
        )
        F_dir_s = spectra["poa_direct"][:, :]
        F_dir = np.multiply(F_dir_s / I_dir_s[None, :], I_dir_clouds[None, :])

        # Diffuse light scaling factor. Equation 7 Ernst et al. 2016
        s_diff = (1 - N_rho[None, :]) * (F_diff_s / I_diff_s[None, :]) + N_rho[
            None, :
        ] * ((F_dir_s + F_diff_s) / I_glob_s[None, :])

        # Equation 8 Ernst et al. 2016
        F_diff = s_diff * I_diff_clouds[None, :]
        return F_dir, F_diff

    def calculate_transmittance(self, cloud_cover):
        """
        Calculates the transmittance based on cloud covers using the equation from the paper:
        Srivastava, Ankur, Jose F. Rodriguez, Patricia M. Saco, Nikul Kumari, and Omer Yetemen. 2021.
        “Global Analysis of Atmospheric Transmissivity Using Cloud Cover, Aridity and Flux Network Datasets.”
        Remote Sensing 13 (9): 1716.

        Args:
            cloud_covers (np.array): numpy array of cloud cover for longitude band (fixed latitude)
            Cloud covers is assumed to be in fractions and converted to percentage multiplying by 100.

        Returns:
            np.array: array containing the calculated transmittance values
        """
        offset = 0.75
        return (1.0 - cloud_cover) * offset
    
    def calculate_irradiance_with_clouds(self, zenith, cloud_covers, dni_extra):
        transmittance = self.calculate_transmittance(cloud_covers)
        return pvlib.irradiance.campbell_norman(
            zenith, transmittance, dni_extra=dni_extra
        )
    def fix_bad_values(self, zenith, ghi: np.ndarray, dni = None):
        # Fix potential problems with the calculated values
        if dni is not None:
            bad_values = (
                (zenith > 87)
                | (ghi < 0)
                | (dni < 0)
                | (np.isnan(ghi))
            )
            dni = np.where(bad_values, 0, dni)
        else:
            bad_values = (
                (zenith > 87)
                | (ghi < 0)
                | (np.isnan(ghi))
            )
            
        ghi = np.where(bad_values, 0, ghi)
        return ghi, dni 
        
    def radiation(
        self,
        cloud_covers,
        water_vapor_content,
        latitude,
        ctime,
        system,
        albedo,
        ozone,
        altitude=0.0,
    ) -> NDArray:
        """Returns an array of calculated diffuse and direct light for each longitude index
        around the globe (fixed latitude). Output has the shape:  [len(wavelengths), 361, 3] and
        the indexes refer to:
        0: len(wavelengths)
        1: longitudes global
        2: [direct light, diffuse light]
        Args:
            cloud_covers (np.array): numpy array of cloudcover for longitude band (fixed latitude)
            latitude (float): latitude
            ctime (pd.DatetimeIndex): datetime for when light will be calculated
            system (json): Return from setup_pv_system
            albedo (np.array): numpy array of albedo for longitude band (fixed latitude)
            ozone (np.array): numpy array of ozone for longitude band (fixed latitude)

        Returns:
            np.array: array containing diffuse and direct light, and zenith for each wavelength and longitude
            but remember units are W/m2/nm so you have to integrate across wavelengths to get irradiance
        """
        wavelengths = np.arange(200, 2700, 10)
        results = np.zeros((len(wavelengths), np.shape(cloud_covers)[0], 3))
        
        # Some calculations are done only on Greenwhich meridian line as they are identical around the globe at the
        # same latitude. For that reason longitude is set to Greenwhich meridian and do not change. The only reason
        # to use longitude would be to have the local sun position for given time but since we calculate position at
        # the same time of the day (hour_of_day) and month (month) we can assume its the same across all longitudes,
        # and only change with latitude.

        longitude = 0.0
        # get_solar-position returns a Pandas dataframe with index so we convert the value
        # to numpy after calculating

        solpos = pvlib.solarposition.get_solarposition(
            ctime, latitude, longitude, altitude=altitude
        )

        airmass_relative = pvlib.atmosphere.get_relative_airmass(
            solpos["apparent_zenith"].to_numpy(), model="kastenyoung1989"
        )
        pressure = pvlib.atmosphere.alt2pres(altitude)
        apparent_zenith = solpos["apparent_zenith"].to_numpy()
        zenith = solpos["zenith"].to_numpy()
        azimuth = solpos["azimuth"].to_numpy()
        surface_azimuth = system["surface_azimuth"]

        shape = np.shape(cloud_covers)
        apparent_zenith = np.broadcast_to(apparent_zenith, shape)
        zenith = np.broadcast_to(zenith, shape)
        azimuth = np.broadcast_to(azimuth, shape)
        surface_azimuth = np.broadcast_to(surface_azimuth, shape)
        surface_tilt = np.zeros(shape)

        aoi = pvlib.irradiance.aoi(
            surface_tilt, surface_azimuth, apparent_zenith, azimuth
        )

        # Fixed atmospheric components used from pvlib example
        tau500 = np.ones((shape)) * 0.1

        # day of year is an int64index array so access first item
        day_of_year = ctime.dayofyear
        day_of_year = np.ones((shape)) * day_of_year[0]

        """
        The radiation reaching the earth's surface can be represented in a number of different ways. Global Horizontal 
        Irradiance (GHI) is the total amount of shortwave radiation received from above by a surface horizontal 
        to the ground. This value is of particular interest to photovoltaic installations and includes both Direct 
        Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DIF).
        DNI is solar radiation that comes in a straight line from the direction of the sun at its current position in 
        the sky. DIF is solar radiation that does not arrive on a direct path from the sun, but has been scattered 
        by molecules and particles in the atmosphere and comes equally from all directions.
        """
        spectra = pvlib.spectrum.spectrl2(
            apparent_zenith=apparent_zenith,
            aoi=aoi,
            surface_tilt=surface_tilt,
            ground_albedo=albedo*0.0,
            surface_pressure=pressure,
            relative_airmass=airmass_relative,
            precipitable_water=water_vapor_content,
            ozone=ozone,
            aerosol_turbidity_500nm=tau500,
            dayofyear=day_of_year,
        )
        dni_extra = pvlib.irradiance.get_extra_radiation(day_of_year) 

        # Select methodology to calculate irradiance with clouds
        use_CAMPBELL_irradiance = False
        use_DISC_irradiance = not use_CAMPBELL_irradiance

        if use_CAMPBELL_irradiance:        
            irrad = self.calculate_irradiance_with_clouds(zenith, cloud_covers, dni_extra)
            ghi=irrad["ghi"]
            dni=irrad["dni"]
            dhi=irrad["dhi"]
            ghi, dni = self.fix_bad_values(zenith, ghi, dni=dni)

        elif use_DISC_irradiance:
            """
            Estimate Direct Normal Irradiance from Global Horizontal Irradiance
            using the DISC model.

            The DISC algorithm converts global horizontal irradiance to direct
            normal irradiance through empirical relationships between the global
            and direct clearness indices.

            The pvlib implementation limits the clearness index to 1.
            """
            transmittance = self.calculate_transmittance(cloud_covers)
            irrads_clouds = pvlib.irradiance.campbell_norman(
                apparent_zenith, transmittance, dni_extra = dni_extra
            ) 
            
            ghi = irrads_clouds["ghi"]
            ghi, dni = self.fix_bad_values(zenith, ghi, dni=None)
           
            kt = pvlib.irradiance.clearness_index(
                ghi,
                zenith,
                dni_extra,
                min_cos_zenith=0.065,
                max_clearness_index=0.9,
            )

         #   am = pvlib.atmosphere.get_absolute_airmass(airmass_relative, pressure)

           # Kn, am = pvlib.irradiance._disc_kn(kt, am, max_airmass=12)
            irrads = pvlib.irradiance.disc(ghi, zenith, day_of_year, max_airmass=12)
           # dni = Kn * dni_extra
            dni = irrads["dni"]
        
           # dni = pvlib.irradiance.dni(ghi, dni, zenith, clearsky_dni=None)
            
            dhi = ghi - dni * np.cos(np.radians(zenith))
            ghi, dni = self.fix_bad_values(zenith, ghi, dni=dni)
            
        # Convert the irradiance to a plane with tilt zero horizontal to the earth. This is done applying tilt=0 to POA
        # calculations using the output from campbell_norman. The POA calculations include calculating sky and ground
        # diffuse light where specific models can be selected (we use default). Here, albedo is used to calculate
        # ground diffuse irradiance.

        POA_irradiance_clouds = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dni=dni, 
            ghi=ghi,
            dhi=dhi,
            solar_zenith=apparent_zenith,
            solar_azimuth=azimuth,
            albedo=albedo,
            dni_extra=dni_extra, 
            model="haydavies")

       
        # Account for cloud opacity on the spectral radiation
        F_dir, F_diff = self.cloud_opacity_factor(POA_irradiance_clouds["poa_diffuse"]\
                                                    +POA_irradiance_clouds["poa_ground_diffuse"]\
                                                    +POA_irradiance_clouds["poa_sky_diffuse"],
            POA_irradiance_clouds["poa_direct"],
            POA_irradiance_clouds["poa_global"],
            spectra,
        )
    
        # Do the linear interpolation
        for lon_index in range(len(F_dir[0, :])):
            interp_fdir = np.interp(
                wavelengths, spectra["wavelength"], F_dir[:, lon_index]
            )
            interp_fdiff = np.interp(
                wavelengths, spectra["wavelength"], F_diff[:, lon_index]
            )
            results[:, lon_index, 0] = np.squeeze(interp_fdir)
            results[:, lon_index, 1] = np.squeeze(interp_fdiff)
        
        results[0, :, 2] = ghi

        # return direct, diffuse, and ghi cloud affected radiation at wavelengths
        return results

    
    """
    Regrid to cartesian grid:
    For any Amon related variables (wind, clouds), the resolution from CMIP6 models is less than
    1 degree longitude x latitude. To interpolate to a 1x1 degree grid we therefore first interpolate to a
    2x2 degrees grid and then subsequently to a 1x1 degree grid.
    """

    def extract_dataset_and_regrid(self, model_obj, t_index):
        extracted: dict = {}
        if self.config.use_local_CMIP6_files:
            for key in model_obj.ds_sets[model_obj.current_member_id].keys():
                extracted[key] = (
                    model_obj.ds_sets[model_obj.current_member_id][key]
                    .isel(time=int(t_index))
                    .to_array()
                )
            return extracted

        ds_out_amon = xe.util.grid_2d(
            self.config.min_lon,
            self.config.max_lon,
            2,
            self.config.min_lat,
            self.config.max_lat,
            2,
        )
        ds_out = xe.util.grid_2d(
            self.config.min_lon,
            self.config.max_lon,
            1,
            self.config.min_lat,
            self.config.max_lat,
            1,
        )

        re = CMIP6_regrid.CMIP6_regrid()
        for key in model_obj.ds_sets[model_obj.current_member_id].keys():
            current_ds = (
                model_obj.ds_sets[model_obj.current_member_id][key]
                .isel(time=int(t_index))
                .sel(
                    y=slice(int(self.config.min_lat), int(self.config.max_lat)),
                    x=slice(int(self.config.min_lon), int(self.config.max_lon)),
                )
            )

            if key in ["uas", "vas"]:
                out_amon = re.regrid_variable(
                    key,
                    current_ds,
                    ds_out_amon,
                    interpolation_method=self.config.interp,
                ).to_dataset()

                out = re.regrid_variable(
                    key, out_amon, ds_out, interpolation_method=self.config.interp
                )

            else:
                out = re.regrid_variable(
                    key, current_ds, ds_out, interpolation_method=self.config.interp
                )
            extracted[key] = out
        return extracted

    def filter_extremes(self, df):
        return np.where(((df < -1000) | (df > 1000)), np.nan, df)

    def values_for_timestep(self, extracted_ds, model_object):
        lat = np.squeeze(extracted_ds["tos"].lat.values)
        lon = np.squeeze(extracted_ds["tos"].lon.values)
        chl = np.squeeze(extracted_ds["chl"].values)
        sisnconc = np.squeeze(extracted_ds["sisnconc"].values)
        sisnthick = np.squeeze(extracted_ds["sisnthick"].values)
        siconc = np.squeeze(extracted_ds["siconc"].values)
        sithick = np.squeeze(extracted_ds["sithick"].values)
        uas = np.squeeze(extracted_ds["uas"].values)
        vas = np.squeeze(extracted_ds["vas"].values)
        clt = np.squeeze(extracted_ds["clt"].values)
        prw = np.squeeze(extracted_ds["prw"].values)
        tas = np.squeeze(extracted_ds["tas"].values)

        clt = self.filter_extremes(clt)
        chl = self.filter_extremes(chl)
        uas = self.filter_extremes(uas)
        vas = self.filter_extremes(vas)
        sisnconc = self.filter_extremes(sisnconc)
        sisnthick = self.filter_extremes(sisnthick)
        siconc = self.filter_extremes(siconc)
        sithick = self.filter_extremes(sithick)
        tas = self.filter_extremes(tas)
        prw = self.filter_extremes(prw)

        percentage_to_ratio = 1.0 / 100.0

        if np.nanmax(sisnconc) > 5:
            sisnconc = sisnconc * percentage_to_ratio
        if np.nanmax(siconc) > 5:
            siconc = siconc * percentage_to_ratio
        if np.nanmax(clt) > 5:
            clt = clt * percentage_to_ratio

        # Calculate scalar wind and organize the data arrays to be used for  given time-step (month-year)
        wind = np.sqrt(uas**2 + vas**2)

        m = len(wind[:, 0])
        n = len(wind[0, :])

        # Set show_table_of_info to True to print the range of data read 
        # at each timestep (for debugging purposes)
        if self.show_table_of_info:
            table = texttable.Texttable()
            table.set_cols_align(["c", "c", "c"])
            table.set_cols_valign(["m", "m", "m"])

            table.header(["Model", "Variable", "Range"])
            table.add_rows([
                ["", "clt", "{:3.3f} to {:3.3f}".format(np.nanmin(clt), np.nanmax(clt))],
                ["", "chl (mg/m3)", "{:3.3f} to {:3.3f}".format(np.nanmin(chl)*1e6, np.nanmax(chl)*1e6)],
                ["", "prw", "{:3.3f} to {:3.3f}".format(np.nanmin(prw), np.nanmax(prw))],
                ["{}".format(model_object.name), "tas", "{:3.3f} to {:3.3f}".format(np.nanmin(tas), np.nanmax(tas))],
                ["", "uas", "{:3.3f} to {:3.3f}".format(np.nanmin(uas), np.nanmax(uas))],
                ["", "vas", "{:3.3f} to {:3.3f}".format(np.nanmin(vas), np.nanmax(vas))],
                ["", "wind", "{:3.3f} to {:3.3f}".format(np.nanmin(wind), np.nanmax(wind))],
                ["", "siconc", "{:3.3f} to {:3.3f}".format(np.nanmin(siconc), np.nanmax(siconc))],
                ["", "sithick", "{:3.3f} to {:3.3f}".format(np.nanmin(sithick), np.nanmax(sithick))],
                ["", "sisnconc", "{:3.3f} to {:3.3f}".format(np.nanmin(sisnconc), np.nanmax(sisnconc))],
                ["", "sisnthick", "{:3.3f} to {:3.3f}".format(np.nanmin(sisnthick), np.nanmax(sisnthick))]
            ])
            table.set_cols_width([30, 20, 30])
            print(table.draw() + "\n")

        return (
            wind,
            lat,
            lon,
            clt,
            chl,
            sisnconc,
            sisnthick,
            siconc,
            sithick,
            tas,
            prw,
            m,
            n,
        )

    def calculate_radiation(
        self,
        ctime,
        pv_system,
        clt: NDArray,
        prw: NDArray,
        ozone: NDArray,
        direct_OSA: NDArray,
        lat: NDArray,
        m: int,
        n: int
    ) -> (NDArray[float], NDArray[float], NDArray[float]):
        
        wavelengths = np.arange(200, 2700, 10)
        
        calc_radiation = [
            dask.delayed(self.radiation)(
                clt[j, :],
                prw[j, :],
                lat[j, 0],
                ctime,
                pv_system,
                direct_OSA[j, :],
                ozone[j, :],
            )
            for j in range(m)
        ]

        # https://github.com/dask/dask/issues/5464
        rad = dask.compute(calc_radiation)
        rads = np.squeeze(np.asarray(rad).reshape((m, len(wavelengths), n, 3)))

        # Transpose to get order: wavelengths, lat, lon, elements
        rads = np.transpose(rads, (1, 0, 2, 3))
       
        direct_sw = np.squeeze(rads[:, :, :, 0])
        diffuse_sw = np.squeeze(rads[:, :, :, 1])
        ghi = np.squeeze(rads[0, :, :, 2])

        return direct_sw, diffuse_sw, ghi

    def get_ozone_dataset(self, current_experiment_id: str) -> xr.Dataset:
        """
        Retrieves the total ozone column dataset for a given experiment ID and regrids it 
        to a consistent 1x1 degree dataset.

        Parameters:
        - current_experiment_id (str): The ID of the current experiment.

        Returns:
        - xr.Dataset: The regridded total ozone column dataset.
        """
        logging.info("[CMIP6_light] Regridding ozone data to standard grid")
        io = CMIP6_IO.CMIP6_IO()

        toz_name = f"light/{current_experiment_id}/TOZ_{current_experiment_id}.nc"
        toz_full = io.open_dataset_on_gs(toz_name)
        assert isinstance(
            toz_full, xr.Dataset
        ), "[CMIP6_light] Unable to open TOZ file: {toz_name}"

        toz_full = toz_full.sel(
            time=slice(self.config.start_date, self.config.end_date)
        ).sel(
            lat=slice(self.config.min_lat, self.config.max_lat),
            lon=slice(self.config.min_lon, self.config.max_lon),
        )

        re = CMIP6_regrid.CMIP6_regrid()
        ds_out = xe.util.grid_2d(
            self.config.min_lon,
            self.config.max_lon,
            1,
            self.config.min_lat,
            self.config.max_lat,
            1,
        )

        toz_ds = re.regrid_variable(
            "TOZ", toz_full, ds_out, interpolation_method=self.config.interp
        )

        return toz_ds

    def convert_dobson_units_to_atm_cm(self, ozone):
        """
        Converts ozone concentration from Dobson Units to atmospheric centimeters.

        One Dobson Unit is the number of molecules of ozone that would be required to create a layer
        of pure ozone 0.01 millimeters thick at a temperature of 0 degrees Celsius and a pressure of 1 atmosphere
        (the air pressure at the surface of the Earth). Expressed another way, a column of air with an ozone
        concentration of 1 Dobson Unit would contain about 2.69x10^16 ozone molecules for every
        square centimeter of area at the base of the column. Over the Earth’s surface, the ozone layer’s
        average thickness is about 300 Dobson Units or a layer that is 3 millimeters thick.

        Args:
            ozone (numpy.ndarray): Array of ozone concentrations in Dobson Units.

        Returns:
            numpy.ndarray: Array of ozone concentrations in atmospheric centimeters.
        """
        ozone = np.where(ozone == 0, np.nan, ozone)
        assert np.nanmax(ozone) <= 700
        assert np.nanmin(ozone) > 100
        ozone = ozone / 1000.0
        assert np.nanmin(ozone) <= 0.7
        assert np.nanmin(ozone) > 0
        return ozone

    def perform_light_calculations(self, model_object, current_experiment_id):
        io = CMIP6_IO.CMIP6_IO()

        times = model_object.ds_sets[model_object.current_member_id]["tos"].time
        self.CMIP6_cesm3 = CMIP6_cesm3.CMIP6_cesm3()

        toz_ds = self.get_ozone_dataset(current_experiment_id)
        time_counter = 0
        start_at_noon = False
        
        for selected_time in range(len(times.values)):
            sel_time = times.values[selected_time]
            if isinstance(sel_time, cftime._cftime.DatetimeNoLeap):
                sel_time = datetime.datetime(
                    year=sel_time.year, month=sel_time.month, day=sel_time.day
                )
            if times.dtype in ["datetime64[ns]"]:
                sel_time = pd.DatetimeIndex(
                    [sel_time], dtype="datetime64[ns]", name="datetime", freq=None
                ).to_pydatetime()[0]

            model_object.current_time = sel_time
            extracted_ds = self.extract_dataset_and_regrid(model_object, selected_time)

            (
                wind,
                lat,
                lon,
                clt,
                chl,
                sisnconc,
                sisnthick,
                siconc,
                sithick,
                tas,
                prw,
                m,
                n,
            ) = self.values_for_timestep(extracted_ds, model_object)

            ozone = self.convert_dobson_units_to_atm_cm(
                toz_ds["TOZ"][selected_time, :, :].values
            )

            # num_days = monthrange(sel_time.year, sel_time.month)[1]
            for day in [15]:  # range(num_days):
                for hour_of_day in range(0, 24, 6):
                    if hour_of_day == 12:
                        start_at_noon = True
                    if start_at_noon is True:
                        model_object.current_time = datetime.datetime(
                            year=sel_time.year,
                            month=sel_time.month,
                            day=day + 1,
                            hour=hour_of_day,
                        )
                        ctime, pv_system = self.setup_pv_system(
                                model_object.current_time
                            )
                         
                        if hour_of_day == 12:
                            calc_zenith = [
                                dask.delayed(self.calculate_zenith)(lat[j, 0], ctime)
                                for j in range(m)
                            ]
                        
                            zenith = dask.compute(calc_zenith)
                            zeniths = np.asarray(zenith).reshape(m)
                        
                        for scenario in self.config.scenarios:
                            if scenario == "no_chl": 
                                chl_scale = 0.0 
                            else: 
                                chl_scale = 1.0 
                            if scenario == "no_wind": 
                                wind_scale = 0.0 
                            else: 
                                wind_scale = 1.0 
                            if scenario == "no_clouds": 
                                cloud_scale = 0.0 
                            else: 
                                cloud_scale = 1.0 
                            if scenario == "no_ice": 
                                ice_scale = 0.0 
                            else: 
                                ice_scale = 1.0
                            if scenario == "no_meltpond": 
                                tas_scale = -20. # Celsius
                            else: 
                                tas_scale = 0.0
                            if scenario == "snow_sensitivity": 
                                snow_attenuation = 5.9 # Lebrun et al. 2023
                            else: 
                                # A value of 20 m-1 was used in the original (Budgell) ROMS sea ice 
                                # module (see line 711 in bulk_flux.F).
                                snow_attenuation = 20.0 # ROMS ice code bulk_flux.f90  

                            logging.info(f"[CMIP6_light] {scenario}: Running {model_object.current_time} model {model_object.name} scenario {current_experiment_id}")
                            # Calculate OSA for each grid point (this is without the effect of sea ice and snow)
                            
                            zr = [
                                CMIP6_albedo_utils.calculate_OSA(
                                    zeniths[i],
                                    wind[i, j] * wind_scale,
                                    chl[i, j] * chl_scale,
                                    self.config.wavelengths,
                                    self.config.refractive_indexes,
                                    self.config.alpha_chl,
                                    self.config.alpha_w,
                                    self.config.beta_w,
                                    self.config.alpha_wc,
                                    self.config.solar_energy,
                                    scenario,
                                )
                                for i in range(m)
                                for j in range(n)
                            ]

                            res = np.squeeze(np.asarray(dask.compute(zr)))
                            OSA = res[:, 0, :].reshape((m, n, 2))
                            
                            direct_OSA = np.squeeze(OSA[:, :, 0])
                            diffuse_OSA = np.squeeze(OSA[:, :, 1])
                            # Estimated values for average albedo from snow and ice prior to real calculations 
                            # later which are used for attenuation into water column through ice and snow.
                            direct_OSA = np.where(siconc > 0.01, 0.52, direct_OSA)
                            direct_OSA = np.where(sisnconc > 0.01, 0.65, direct_OSA)
                           # logging.info(f"Mean OSA {np.nanmean(direct_OSA)} between {np.nanmin(direct_OSA)} and {np.nanmax(direct_OSA)}")
                            
                            direct_OSA_ice_snow = self.CMIP6_cesm3.direct_and_diffuse_albedo_from_snow_and_ice(
                                direct_OSA, 
                                sisnconc*ice_scale, 
                                sisnthick*ice_scale, 
                                siconc*ice_scale, 
                                sithick*ice_scale, 
                                tas+tas_scale)
                          #  logging.info(f"Mean direct_OSA_ice_snow {np.nanmean(direct_OSA_ice_snow)} between {np.nanmin(direct_OSA_ice_snow)} and {np.nanmax(direct_OSA_ice_snow)}")
                            
                            # Calculate radiation calculation uses the direct_OSA to calculate the diffuse radiation
                            # Effect of albedo is added in `compute_surface_solar_for_specific_wavelength_band`
                            direct_sw, diffuse_sw, ghi = self.calculate_radiation(
                                ctime,
                                pv_system,
                                clt*cloud_scale,
                                prw,
                                ozone,
                                direct_OSA_ice_snow,
                                lat,
                                m,
                                n
                            )
                            
                            # scale
                            direct_sw = direct_sw*1.17
                            diffuse_sw = direct_sw*1.17
                            ghi = ghi*1.17
                            wavelengths = np.arange(200, 2700, 10)
                         #   logging.info(f"Mean radiation {np.nanmean(direct_sw)} between {np.nanmin(direct_sw)} and {np.nanmax(direct_sw)}")
                            
                            
                            dr_sw_broadband = np.squeeze(
                                np.trapezoid(y=direct_sw, x=wavelengths, axis=0)
                            )
                            df_sw_broadband = np.squeeze(
                                np.trapezoid(y=diffuse_sw, x=wavelengths, axis=0)
                            )

                            # Scale the diffuse and direct albedo to get total broadband albedo for use going forward
                            # Equation 17 in Sefarian et al. 2018
                            
                            OSA = (
                                direct_OSA * dr_sw_broadband
                                + diffuse_OSA * df_sw_broadband
                            ) / (dr_sw_broadband + df_sw_broadband)

                            OSA = np.where(np.isnan(OSA), 0.06, OSA)
                            
                            # Add the effect of snow and ice on broadband albedo
                            OSA_ice_ocean = self.CMIP6_cesm3.direct_and_diffuse_albedo_from_snow_and_ice(
                                OSA, 
                                sisnconc*ice_scale, 
                                sisnthick*ice_scale, 
                                siconc*ice_scale, 
                                sithick*ice_scale, 
                                tas+tas_scale
                            )

                            logging.debug(
                                f"[CMIP6_light] GHI range {np.nanmin(ghi)} to {np.nanmax(ghi)} (scenario: {scenario})"
                            )
                            logging.debug(
                                f"[CMIP6_light] OSA range {np.nanmin(OSA_ice_ocean)} to {np.nanmax(OSA_ice_ocean)} (scenario: {scenario})"
                            )
                            
                            # Calculate shortwave radiation entering the ocean after accounting for the effect of snow
                            # and ice on the direct and diffuse albedos and for attenuation (no scattering). The effect of
                            # snow and ice on the albedo is calculated in the `compute_surface_solar_for_specific_wavelength_band`
                            # for different bands of the spectrum.
                            
                            def compute_surface_solar_for_band(start_index, end_index, spectrum):
                                return self.CMIP6_cesm3.compute_surface_solar_for_specific_wavelength_band(
                                    OSA_ice_ocean,
                                    direct_sw[start_index:end_index, :, :],
                                    diffuse_sw[start_index:end_index, :, :],
                                    chl * chl_scale,
                                    sisnthick * ice_scale,
                                    sithick * ice_scale,
                                    snow_attenuation,
                                    spectrum=spectrum,
                                )

                            sw_vis_attenuation_corrected_for_snow_ice_chl = compute_surface_solar_for_band(
                                self.config.start_index_visible, self.config.end_index_visible, "vis"
                            )
                            sw_uv_attenuation_corrected_for_snow_ice_chl = compute_surface_solar_for_band(
                                self.config.start_index_uv, self.config.end_index_uv, "uv"
                            )
                            sw_uva_attenuation_corrected_for_snow_ice_chl = compute_surface_solar_for_band(
                                self.config.start_index_uva, self.config.end_index_uva, "uva"
                            )
                            sw_uvb_attenuation_corrected_for_snow_ice_chl = compute_surface_solar_for_band(
                                self.config.start_index_uvb, self.config.end_index_uvb, "uvb"
                            )

                            # Integrate values across wavelengths
                            par = np.squeeze(
                                np.trapezoid(
                                    y=sw_vis_attenuation_corrected_for_snow_ice_chl,
                                    x=wavelengths[self.config.start_index_visible:self.config.end_index_visible],
                                    axis=0,
                                )
                            )
                            logging.debug(
                                f"[CMIP6_light] PAR range {np.nanmin(par)} to {np.nanmax(par)} (scenario: {scenario})"
                            )

                            uv = np.squeeze(
                                np.trapezoid(
                                    y=sw_uv_attenuation_corrected_for_snow_ice_chl,
                                    x=wavelengths[self.config.start_index_uv:self.config.end_index_uv],
                                    axis=0,
                                )
                            )

                            uvb = np.squeeze(
                                np.trapezoid(
                                    y=sw_uvb_attenuation_corrected_for_snow_ice_chl,
                                    x=wavelengths[self.config.start_index_uvb:self.config.end_index_uvb],
                                    axis=0,
                                )
                            )

                            uva = np.squeeze(
                                np.trapezoid(
                                    y=sw_uva_attenuation_corrected_for_snow_ice_chl,
                                    x=wavelengths[self.config.start_index_uva:self.config.end_index_uva],
                                    axis=0,
                                )
                            )

                            # Calculate the UV index and UVI at surface
                            uvi = self.CMIP6_cesm3.calculate_uvi(
                                sw_uv_attenuation_corrected_for_snow_ice_chl,
                                ozone,
                                wavelengths[self.config.start_index_uv:self.config.end_index_uv],
                            )

                            uvb_srf = np.squeeze(np.trapezoid(y=direct_sw[self.config.start_index_uvb:self.config.end_index_uvb],
                                                     x=wavelengths[self.config.start_index_uvb:self.config.end_index_uvb], axis=0)) + \
                                 np.squeeze(np.trapezoid(y=diffuse_sw[self.config.start_index_uvb:self.config.end_index_uvb],
                                                     x=wavelengths[self.config.start_index_uvb:self.config.end_index_uvb], axis=0))

                            do_plot = False
                            if do_plot and hour_of_day == 12: # and sel_time.month in [1,2,3,4,5,6,7,8,9,10,11,12]:
                                plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()
                                
                                plotter.create_plots(
                                    lon,
                                    lat,
                                    model_object,
                                    OSA_VIS=OSA_ice_ocean,
                                    plotname_postfix=f"_OSA_BROADBAND_{scenario}"
                                )
                            
                                plotter.create_plots(lon, lat, model_object,
                                                    irradiance_water=par,
                                                    plotname_postfix=f"_vis_{scenario}")
                                
                                plotter.create_plots(lon, lat, model_object,
                                                    siconc=siconc,
                                                    plotname_postfix=f"_siconc_{scenario}")
                                plotter.create_plots(lon, lat, model_object,
                                                    sithick=sithick,
                                                    plotname_postfix="_sithick_{}".format(scenario))
                                plotter.create_plots(lon, lat, model_object,
                                                    chl=chl,
                                                    plotname_postfix="_chl_{}".format(scenario))
                                plotter.create_plots(lon, lat, model_object,
                                                    sisnthick=sisnthick,
                                                    plotname_postfix="_sithick_{}".format(scenario))
                                """
                                plotter.create_plots(lon, lat, model_object,
                                                    uvi=uvi,
                                                    plotname_postfix="_UVI_{}".format(scenario))

                                plotter.create_plots(lon, lat, model_object,
                                                    siconc=siconc,
                                                    plotname_postfix="_siconc_{}".format(scenario))

                                plotter.create_plots(lon, lat, model_object,
                                                    sithick=sithick,
                                                    plotname_postfix="_sithick_{}".format(scenario))

                                plotter.create_plots(lon, lat, model_object,
                                                    sithick=sithick,
                                                    plotname_postfix="_sithick_{}".format(scenario))

                                plotter.create_plots(lon, lat, model_object,
                                                    chl=chl,
                                                    plotname_postfix="_chl_{}".format(scenario))

                                plotter.create_plots(lon, lat, model_object,
                                                    clt=clt,
                                                    plotname_postfix="_clt_{}".format(scenario))
                                """

                            for data_list, vari in zip(
                                [
                                    par,
                                    ghi,
                                    uvb,
                                    uva,
                                    uv,
                                    uvi,
                                    uvb_srf,
                                    OSA_ice_ocean,
                                ],
                                [
                                    "par",
                                    "ghi",
                                    "uvb",
                                    "uva",
                                    "uv",
                                    "uvi",
                                    "uvb_srf",
                                    "osa",
                                ],
                            ):
                                filename = self.get_filename(
                                    vari,
                                    model_object.name,
                                    model_object.current_member_id,
                                    scenario,
                                    current_experiment_id,
                                )
                             
                                if not os.path.exists(Path(filename).parent):
                                    os.makedirs(os.path.dirname(Path(filename)).parent, exist_ok=True)
                    
                                self.save_irradiance_to_netcdf(
                                    filename,
                                    data_list,
                                    vari,
                                    time_counter,
                                    model_object.current_time,
                                    lat[:, 0],
                                    lon[0, :],
                                )

                        time_counter += 1

        # Upload final results to GCS
        for scenario in self.config.scenarios:
            for vari in [
                "par",
                "ghi",
                "uvb",
                "uva",
                "uv",
                "uvi",
                "uvb_srf",
                "osa",
            ]:
                filename = self.get_filename(
                    vari,
                    model_object.name,
                    model_object.current_member_id,
                    scenario,
                    current_experiment_id,
                )
                filename_gcs = filename.replace(self.config.outdir, "")
                filename_gcs = f"{self.config.cmip6_outdir}{filename_gcs}"
                if not os.path.exists(filename):
                    os.makedirs(os.path.dirname(Path(filename)).parent, exist_ok=True)
                io.upload_to_gcs(filename, fname_gcs=filename_gcs)
                os.remove(filename)

    def get_filename(
        self, vari, model_name, member_id, scenario, current_experiment_id
    ):
        """Create the filename depending on scenario, member_id, and experiment_id"""
        out = f"{self.config.outdir}/{current_experiment_id}"
        if not os.path.exists(out):
            os.makedirs(out, exist_ok=True)
        return f"{out}/{vari}_{model_name}_{member_id}_{self.config.start_date}-{self.config.end_date}_scenario_{scenario}_{current_experiment_id}.nc"

    def save_irradiance_to_netcdf(
        self, filename, da, vari, time_counter, time, lat, lon
    ):
        if time_counter == 0 and os.path.exists(filename) is False:
          
            cdf = netCDF4.Dataset(filename, mode="w")
            cdf.title = f"RTM calculations of {vari}"
            cdf.description = "Created for revision 2 of Kristiansen et al. 2024 (in review)"

            cdf.history = f"Created {datetime.datetime.now()}"
            cdf.link = "https://github.com/trondkr/RTM"

            # Define dimensions
            cdf.createDimension("lon", len(lon))
            cdf.createDimension("lat", len(lat))
            cdf.createDimension("time", None)

            vnc = cdf.createVariable("lon", "d", ("lon",))
            vnc.long_name = "Longitude"
            vnc.units = "degree_east"
            vnc.standard_name = "lon"
            vnc[:] = lon

            vnc = cdf.createVariable("lat", "d", ("lat",))
            vnc.long_name = "Latitude"
            vnc.units = "degree_north"
            vnc.standard_name = "lat"
            vnc[:] = lat

            v_time = cdf.createVariable("time", "d", ("time",))
            v_time.long_name = "days since 2000-01-16 00:00:00"
            v_time.units = "days since 2000-01-16 00:00:00"
            v_time.field = "time, scalar, series"
            v_time.calendar = "standard"

            v_u = cdf.createVariable(
                vari,
                "f",
                (
                    "time",
                    "lat",
                    "lon",
                ),
            )
            v_u.long_name = "CMIP6_light"
            v_u.units = "W m-2"
            v_u.time = "time"

        else:
            cdf = netCDF4.Dataset(filename, "a")
        cdf.variables[vari][time_counter, :, :] = da
        cdf.variables["time"][time_counter] = netCDF4.date2num(
            time, units="days since 2000-01-16 00:00:00", calendar="standard"
        )
        cdf.close()

    # logging.info("[CMIP6_light] Wrote results to {}".format(result_file))

    def calculate_light(self, current_experiment_id):
        io = CMIP6_IO.CMIP6_IO()
        if self.config.use_local_CMIP6_files:
            io.organize_cmip6_netcdf_files_into_datasets(
                self.config, current_experiment_id
            )
        else:
            io.organize_cmip6_datasets(self.config, current_experiment_id)
        io.print_table_of_models_and_members()

        self.cmip6_models = io.models
        logging.info(
            "[CMIP6_light] Light calculations will involve {} CMIP6 model(s)".format(
                len(self.cmip6_models)
            )
        )

        for ind, model in enumerate(self.cmip6_models):
            for member_id in model.member_ids:
                model.current_member_id = member_id
                logging.info(
                    f"[CMIP6_light] {ind}/{len(self.cmip6_models)}: {model.name} member_id: {model.current_member_id}"
                )

                # Save datafiles to do calculations locally
                if self.config.write_CMIP6_to_file:
                    io.extract_dataset_and_save_to_netcdf(
                        model, self.config, current_experiment_id
                    )
                if self.config.perform_light_calculations:
                    self.perform_light_calculations(model, current_experiment_id)


def main(args):
    light = CMIP6_light(args)
    light.config.setup_logging()
    light.config.setup_parameters()
    logging.info("[CMIP6_config] logging started")

    for current_experiment_id in light.config.experiment_ids:
        light.calculate_light(current_experiment_id)

    
if __name__ == "__main__":
    #np.warnings.filterwarnings("ignore")
    # https://docs.dask.org/en/latest/diagnostics-distributed.html
    # https://docs.dask.org/en/latest/setup/single-distributed.html
    dask.config.set({'array.slicing.split_large_chunks': True})
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--source_id",
        dest="source_id",
        help="source_id",
        type=str,
        required=True
    )
    parser.add_argument(
        "-i",
        "--member_id",
        dest="member_id",
        help="member_id",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args)

    logging.info("[CMIP6_light] Execution of downscaling completed")
