import datetime
import logging
import os
from calendar import monthrange
from typing import List, Any
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dask.distributed import Client
    
# Computational modules
import cftime
import dask
import netCDF4
import numpy as np
import pandas as pd
import pvlib
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


class CMIP6_light:
    def __init__(self, args):
        np.seterr(divide="ignore")
        self.config = CMIP6_config.Config_albedo()
        self.config.source_id = args.source_id
        self.config.member_id = args.member_id

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

        I_diff_s = np.trapz(
            y=spectra["poa_sky_diffuse"], x=spectra["wavelength"], axis=0
        ) + np.trapz(y=spectra["poa_ground_diffuse"], x=spectra["wavelength"], axis=0)
        I_dir_s = np.trapz(y=spectra["poa_direct"], x=spectra["wavelength"], axis=0)
        I_glob_s = np.trapz(y=spectra["poa_global"], x=spectra["wavelength"], axis=0)

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
        bias_delta=None,
    ) -> np.array:
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
        )  # , temperature=temp)

        airmass_relative = pvlib.atmosphere.get_relative_airmass(
            solpos["apparent_zenith"].to_numpy(), model="kasten1966"
        )
        pressure = pvlib.atmosphere.alt2pres(altitude)
        apparent_zenith = (
            np.ones((np.shape(cloud_covers))) * solpos["apparent_zenith"].to_numpy()
        )
        zenith = np.ones((np.shape(cloud_covers))) * solpos["zenith"].to_numpy()
        azimuth = np.ones((np.shape(cloud_covers))) * solpos["azimuth"].to_numpy()
        surface_azimuth = np.ones((np.shape(cloud_covers))) * system["surface_azimuth"]

        # Always we use zero tilt when working with pvlib and incoming
        # irradiance on a horizontal plane flat on earth
        surface_tilt = np.zeros((np.shape(cloud_covers)))

        # cloud cover in fraction units here. this is used in campbell_norman functions
        # TODO: Use the implemented function instead :
        # pvlib.forecast.GFS.cloud_cover_to_irradiance_campbell_norman

        aoi = pvlib.irradiance.aoi(
            surface_tilt, surface_azimuth, apparent_zenith, azimuth
        )

        # Fixed atmospheric components used from pvlib example
        tau500 = np.ones((np.shape(cloud_covers))) * 0.1

        # day of year is an int64index array so access first item
        day_of_year = ctime.dayofyear
        day_of_year = np.ones((np.shape(cloud_covers))) * day_of_year[0]

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
            ground_albedo=albedo,
            surface_pressure=pressure,
            relative_airmass=airmass_relative,
            precipitable_water=water_vapor_content,
            ozone=ozone,
            aerosol_turbidity_500nm=tau500,
            dayofyear=day_of_year,
        )

        dni_extra = pvlib.irradiance.get_extra_radiation(day_of_year)
        transmittance = (1.0 - cloud_covers) * 0.75
        irrads_clouds = pvlib.irradiance.campbell_norman(
            apparent_zenith, transmittance, dni_extra=dni_extra
        )

        if bias_delta is not None:
            delta = bias_delta
        else:
            delta = 0.0
        ghi_cloud_corrected = irrads_clouds["ghi"] + delta

        """
            Estimate Direct Normal Irradiance from Global Horizontal Irradiance
            using the DISC model.

            The DISC algorithm converts global horizontal irradiance to direct
            normal irradiance through empirical relationships between the global
            and direct clearness indices.

            The pvlib implementation limits the clearness index to 1.
        """

        # Correct the clear sky for clouds
        # ghi_cloud_corrected = (0.35 + (1 - 0.35) * (1 - cloud_covers)) * ghi_cloud_corrected
        # ghi_cloud_corrected = ghi_cloud_corrected # + bias_delta.values

        ghi_cloud_corrected = np.where(
            ghi_cloud_corrected < 0, 0.0, ghi_cloud_corrected
        )

        bad_values = np.isnan(ghi_cloud_corrected)
        ghi_cloud_corrected = np.where(bad_values, 0, ghi_cloud_corrected)

        kt = pvlib.irradiance.clearness_index(
            ghi_cloud_corrected,
            zenith,
            dni_extra,
            min_cos_zenith=0.065,
            max_clearness_index=1,
        )

        am = pvlib.atmosphere.get_absolute_airmass(airmass_relative, pressure)

        Kn, am = pvlib.irradiance._disc_kn(kt, am, max_airmass=12)
        dni = Kn * dni_extra

        bad_values = (
            (zenith > 87)
            | (ghi_cloud_corrected < 0)
            | (dni < 0)
            | (np.isnan(ghi_cloud_corrected))
        )
        dni = np.where(bad_values, 0, dni)
        dhi = ghi_cloud_corrected - dni * np.cos(np.radians(zenith))

        # Convert the irradiance to a plane with tilt zero horizontal to the earth. This is done applying tilt=0 to POA
        # calculations using the output from campbell_norman. The POA calculations include calculating sky and ground
        # diffuse light where specific models can be selected (we use default). Here, albedo is used to calculate
        # ground diffuse irradiance

        POA_irradiance_clouds = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dni=dni,
            ghi=ghi_cloud_corrected,
            dhi=dhi,
            solar_zenith=apparent_zenith,
            solar_azimuth=azimuth,
            albedo=albedo,
            model="klucher",
            model_perez="allsitescomposite1990",
        )

        # Account for cloud opacity on the spectral radiation
        F_dir, F_diff = self.cloud_opacity_factor(
            POA_irradiance_clouds["poa_diffuse"],
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
        results[0, :, 2] = ghi_cloud_corrected

        # return direct, diffuse, and ghi cloud affected radiation at wavelengths
        return results

    def season_mean(self, ds, calendar="standard"):
        # Make a DataArray of season/year groups

        year_season = xr.DataArray(
            ds.time.to_index().to_period(freq="Q-NOV").to_timestamp(how="E"),
            coords=[ds.time],
            name="year_season",
        )

        # Make a DataArray with the number of days in each month, size = len(time)
        date_tool = CMIP6_date_tools.CMIP6_date_tools()
        month_length = xr.DataArray(
            date_tool.get_dpm(ds.time.to_index(), calendar=calendar),
            coords=[ds.time],
            name="month_length",
        )
        # Calculate the weights by grouping by 'time.season'
        weights = (
            month_length.groupby("time.season")
            / month_length.groupby("time.season").sum()
        )

        # Test that the sum of the weights for each season is 1.0
        np.testing.assert_allclose(
            weights.groupby("time.season").sum().values, np.ones(4)
        )

        # Calculate the weighted average
        return (ds * weights).groupby("time.season").sum(dim="time")

    def create_chlorophyll_avg_for_year(self, year, ds_in, ds_out):
        start_date = "{}-01-01".format(year)
        end_date = "{}-12-31".format(year)
        ds_chl_2020 = ds_in.sel(time=slice(start_date, end_date))  # .mean('time')

        year2 = 2050
        start_date = "{}-01-01".format(year2)
        end_date = "{}-12-31".format(year2)
        ds_chl_2050 = ds_in.sel(time=slice(start_date, end_date))
        re = CMIP6_regrid.CMIP6_regrid()
        chl2020 = re.regrid_variable("chl", ds_chl_2020, ds_out)
        chl2050 = re.regrid_variable("chl", ds_chl_2050, ds_out)

        ds_2020 = chl2020.to_dataset()
        ds_2050 = chl2050.to_dataset()

        lat = ds_2020.lat.values
        lon = ds_2020.lon.values

        weighted_average_2020 = self.season_mean(ds_2020, calendar="noleap")
        weighted_average_2050 = self.season_mean(ds_2050, calendar="noleap")

        ds_diff = (
            100
            * (weighted_average_2050 - weighted_average_2020)
            / weighted_average_2020
        )
        chl2020 = weighted_average_2020.sel(season="MAM").chl.values
        chl2050 = weighted_average_2050.sel(season="MAM").chl.values
        chl2050_diff = ds_diff.sel(season="MAM").chl.values

        # kg/m3 to mg/m3 multiply by 1e6
        plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()

        plotter.create_plot(
            (chl2020 * 1.0e6),
            lon[0, :],
            lat[:, 0],
            "chl2020",
            nlevels=np.arange(0, 5, 0.2),
            regional=True,
            logscale=True,
        )
        plotter.create_plot(
            (chl2050 * 1.0e6),
            lon[0, :],
            lat[:, 0],
            "chl2050",
            nlevels=np.arange(0, 5, 0.2),
            regional=True,
            logscale=True,
        )
        plotter.create_plot(
            chl2050_diff,
            lon[0, :],
            lat[:, 0],
            "chl2050-2020",
            nlevels=np.arange(-101, 101, 1),
            regional=True,
        )

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

    def get_linke_turbidity(self, ctime, lat, lon=0.0):
        # Calculating the linke turbidity requires opening a h5 file which crashes dask. Therefore,
        # we do the calculations outside of the loop and send the ready made array into the radiation
        # function. This variable is used by inchein clearsky calculation
        # https://meteonorm.com/assets/publications/ieashc36_report_TL_AOD_climatologies.pdf

        linke_turbidity = np.zeros(len(lat[:, 0]))
        for i in range(len(lat[:, 0])):
            linke_turbidity[i] = pvlib.clearsky.lookup_linke_turbidity(
                ctime, lat[i, 0], lon, interp_turbidity=False
            )
        return linke_turbidity

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

        """
        table = texttable.Texttable()
        table.set_cols_align(["c", "c", "c"])
        table.set_cols_valign(["m", "m", "m"])

        table.header(["Model", "Variable", "Range"])
        table.add_rows([
            ["", "clt", "{:3.3f} to {:3.3f}.".format(np.nanmin(clt), np.nanmax(clt))],
            ["", "chl", "{:3.3f} to {:3.3f}.".format(np.nanmin(chl), np.nanmax(chl))],
            ["", "prw", "{:3.3f} to {:3.3f}.".format(np.nanmin(prw), np.nanmax(prw))],
            ["{}".format(model_object.name), "tas", "{:3.3f} to {:3.3f}.".format(np.nanmin(tas), np.nanmax(tas))],
            ["", "uas", "{:3.3f} to {:3.3f}.".format(np.nanmin(uas), np.nanmax(uas))],
            ["", "vas", "{:3.3f} to {:3.3f}.".format(np.nanmin(vas), np.nanmax(vas))],
            ["", "siconc", "{:3.3f} to {:3.3f}.".format(np.nanmin(siconc), np.nanmax(siconc))],
            ["", "sithick", "{:3.3f} to {:3.3f}.".format(np.nanmin(sithick), np.nanmax(sithick))],
            ["", "sisnconc", "{:3.3f} to {:3.3f}.".format(np.nanmin(sisnconc), np.nanmax(sisnconc))],
            ["", "sisnthick", "{:3.3f} to {:3.3f}.".format(np.nanmin(sisnthick), np.nanmax(sisnthick))]
        ])
        table.set_cols_width([30, 20, 30])
        print(table.draw() + "\n")
        """

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
        clt: np.ndarray,
        prw: np.ndarray,
        ozone: np.ndarray,
        direct_OSA: np.ndarray,
        lat: np.ndarray,
        m: int,
        n: int,
        bias_delta: np.ndarray = None,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        wavelengths = np.arange(200, 2700, 10)

        if self.config.bias_correct_ghi:
            bias = np.zeros(
                np.shape(bias_delta["ghi"][ctime.month[0] - 1, :, :])
            ) + np.nanmean(np.squeeze(bias_delta["ghi"][ctime.month[0] - 1, :, :]))
            
            calc_radiation = [
                dask.delayed(self.radiation)(
                    clt[j, :],
                    prw[j, :],
                    lat[j, 0],
                    ctime,
                    pv_system,
                    direct_OSA[j, :],
                    ozone[j, :],
                    bias_delta=bias[j, :],
                )
                for j in range(m)
            ]
        else:
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
        # Method that reads the total ozone column from input4MPI dataset (Micahela Heggelin)
        # and regrid to consistent 1x1 degree dataset.
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
        # One Dobson Unit is the number of molecules of ozone that would be required to create a layer
        # of pure ozone 0.01 millimeters thick at a temperature of 0 degrees Celsius and a pressure of 1 atmosphere
        # (the air pressure at the surface of the Earth). Expressed another way, a column of air with an ozone
        # concentration of 1 Dobson Unit would contain about 2.69x1016 ozone molecules for every
        # square centimeter of area at the base of the column. Over the Earth’s surface, the ozone layer’s
        # average thickness is about 300 Dobson Units or a layer that is 3 millimeters thick.
        #
        # https://ozonewatch.gsfc.nasa.gov/facts/dobson_SH.html
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

        if self.config.bias_correct_ghi:
            bias_delta = xr.open_dataset(self.config.bias_correct_file)
            bias_delta = bias_delta.assign_coords(lon=(bias_delta.lon % 360)).sortby(
                "lon"
            )

        start_at_noon = False
        for selected_time in range(0, len(times.values)):
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
                for hour_of_day in range(0, 23, 4):
                    if hour_of_day == 12:
                        start_at_noon = True
                    if start_at_noon is True:
                        model_object.current_time = datetime.datetime(
                            year=sel_time.year,
                            month=sel_time.month,
                            day=day + 1,
                            hour=hour_of_day,
                        )

                        logging.info(
                            f"[CMIP6_light] Running {model_object.current_time} model {model_object.name} scenario {current_experiment_id}"
                        )

                        ctime, pv_system = self.setup_pv_system(
                            model_object.current_time
                        )
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
                                tas_scale = -2. # Celsius
                            else: 
                                tas_scale = 0.0
                            if scenario == "snow_sensitivity": 
                                snow_attenuation = 5.9 # Lebrun et al. 2023
                            else: 
                                # A value of 20 m-1 was used in the original (Budgell) ROMS sea ice module (see line 711 in bulk_flux.F).
                                snow_attenuation = 20.0 # ROMS ice code bulk_flux.f90

                            logging.info("[CMIP6_light] Running scenario: {}".format(scenario))
                            # Calculate OSA for each grid point (this is without the effect of sea ice and snow)
                            if hour_of_day == 12:
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
                                    )
                                    for i in range(m)
                                    for j in range(n)
                                ]

                                res = np.squeeze(np.asarray(dask.compute(zr)))
                                OSA = res[:, 0, :].reshape((m, n, 2))

                                direct_OSA = np.squeeze(OSA[:, :, 0])
                                diffuse_OSA = np.squeeze(OSA[:, :, 1])

                            # Calculate radiation calculation uses the direct_OSA to calculate the diffuse radiation
                            # Effect of albedo is added in `compute_surface_solar_for_specific_wavelength_band`

                            direct_sw, diffuse_sw, ghi = self.calculate_radiation(
                                ctime,
                                pv_system,
                                clt*cloud_scale,
                                prw,
                                ozone,
                                direct_OSA,
                                lat,
                                m,
                                n,
                                bias_delta,
                            )

                            wavelengths = np.arange(200, 2700, 10)

                            dr_sw_broadband = np.squeeze(
                                np.trapz(y=direct_sw, x=wavelengths, axis=0)
                            )
                            df_sw_broadband = np.squeeze(
                                np.trapz(y=diffuse_sw, x=wavelengths, axis=0)
                            )

                            # Scale the diffuse and direct albedo to get total broadband albedo for use going forward
                            # Equation 17 in Sefarian et al. 2018
                            if hour_of_day == 12:
                                OSA = (
                                    direct_OSA * dr_sw_broadband
                                    + diffuse_OSA * df_sw_broadband
                                ) / (dr_sw_broadband + df_sw_broadband)

                                OSA = np.where(np.isnan(OSA), np.nanmean(OSA), OSA)

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

                            if scenario == "no_osa":
                                OSA_ice_ocean = np.where(
                                    OSA_ice_ocean < 0.08, 0.06, OSA_ice_ocean
                                )

                            # Calculate shortwave radiation entering the ocean after accounting for the effect of snow
                            # and ice to the direct and diffuse albedos and for attenuation (no scattering).
                            # The final product adds diffuse and direct
                            # light for the spectrum in question (vis or uv).
                            sw_vis_attenuation_corrected_for_snow_ice_chl = self.CMIP6_cesm3.compute_surface_solar_for_specific_wavelength_band(
                                OSA_ice_ocean,
                                direct_sw[
                                    self.config.start_index_visible : self.config.end_index_visible,
                                    :,
                                    :,
                                ],
                                diffuse_sw[
                                    self.config.start_index_visible : self.config.end_index_visible,
                                    :,
                                    :,
                                ],
                                chl * chl_scale,
                                sisnthick*ice_scale,
                                sithick*ice_scale,
                                snow_attenuation,
                                spectrum = "vis",
                            )                                                               

                            sw_uv_attenuation_corrected_for_snow_ice_chl = self.CMIP6_cesm3.compute_surface_solar_for_specific_wavelength_band(
                                OSA_ice_ocean,
                                direct_sw[
                                    self.config.start_index_uv : self.config.end_index_uv,
                                    :,
                                    :,
                                ],
                                diffuse_sw[
                                    self.config.start_index_uv : self.config.end_index_uv,
                                    :,
                                    :,
                                ],
                                chl * chl_scale,
                                sisnthick*ice_scale,
                                sithick*ice_scale,
                                snow_attenuation,
                                spectrum = "uv",
                            )
                            # Integrate values across wavelengths according to which variable we are considering
                            par = np.squeeze(
                                np.trapz(
                                    y=sw_vis_attenuation_corrected_for_snow_ice_chl,
                                    x=wavelengths[
                                        self.config.start_index_visible : self.config.end_index_visible
                                    ],
                                    axis=0,
                                )
                            )
                            uv = np.squeeze(
                                np.trapz(
                                    y=sw_uv_attenuation_corrected_for_snow_ice_chl,
                                    x=wavelengths[
                                        self.config.start_index_uv : self.config.end_index_uv
                                    ],
                                    axis=0,
                                )
                            )

                            sw_srf = np.squeeze(
                                np.trapz(y=direct_sw, x=wavelengths, axis=0)
                            ) + np.squeeze(
                                np.trapz(y=diffuse_sw, x=wavelengths, axis=0)
                            )

                            uv_srf = np.squeeze(
                                np.trapz(
                                    y=direct_sw[
                                        self.config.start_index_uv : self.config.end_index_uv
                                    ],
                                    x=wavelengths[
                                        self.config.start_index_uv : self.config.end_index_uv
                                    ],
                                    axis=0,
                                )
                            ) + np.squeeze(
                                np.trapz(
                                    y=diffuse_sw[
                                        self.config.start_index_uv : self.config.end_index_uv
                                    ],
                                    x=wavelengths[
                                        self.config.start_index_uv : self.config.end_index_uv
                                    ],
                                    axis=0,
                                )
                            )

                            uv_b = np.squeeze(
                                np.trapz(
                                    y=diffuse_sw[
                                        self.config.start_index_uvb : self.config.end_index_uvb
                                    ],
                                    x=wavelengths[
                                        self.config.start_index_uvb : self.config.end_index_uvb
                                    ],
                                    axis=0,
                                )
                            ) + np.squeeze(
                                np.trapz(
                                    y=direct_sw[
                                        self.config.start_index_uvb : self.config.end_index_uvb
                                    ],
                                    x=wavelengths[
                                        self.config.start_index_uvb : self.config.end_index_uvb
                                    ],
                                    axis=0,
                                )
                            )

                            uv_a = np.squeeze(
                                np.trapz(
                                    y=diffuse_sw[
                                        self.config.start_index_uva : self.config.end_index_uva
                                    ],
                                    x=wavelengths[
                                        self.config.start_index_uva : self.config.end_index_uva
                                    ],
                                    axis=0,
                                )
                            ) + np.squeeze(
                                np.trapz(
                                    y=direct_sw[
                                        self.config.start_index_uva : self.config.end_index_uva
                                    ],
                                    x=wavelengths[
                                        self.config.start_index_uva : self.config.end_index_uva
                                    ],
                                    axis=0,
                                )
                            )

                            uvi = self.CMIP6_cesm3.calculate_uvi(
                                sw_uv_attenuation_corrected_for_snow_ice_chl,
                                ozone,
                                wavelengths[
                                    self.config.start_index_uv : self.config.end_index_uv
                                ],
                            )

                            do_plot = False
                            if do_plot:
                                plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()
                                plotter.create_plots(
                                    lon,
                                    lat,
                                    model_object,
                                    OSA_VIS=OSA_ice_ocean,
                                    plotname_postfix="_OSA_BROADBAND_{}".format(
                                        scenario
                                    ),
                                )
                                """
                                plotter.create_plots(lon, lat, model_object,
                                                    direct_sw=par,
                                                    plotname_postfix="_vis_{}".format(scenario))

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
                                    sw_srf,
                                    ghi,
                                    uv_b,
                                    uv_a,
                                    uv,
                                    uv_srf,
                                    uvi,
                                    OSA_ice_ocean,
                                ],
                                [
                                    "par",
                                    "sw_srf",
                                    "ghi",
                                    "uvb",
                                    "uva",
                                    "uv",
                                    "uv_srf",
                                    "uvi",
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
                "sw_srf",
                "ghi",
                "uvb",
                "uva",
                "uv",
                "uv_srf",
                "uvi",
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
            cdf.description = "Created for revision 2 of paper."

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
    np.warnings.filterwarnings("ignore")
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
    """    
    with Client() as client:  # (n_workers=4, threads_per_worker=4, processes=True, memory_limit='15GB') as client:
        status = client.scheduler_info()["services"]
        assert client.status == "running"
        logging.info("[CMIP6_light] client {}".format(client))
        logging.info(
            "Dask started with status at: http://localhost:{}/status".format(
                status["dashboard"]
            )
        )
        main(args)
        client.close()
        assert client.status == "closed"
    """

    logging.info("[CMIP6_light] Execution of downscaling completed")
