import dask
import numpy as np
import logging
import xarray as xr
import datetime
import cftime

# http://www.temis.nl/data/conversions.pdf
# Data availability: https://esgf-node.llnl.gov/search/input4mips/

# NOTE 1:
# The downloaded data from input4MPI was split into several files which I concatenated
# cdo mergetime vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_195001-199912.nc
# cdo mergetime vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_195001-199912.nc
# cdo mergetime  vmro3_input4MIPs_ozone_CMIP_UReading-CCMI*  vmro3_input4MIPs_ozone_CMIP_UReading-CCMI_1950_2015.nc
# cdo mergetime   vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp245*  vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp245-1-0_gn_2015_2100.nc
# cdo mergetime   vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp585*  vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp585-1-0_gn_2015_2100.nc

# NOTE 2:
# Prior to using the concatenated data from input4MPIs I had to convert the units
# from months to hours as xarray and Python can not handle months.
#
# cdo -settunits,hours  vmro3_input4MIPs_ozone_CMIP_UReading-CCMI_1950_2015.nc test.nc
# mv test.nc vmro3_input4MIPs_ozone_CMIP_UReading
# cdo -settunits,hours  vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp245-1-0_gn_2015_2100.nc test.nc
# mv test.nc vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp245-1-0_gn_2015_2100.nc
# cdo -settunits,hours  vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp585-1-0_gn_2015_2100.nc test.nc
# mv test.nc vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp585-1-0_gn_2015_2100.nc

class CMIP6_ozone():

    def get_input4mpis_forcing(self, scenario: str, baseurl: str) -> xr.Dataset:
        logging.info("[CMIP6_ozone] Getting ozone input4MPI forcing data...")

        histfile = baseurl + "vmro3_input4MIPs_ozone_CMIP_UReading-CCMI_1950_2015.nc"
        if scenario == "ssp585":
            profile = baseurl + "vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp585-1-0_gn_2015_2100.nc"
        if scenario == "ssp245":
            profile = baseurl + "vmro3_input4MIPs_ozone_ScenarioMIP_UReading-CCMI-ssp245-1-0_gn_2015_2100.nc"

        ds_hist = xr.open_dataset(histfile).sel(lat=(slice(0, 90))).sel(time=(slice("1950-01-01", "2099-12-31")))
        ds_proj = xr.open_dataset(profile).sel(lat=(slice(0, 90))).sel(time=(slice("1950-01-01", "2099-12-31")))

        # Concatenate the timeseries
        ds = xr.concat([ds_hist, ds_proj], dim="time")
        time = ds.time.values
        logging.info("[CMIP6_ozone] Ozone input4MPI forcing data range: {} to {}".format(time[0], time[-1]))

        return ds

    def convert_vmro3_to_toz(self, scenario: str, ds: xr.Dataset, baseurl: str):

        R = 287.3  # Jkg-1K-1 (Specific gas constant for air)
        T0 = 273.15  # Kelvin(Standard temperaure)
        P0 = 1.01325e5  # Pa (Standard pressure at surface)
        g0 = 9.80665  # ms-2 (Global average gravity at surface)
        Na = 6.0220e23  # Avogadro´s number

        # Integrating the total column of a trace gas from input4MPI forcing data. Here
        # P is the pressure in hPa, VMR is the colume mixing ration in ppm and TOZ is the trace gas
        # column amount in Dobson Units (DU):'bnds', 'lat', 'lon', 'plev', 'time'

        mole2ppmv = 1e6
        VMR = ds["vmro3"].values * mole2ppmv
        plev = ds["plev"].values
          
        times=ds["time"].values
        times_plus=[]
        

        # Mix of Timestamp and DateTimeNoLeap - convert all to DateTimeNoLeap
        for t in times:
            if isinstance(t, datetime.datetime):
                times_plus.append(
                    cftime.DatetimeNoLeap(t.year, t.month, t.day, t.hour))
            else:
                times_plus.append(t)

        VMR = np.where(VMR > 1000, np.nan, VMR)
        plev = np.where(plev > 1000, np.nan, plev)

        TOZ = 10 * ((R * T0) / (g0 * P0)) * np.nansum(0.5 * ((VMR[:, 0:-2:1, :, :] + VMR[:, 1:-1:1, :, :]) * (
                    plev[None, 0:-2:1, None, None] - plev[None, 1:-1:1, None, None])), axis=1)

        # Create a dataset
        toz_ds = xr.DataArray(
            name="TOZ",
            data=TOZ,
            coords={'time': (['time'], times_plus),
                    'lat': (['lat'], ds["lat"].values),
                    'lon': (['lon'], ds["lon"].values)},
            dims=["time", "lat", "lon"],
        ).to_dataset()
        toz_ds.to_netcdf(baseurl+"/TOZ_{}.nc".format(scenario))
           
        logging.info("[CMIP6_ozone] Results written to file covering period {} to {}".format(times_plus[0],times_plus[-1]))
        logging.info("[CMIP6_ozone] TOZ min {} to max {} and mean {}".format(np.nanmin(TOZ),np.nanmax(TOZ),np.nanmean(TOZ)))

    def setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def convert_to_toz(self):
        scenario = "ssp585"
        baseurl = "../data/ozone-absorption/"
        ds = self.get_input4mpis_forcing(scenario, baseurl)
        self.convert_vmro3_to_toz(scenario, ds, baseurl)


def main():
    ozone = CMIP6_ozone()
    ozone.setup_logging()
    ozone.convert_to_toz()


if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    # https://docs.dask.org/en/latest/diagnostics-distributed.html
    from dask.distributed import Client

    dask.config.set(scheduler='processes')

    client = Client()
    status = client.scheduler_info()['services']
    print("Dask started with status at: http://localhost:{}/status".format(status["dashboard"]))
    print(client)
    main()
