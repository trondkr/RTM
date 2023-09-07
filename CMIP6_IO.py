from datetime import datetime
import logging
import xarray as xr
import cftime
import numpy as np
from cmip6_preprocessing.preprocessing import combined_preprocessing
import CMIP6_model
import CMIP6_config
import CMIP6_regrid
import xesmf as xe
import os
import texttable
import pandas as pd
from google.cloud import storage
import logging
import gcsfs
import hashlib
import time
import base64

class CMIP6_IO:

    def __init__(self):
        self.models = []
        self. storage_client = storage.Client()
        self.bucket_name = "actea-shared"
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.fs = gcsfs.GCSFileSystem(project="downscale")
        self.logger = logging.getLogger("CMIP6-log")
        self.logger.setLevel(logging.INFO)

    def format_netcdf_filename(self, dir, model_name, member_id, current_experiment_id, key):
        return "{}/{}/{}/CMIP6_{}_{}_{}_{}.nc".format(dir, current_experiment_id,
                                                      model_name,
                                                      model_name, member_id, current_experiment_id, key)


    def calculate_md5_sha(self, file_name: str) -> (str, str):
        """
        Calculate the md5 hash tag for a file to ensure that upload to gs works correctly.
        https://stackoverflow.com/questions/52686848/does-google-cloud-storage-client-in-python-check-crc-or-md5-automatically

        Parameters
        -----------
            file_name: name of the file to calculate md5 and sha1

        Returns
        -----------
            md5, sha1
        """
        start = time.time()
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return base64.b64encode(hash_md5.digest()).decode()

    def upload_to_gcs(self, fname: str):
        """ upload file to GCS.

        Method that uploads file to the GCS blob. Calculates the md5 sha
        prior to uploading which is used by GCS to ensure that
        upload was successful.
        """
       
        md5 = self.calculate_md5_sha(f"{fname}")
        blob = self.bucket.blob(fname)
        blob.md5_hash = md5
        blob.upload_from_filename(fname)
        self.logger.info(f"[CMIP6_IO] Finished uploading to : {fname}")


    def print_table_of_models_and_members(self):
        table = texttable.Texttable()
        table.set_cols_align(["c", "c", "c"])
        table.set_cols_valign(["t", "m", "b"])

        table.header(["Model", "Member", "Variable"])
        for model in self.models:
            for member_id in model.member_ids:
                var_info = ""
                for var in model.ocean_vars[member_id]:
                    var_info += "{} ".format(var)
                table.add_row([model.name, member_id, var_info])

        table.set_cols_width([30, 30, 50])
        print(table.draw() + "\n")

    def open_dataset_on_gs(self, file_name: str, decode_times:bool=True) -> xr.Dataset:
       
        if storage.Blob(bucket=self.bucket, name=str(file_name)).exists(self.storage_client):
            file_name = f"{self.bucket_name}/{file_name}"
       
            self.logger.info(f"[CMIP6_IO] Opening file {file_name}")
            fileObj = self.fs.open(file_name)
            return xr.open_dataset(fileObj, engine="h5netcdf", decode_times=decode_times)

    def organize_cmip6_netcdf_files_into_datasets(self, config: CMIP6_config.Config_albedo, current_experiment_id):
        
        for source_id in config.source_ids:
            if source_id in config.models.keys():
                model_object = config.models[source_id]
            else:
                model_object = CMIP6_model.CMIP6_MODEL(name=source_id)

            logging.info("[CMIP6_IO] Organizing NetCDF CMIP6 model object {}".format(model_object.name))

            for member_id in config.member_ids:
                for variable_id, table_id in zip(config.variable_ids, config.table_ids):

                    netcdf_filename = self.format_netcdf_filename(config.cmip6_netcdf_dir,
                                                                  model_object.name,
                                                                  member_id,
                                                                  current_experiment_id,
                                                                  variable_id)
                    
                    if storage.Blob(bucket=self.bucket, name=netcdf_filename).exists(self.storage_client):
                        ds = self.open_dataset_on_gs(netcdf_filename, decode_times=True)

                        # Extract the time period of interest
                        ds = ds.sel(time=slice(config.start_date, config.end_date),y=slice(config.min_lat, config.max_lat))

                        logging.info("[CMIP6_IO] {} => NetCDF: Extracted {} range from {} to {} for {}".format(source_id,
                                                                                                        variable_id,
                                                                                                        ds["time"].values[0],
                                                                                                        ds["time"].values[-1],current_experiment_id))
                        # Save the info to model object
                        if not member_id in model_object.member_ids:
                            model_object.member_ids.append(member_id)

                        if not member_id in model_object.ocean_vars.keys():
                            model_object.ocean_vars[member_id] = []
                        if not variable_id in model_object.ocean_vars[member_id]:
                            current_vars = model_object.ocean_vars[member_id]
                            current_vars.append(variable_id)
                            model_object.ocean_vars[member_id] = current_vars

                        self.dataset_into_model_dictionary(member_id, variable_id, ds, model_object)
                #  else:
                #      logging.info("[CMIP6_IO] {} did not have member id {} - continue...".format(model_object.name,
                #                                                                                  member_id))
            self.models.append(model_object)
            logging.info("[CMIP6_IO] Stored {} variables for model {}".format(len(model_object.ocean_vars),
                                                                              model_object.name))

    def to_360day_monthly(self, ds:xr.Dataset):
        """Change the calendar to datetime and precision to monthly."""
        # https://github.com/pydata/xarray/issues/3320
        time1 = ds.time.copy()
        for itime in range(ds.sizes['time']):
            bb = ds.time.values[itime].timetuple()
            time1.values[itime] = datetime(bb[0], bb[1], 16)
        logging.info("[CMIP6_IO] Fixed time units start at {} and end at {}".format(time1.values[0],time1.values[-1]))
        ds = ds.assign_coords({'time': time1})
        return ds

    # Loop over all models and scenarios listed in CMIP6_light.config
    # and store each CMIP6 variable and scenario into a CMIP6 model object
    def organize_cmip6_datasets(self, config: CMIP6_config.Config_albedo, current_experiment_id):

       # for experiment_id in config.experiment_ids:
        for grid_label in config.grid_labels:
            for source_id in config.source_ids:

                if source_id in config.models.keys():
                    model_object = config.models[source_id]
                else:
                    model_object = CMIP6_model.CMIP6_MODEL(name=source_id)

                logging.info("[CMIP6_IO] Organizing CMIP6 model object {}".format(model_object.name))

                for member_id in config.member_ids:
                    collection_of_variables = []
                    missing=[]
                    for variable_id, table_id in zip(config.variable_ids, config.table_ids):

                        # Historical query string
                        query_string = "source_id=='{}'and table_id=='{}' and member_id=='{}' and grid_label=='{}' and experiment_id=='historical' and variable_id=='{}'".format(
                            source_id,
                            table_id,
                            member_id,
                            grid_label,
                            variable_id)

                        ds_hist = self.perform_cmip6_query(config, query_string)

                        # Future projection depending on choice in experiment_id
                        query_string = "source_id=='{}'and table_id=='{}' and member_id=='{}' and grid_label=='{}' and experiment_id=='{}' and variable_id=='{}'".format(
                            source_id,
                            table_id,
                            member_id,
                            grid_label,
                            current_experiment_id,
                            variable_id,
                        )
                        ds_proj = self.perform_cmip6_query(config, query_string)

                        if isinstance(ds_proj, xr.Dataset) and isinstance(ds_hist, xr.Dataset):
                            # Concatenate the historical and projections datasets
                            ds = xr.concat([ds_hist, ds_proj], dim="time")

                            if not ds.indexes["time"].dtype in ["datetime64[ns]"]:
                                start_date = datetime.fromisoformat(config.start_date)
                                end_date = datetime.fromisoformat(config.end_date)
                                ds = self.to_360day_monthly(ds)
                            else:
                                start_date = config.start_date
                                end_date = config.end_date
                            ds = xr.decode_cf(ds)
                            logging.info(
                                "[CMIP6_IO] Variable: {} and units {}".format(variable_id, ds[variable_id].units))
                            if variable_id in ["prw"]:
                                # 1 kg of rain water spread over 1 square meter of surface is 1 mm in thickness
                                # The pvlib functions takes cm so we convert values
                                ds[variable_id].values = ds[variable_id].values / 10.0
                                ds.attrs["units"] = "cm"
                                logging.info(
                                    "[CMIP6_IO] Minimum {} and maximum {} values after converting to {} units".format(np.nanmin(ds[variable_id].values),
                                                                                                             np.nanmax(ds[variable_id].values),
                                                                                                             ds[variable_id].units))

                            if variable_id in ["tas"]:
                                if ds[variable_id].units in ["K","Kelvin","kelvin"]:
                                    ds[variable_id].values = ds[variable_id].values - 273.15
                                    ds.attrs["units"] = "C"
                                    logging.info(
                                        "[CMIP6_IO] Minimum {} and maximum {} values after converting to {} units".format(
                                            np.nanmin(ds[variable_id].values),
                                            np.nanmax(ds[variable_id].values),
                                            ds[variable_id].units))


                            # Remove the duplicate overlapping times (e.g. 2001-2014)
                            _, index = np.unique(ds["time"], return_index=True)
                            ds = ds.isel(time=index)
                           # if not isinstance((ds.indexes["time"]), pd.DatetimeIndex):
                           #     ds["time"] = ds.indexes["time"].to_datetimeindex()
                            ds = ds.sel(time=slice(start_date, end_date))
                            ds["time"] = pd.to_datetime(ds.indexes["time"])

                            # Extract the time period of interest
                            ds = ds.sel(time=slice(start_date, end_date))

                            logging.info(
                                "[CMIP6_IO] {} => Extracted {} range from {} to {} for member {}".format(source_id,
                                                                                                         variable_id,
                                                                                                         ds[
                                                                                                             "time"].values[
                                                                                                             0],
                                                                                                         ds[
                                                                                                             "time"].values[
                                                                                                             -1],
                                                                                                         member_id))

                            # pass the pre-processing directly
                            dset_processed = combined_preprocessing(ds)
                            if variable_id in ["chl"]:
                                if source_id in ["CESM2", "CESM2-FV2", "CESM2-WACCM-FV2"]:
                                    dset_processed = dset_processed.isel(lev_partial=config.selected_depth)
                                else:
                                    dset_processed = dset_processed.isel(lev=config.selected_depth)
                            if variable_id in ["ph"]:

                                logging.info("[CMIP6_IO] => Extract only depth level {}".format(config.selected_depth))
                                dset_processed = dset_processed.isel(lev=config.selected_depth)
                            collection_of_variables.append(variable_id)
                            
                            # Save the info to model object
                            if not member_id in model_object.member_ids:
                                model_object.member_ids.append(member_id)

                            if not member_id in model_object.ocean_vars.keys():
                                model_object.ocean_vars[member_id] = []
                            if not variable_id in model_object.ocean_vars[member_id]:
                                current_vars = model_object.ocean_vars[member_id]
                                current_vars.append(variable_id)
                                model_object.ocean_vars[member_id] = current_vars

                            self.dataset_into_model_dictionary(member_id, variable_id,
                                                               dset_processed,
                                                               model_object)

                    if collection_of_variables!=config.variable_ids:
                        missing = [x for x in config.variable_ids if not x in collection_of_variables or collection_of_variables.remove(x)]        
                        logging.error(f"[CMIP6_IO] Error - unable to find some variable {missing}")
                    else:
                        missing=[]
                    if len(missing)==0:
                        collection_of_variables=[]
                        missing=[]
                        self.models.append(model_object)
                        logging.info("[CMIP6_IO] Stored {} variables for model {}".format(len(model_object.ocean_vars),
                                                                                        model_object.name))

    def dataset_into_model_dictionary(self,
                                      member_id: str,
                                      variable_id: str,
                                      dset: xr.Dataset,
                                      model_object: CMIP6_model.CMIP6_MODEL):
        # Store each dataset for each variable as a dictionary of variables for each member_id
        try:
            existing_ds = model_object.ds_sets[member_id]
        except KeyError:
            existing_ds = {}

        existing_ds[variable_id] = dset

        model_object.ds_sets[member_id] = existing_ds

    def perform_cmip6_query(self, config, query_string: str) -> xr.Dataset:
        df_sub = config.df.query(query_string)
        if df_sub.zstore.values.size == 0:
            return df_sub

        mapper = config.fs.get_mapper(df_sub.zstore.values[-1])
        logging.debug("[CMIP6_IO] df_sub: {}".format(df_sub))

        ds = xr.open_zarr(mapper, consolidated=True, mask_and_scale=True)

        # print("Time encoding: {} - {}".format(ds.indexes['time'], ds.indexes['time'].dtype))
        if not ds.indexes["time"].dtype in ["datetime64[ns]", "object"]:

            time_object = ds.indexes['time'].to_datetimeindex()  # pd.DatetimeIndex([ds["time"].values[0]])

            # Convert if necessary
            if time_object[0].year == 1:

                times = ds.indexes['time'].to_datetimeindex()  # pd.DatetimeIndex([ds["time"].values])
                times_plus_2000 = []
                for t in times:
                    times_plus_2000.append(
                        cftime.DatetimeNoLeap(t.year + 2000, t.month, t.day, t.hour)
                    )
                ds["time"].values = times_plus_2000
                ds = xr.decode_cf(ds)

        return ds

    def write_netcdf(self, ds: xr.Dataset, out_file: str) -> None:
        enc = {}

        for k in ds.data_vars:
            if ds[k].ndim < 2:
                continue

            enc[k] = {
                "zlib": True,
                "complevel": 3,
                "fletcher32": True,
                "chunksizes": tuple(map(lambda x: x//2, ds[k].shape))
            }

        ds.to_netcdf(out_file, format="NETCDF4", engine="netcdf4", encoding=enc)
    
    """
        Regrid to cartesian grid and save to NetCDF:
        For any Amon related variables (wind, clouds), the resolution from CMIP6 models is less than
        1 degree longitude x latitude. To interpolate to a 1x1 degree grid we therefore first interpolate to a
        2x2 degrees grid and then subsequently to a 1x1 degree grid.
    """

    def extract_dataset_and_save_to_netcdf(self, model_obj, config: CMIP6_config.Config_albedo, current_experiment_id):

        if os.path.exists(config.cmip6_outdir) is False:
            os.mkdir(config.cmip6_outdir)
        #  ds_out_amon = xe.util.grid_global(2, 2)
        ds_out_amon = xe.util.grid_2d(config.min_lon,
                                      config.max_lon, 2,
                                      config.min_lat,
                                      config.max_lat, 2)
        #  ds_out = xe.util.grid_global(1, 1)
        ds_out = xe.util.grid_2d(config.min_lon,
                                 config.max_lon, 1,
                                 config.min_lat,
                                 config.max_lat, 1)

        re = CMIP6_regrid.CMIP6_regrid()

        for key in model_obj.ds_sets[model_obj.current_member_id].keys():

            current_ds = model_obj.ds_sets[model_obj.current_member_id][key]  # .sel(
            #   y=slice(int(config.min_lat), int(config.max_lat)),
            #   x=slice(int(config.min_lon), int(config.max_lon)))

            if all(item in current_ds.dims for item in ['time', 'y', 'x', 'vertex', 'bnds']):
                ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'time', 'vertex', 'y', 'x')
            elif all(item in current_ds.dims for item in ['time', 'y', 'x', 'vertices', 'bnds']):
                ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'time', 'vertices', 'y', 'x')
            elif all(item in current_ds.dims for item in ['time', 'y', 'x', 'bnds']):
                ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'time', 'y', 'x')
            elif all(item in current_ds.dims for item in ['time', 'y', 'x', 'nvertices']):
                ds_trans = current_ds.chunk({'time': -1}).transpose('time', 'y', 'x','nvertices')
            else:
                ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'time', 'y', 'x')

            if key in ["uas", "vas", "clt", "tas"]:
                out_amon = re.regrid_variable(key,
                                              ds_trans,
                                              ds_out_amon,
                                              interpolation_method=config.interp) #.to_dataset()

                out = re.regrid_variable(key, out_amon, ds_out,
                                         interpolation_method=config.interp)
            else:
                out = re.regrid_variable(key, ds_trans,
                                         ds_out,
                                         interpolation_method=config.interp)
            
            if config.write_CMIP6_to_file:
                out_dir = "{}/{}/{}".format(config.cmip6_outdir, current_experiment_id, model_obj.name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                outfile = "{}/{}/{}/CMIP6_{}_{}_{}_{}.nc".format(config.cmip6_outdir,
                                                                 current_experiment_id,
                                                                 model_obj.name,
                                                                 model_obj.name,
                                                                 model_obj.current_member_id,
                                                                 current_experiment_id,
                                                                 key)
                if os.path.exists(outfile): os.remove(outfile)

                # Convert to dataset before writing to netcdf file. Writing to file downloads and concatenates all
                # of the data and we therefore re-chunk to split the process into several using dask
                #    ds = ds_trans.to_dataset()
                self.write_netcdf(out.chunk({'time': -1}), out_file=outfile)
                
                logging.info("[CMIP6_light] wrote variable {} to file".format(key))
