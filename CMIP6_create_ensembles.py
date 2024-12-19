# %%
from xclim import ensembles
from pathlib import Path
import xarray as xr
from CMIP6_IO import CMIP6_IO
import os
import dask
import flox
from dask.distributed import Client
import numpy as np

# ### Create ensemble files from the outputs of running CMIP6_light.py and from forcing files
# 
# This script will loop over all files found in a speficic folder `lightpath` and find files of the same variable and scenario to create ensemble files from. The various variables to create ensembles for include `["uvb", "uv", "uvi", "par", "uva", "tos", "siconc", "sithick", "tas", "sisnthick", "sisnconc", "uas", "vas"]`.
# 
# The output is stored under folder `ensemble_path` and the results are used to create the final plots of modeled lightlyfor the paper using notebooks `CMIP6_plot_light_results.ipynb` and `CMIP6_calculate_MPIESM2.ipynb`.
# 
# This script may require substantial RAM depending on the region you are creating ensemble for. For datasets covering the Northern Hemisphere we used a 64CPU and 120GB RAM machine.
# 

# Define what models, scenarios, and variables to calculate ensemble files from

source_ids = ["MPI-ESM1-2-HR","MPI-ESM1-2-LR", "CanESM5", "UKESM1-0-LL"]

# The weighting file is created using CMIP6_create_weights.ipynb together with 
# the ClimWIP package. The weights are calculated for variable tos using ERSSTv5 and CORA5.2 as observations
# to compare with. The weights are calculated for the period 1993-2020.
weights_file = "data/calculated_weights_18092024.nc"

create_ensemble = True
scenarios = ["ssp245","ssp585"]
var_names = ["sithick", "tas", "sisnthick", "sisnconc", "uas", "vas"]
var_names = ["tos", "uvb", "uv", "uva", "siconc", "sithick", "tas", "sisnthick", "sisnconc", "uas", "vas"]

var_names = ["uvb", "ghi", "osa","uv", "uvb_srf", "uva", "uvi", "osa", "ghi"]
var_names=["uv_srf"]

root="light/ncfiles_nobias"
percentiles = [2.5, 50.0, 97.5]

create_ensemble_files = True
io = CMIP6_IO()

def get_weights(weights_file: str, f: str):
    weights = xr.open_dataset(weights_file)

    weights = weights["weights"]
    models = weights["model_ensemble"]
    for source_id in source_ids:
        if source_id in f:
            for d, w in zip(models.values, weights.values):
                if source_id in d.split("_")[0]:
                    print(f"Found model in models {source_id} here {d} weight {w}")
                    return w


def create_ensemble_files(scenarios, var_names):
    max_models = 3000
    only_upload = False

    for use_weights in [True, False]:
        for var_name in var_names:
            for scenario in scenarios:
                counter=0
                ds_list = []
                ensemble_stats = None
                ensemble_perc = None
                if var_name in ["prw", "clt", "tos", "siconc", "sithick", "tas", "sisnthick", "sisnconc", "uas", "vas"]:
                    lightpath = f"light/{scenario}"
                    ensemble_path = "light/ensemble/{scenario}"
                else:
                    lightpath = f"{root}/{scenario}"
                    ensemble_path = f"{root}/{scenario}/ensemble"
                if use_weights:
                    ensemble_stats =  f"{ensemble_path}/CMIP6_ensemble_stats_{scenario}_{var_name}_weighted.nc"
                    ensemble_perc =  f"{ensemble_path}/CMIP6_ensemble_perc_{scenario}_{var_name}_weighted.nc"
                else:
                    ensemble_stats =  f"{ensemble_path}/CMIP6_ensemble_stats_{scenario}_{var_name}.nc"
                    ensemble_perc =  f"{ensemble_path}/CMIP6_ensemble_perc_{scenario}_{var_name}.nc"
                 
                if not os.path.exists(ensemble_path):
                    os.makedirs(ensemble_path, exist_ok=True)

                assert (
                    ensemble_stats is not None
                ), "Unable to identify correct variable name to create output filename"
                assert (
                    ensemble_perc is not None
                ), "Unable to identify correct variable name to create output filename"

                if not only_upload:
                    if os.path.exists(ensemble_stats):
                        os.remove(ensemble_stats)
                    if os.path.exists(ensemble_perc):
                        os.remove(ensemble_perc)

                    current = f"{lightpath}"
                
                    file_on_gcs = io.list_dataset_on_gs(current)
                    
                    # Loop over all files and filter on the models defined in source_ids.
                    # For each model read the corresponding weight from the weights file. We
                    # will use these values to create a weights xr.Dataarray to be used to weight each 
                    # model when calculating the ensemble mean, std, max, min, and percentiles.
                    models_weights = []
                    model_names = []
                    for f in file_on_gcs:
                    
                        var_name_mod = var_name
                        if var_name not in ["prw", "clt", "tos", "siconc", "sithick", "tas", "sisnthick", "sisnconc", "uas", "vas"]:
                            var_name_mod = f"{var_name}_"
                            
                        # Filter to only use the models we prefer: and "uv_srf" not in f
                        if any(model in f for model in source_ids) and var_name_mod in f and scenario in f and "weight" not in f and "rsus" not in f and "rsds" not in f:
                            if counter >= max_models:
                                pass
                            else:
                                if var_name=="osa" and f"{scenario}/{var_name}_" not in f:
                                    pass
                                else:
                                    if var_name=="uvb" and "uvb_srf" in f:
                                        pass
                                    else:
                                        ds = io.open_dataset_on_gs(f).persist()  
                                        models_weights.append(get_weights(weights_file, f))
                                        model_names.append(f)
                                        # Drop variable mask as we dont need it and it causes problems when
                                        # calculating the ensemble mean.
                                        if "mask" in ds.variables:
                                            ds = ds.drop("mask")       
                                        ds_list.append(ds)
                                        counter+=1
                                    
                    # Create the final xr.DataArray with the weights.    
                    d = {
                        "dims": "realization",
                        "data": np.array(models_weights),
                        "name": "weights",
                        "models": model_names,
                    }
                    print(f"Number of datasets found {len(ds_list)} weights {d}")
                    weights = xr.DataArray.from_dict(d)

                    
                    for i, ds in enumerate(ds_list):
                        duplicates = ds['time'].to_index().duplicated()
                        if duplicates.any():
                            print(f"Dataset {i} has {duplicates.sum()} duplicate time values for model {model_names[i]}")
            
                    print(f"Creating ensemble for {var_name} and scenario {scenario}")
                    if var_name in ["tos"]:
                        ens = ensembles.create_ensemble(ds_list, resample_freq="MS")
                    else:
                        ens = ensembles.create_ensemble(ds_list) #, resample_freq="MS", calendar="noleap")
                    
                    if use_weights:
                       # if not only_perc:
                        ens_stats = ensembles.ensemble_mean_std_max_min(ens, weights=weights)
                        ens_perc = ensembles.ensemble_percentiles(
                        ens, values=percentiles, split=False, weights=weights)   
                    else:
                        #if not only_perc:
                        ens_stats = ensembles.ensemble_mean_std_max_min(ens)
                        ens_perc = ensembles.ensemble_percentiles(
                        ens, values=percentiles, split=False)
                    
                # Save to file and upload to GCS.
                #if not only_upload:
                #    if not only_perc:
             #   if not only_perc:
                ens_stats.to_netcdf(ensemble_stats)
                ens_perc.to_netcdf(ensemble_perc)
            
                if Path(ensemble_stats).exists():
                    io.upload_to_gcs(ensemble_stats)
                print(f"Created ensemble {ensemble_stats}")

                if Path(ensemble_perc).exists():
                    io.upload_to_gcs(ensemble_perc)
                print(f"Created ensemble {ensemble_perc}")

               # if not only_perc:
                if Path(ensemble_stats).exists():
                    Path(ensemble_stats).unlink()
                if Path(ensemble_perc).exists():
                     Path(ensemble_perc).unlink()
      
if __name__ == '__main__':
    dask.config.set(**{"array.slicing.split_large_chunks": False})  
    with Client() as client: # set up local cluster on your laptop
        create_ensemble_files(scenarios, var_names)          


