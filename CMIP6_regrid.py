import ESMF
import xarray as xr
import numpy as np
import xesmf as xe
from cmip6_preprocessing.preprocessing import combined_preprocessing


class CMIP6_regrid:

    def regrid_variable(self, varname, ds_in, ds_out, interpolation_method="bilinear"):

        if "lat_bounds" and "lon_bounds" in list(ds_in.coords):
            ds_in = ds_in.drop({"lat_bounds", "lon_bounds"})
        if "yTe" and "xTe" in list(ds_in.coords):
            ds_in = ds_in.drop({"yTe", "xTe"})
        if "vertices_latitude" and "vertices_longitude" in list(ds_in.coords):
            ds_in = ds_in.drop({"vertices_latitude", "vertices_longitude"})

      
        regridder = xe.Regridder(ds_in, ds_out, interpolation_method,
                                     periodic=True,
                                     extrap_method='inverse_dist',
                                     extrap_num_src_pnts=10,
                                     extrap_dist_exponent=1,
                                     ignore_degenerate=False)
     

        print("[CMIP6_regrid] regridding {}".format(varname))

        return regridder(ds_in[varname]).to_dataset(name=varname)


