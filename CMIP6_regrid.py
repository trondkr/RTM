try:
    import esmpy as ESMF
except ImportError:
    import ESMF
import xarray as xr
import numpy as np
import xesmf as xe
from xmip.preprocessing import combined_preprocessing

class CMIP6_regrid:

    def regrid_variable(self, varname, ds_in, ds_out, interpolation_method="bilinear"):
        """
        Regrids a variable from the input dataset to the output dataset using the specified interpolation method.

        Parameters:
            varname (str): The name of the variable to regrid.
            ds_in (xarray.Dataset): The input dataset containing the variable to regrid.
            ds_out (xarray.Dataset): The output dataset to regrid the variable onto.
            interpolation_method (str, optional): The interpolation method to use. Defaults to "bilinear".

        Returns:
            xarray.Dataset: The regridded dataset containing the regridded variable.

        Raises:
            None

        Examples:
            # Regrid the variable "temperature" from the input dataset to the output dataset
            regrid_variable("temperature", ds_in, ds_out, interpolation_method="nearest")
        """
        if "lat_bounds" in list(ds_in.coords) and "lon_bounds" in list(ds_in.coords):
            ds_in = ds_in.drop({"lat_bounds", "lon_bounds"})
        if "yTe" in list(ds_in.coords) and "xTe" in list(ds_in.coords):
            ds_in = ds_in.drop({"yTe", "xTe"})
        if "vertices_latitude" in list(ds_in.coords) and "vertices_longitude" in list(ds_in.coords):
            ds_in = ds_in.drop({"vertices_latitude", "vertices_longitude"})

        regridder = xe.Regridder(ds_in, ds_out, interpolation_method,
                                 periodic=True,
                                 extrap_method='inverse_dist',
                                 extrap_num_src_pnts=10,
                                 extrap_dist_exponent=1,
                                 ignore_degenerate=False)

        print("[CMIP6_regrid] regridding {}".format(varname))

        return regridder(ds_in[varname]).to_dataset(name=varname)


