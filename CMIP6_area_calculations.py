from datetime import datetime

import iris
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from scipy import stats


def calc_trend(xarr: xr.DataArray):
    # getting shapes

    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]
    # creating x and y variables for linear regression
    #
    # Some CMIP6 models return CFTimeIndex while others return DatetImeIndex - need the latter
    # to convert to Datetime objects
    if isinstance(xarr.time.to_pandas().index, pd.DatetimeIndex):
        x = xarr.time.to_pandas().index.to_julian_date().values[:, None]
    else:
        x = (
            xarr.time.to_pandas()
            .index.to_datetimeindex()
            .to_julian_date()
            .values[:, None]
        )
    y = xarr.to_masked_array().reshape(n, -1)

    # ############################ #
    # LINEAR REGRESSION DONE BELOW #
    xm = x.mean(0)  # mean
    ym = y.mean(0)  # mean

    ya = y - ym  # anomaly
    xa = x - xm  # anomaly

    # variance and covariances
    xss = (xa**2).sum(0) / (n - 1)  # variance of x (with df as n-1)
    yss = (ya**2).sum(0) / (n - 1)  # variance of y (with df as n-1)
    xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    intercept = ym - (slope * xm)
    # statistics about fit
    df = n - 2
    r = xys / (xss * yss) ** 0.5
    t = r * (df / ((1 - r) * (1 + r))) ** 0.5
    p = stats.distributions.t.sf(abs(t), df)
    # misclaneous additional functions
    # yhat = dot(x, slope[None]) + intercept
    # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
    # se = ((1 - r**2) * yss / xss / df)**0.5

    # preparing outputs
    out = xarr[:2].mean("time")
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name += "_slope"
    xarr_slope.attrs["units"] = "units / month"
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name += "_Pvalue"
    xarr_p.attrs["info"] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name="slope")
    xarr_out["pval"] = xarr_p
    xarr_out = xarr_out.expand_dims("time")
    return xarr_out


def xr_add_cyclic_point(da, varname):
    """
    Inputs
    da: xr.DataArray with dimensions (time,lat,lon)
    """

    # Use add_cyclic_point to interpolate input data
    lon_idx = 2  # da.dims.index('lon')

    wrap_data, wrap_lon = add_cyclic_point(da.values, coord=da.lon, axis=lon_idx)

    # Generate output DataArray with new data but same structure as input
    return xr.DataArray(
        data=wrap_data,
        coords={"time": da.time, "lat": da.lat, "lon": wrap_lon},
        dims=da.dims,
        name=varname,
        attrs=da.attrs,
    )


def add_variable_units(cube, units: str):
    cube.units = units
    return cube


def add_attributes_to_cube(cube):
    a = cube.attributes
    a["date"] = str(datetime.now())
    cube.attributes = a
    return cube


def ds_to_iris(
    ds: xr.Dataset,
    var_name: str,
):
    ds_iris = ds[var_name].to_iris()
    ds_iris = fix_coordinates_cube(ds_iris)

    ds_iris = add_attributes_to_cube(ds_iris)
    if var_name == "thetao":
        ds_iris = add_variable_units(ds_iris, "celsius")
    if var_name == "depth":
        ds_iris = add_variable_units(ds_iris, "meter")
    if var_name == "o2":
        ds_iris = add_variable_units(ds_iris, "ml/l")
    if var_name == "areacello":
        ds_iris = add_variable_units(ds_iris, "m^2")
    if var_name == "siconc":
        ds_iris = add_variable_units(ds_iris, "1")
    if var_name in ["par", "uvb", "uv"]:
        ds_iris = add_variable_units(ds_iris, "W/m^2")
    return ds_iris


def fix_coordinates_cube(cube):
    for coord in cube.coords():
        if coord.name() == "lat":
            lat = cube.coord("lat")
            cube.remove_coord("lat")
        if coord.name() == "latitude":
            lat = cube.coord("latitude")
            cube.remove_coord("latitude")
        if coord.name() == "lon":
            lon = cube.coord("lon")
            cube.remove_coord("lon")
        if coord.name() == "longitude":
            lon = cube.coord("longitude")
            cube.remove_coord("longitude")

    lat.standard_name = "latitude"
    lon.standard_name = "longitude"
    lat.units = "degrees"
    lon.units = "degrees"

    # Depth is ndim=2 and thetao has ndim=3
    cube.add_dim_coord(lat, cube.ndim - 2)
    cube.add_dim_coord(lon, cube.ndim - 1)

    if not cube.coord("latitude").has_bounds():
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()

    return cube


def calculate_areacello(ds, var_name):
    # Calculate the area based on the longitude - latitude
    if ds.lon.ndim == 2:
        lon = ds.lon.values[0, :]
        lat = ds.lat.values[:, 0]
    else:
        lon = ds.lon.values
        lat = ds.lat.values

    ds_singletime = ds.isel(time=0)
    # Convert the dataset to a cube as this adds correct units required by iris
    cube = ds_to_iris(ds_singletime, var_name)
    print("cube", cube, lon, lat)
    # Calculate the areacello for the grid and convert the result to km2
    # Uses iris area_weights function.
    # https://scitools.org.uk/iris/docs/v2.4.0/iris/iris/analysis/cartography.html#iris.analysis.cartography.area_weights
    m2_to_km2 = 1.0e-6

    area_ends = (
        iris.analysis.cartography.area_weights(cube, normalize=False)
    ) * m2_to_km2
    print("area_ends", area_ends)
    # Now convert the numpy array of areas to a dataset with the same dimension as the siconc
    area_ds = xr.DataArray(
        name="areacello",
        data=area_ends,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
    ).to_dataset()

    # Convert the resulting dataset to an iris cube
    area_cube = ds_to_iris(area_ds, "areacello")

    # Fix the coordinates so that we add geographic information to the cube,
    # before saving the cube to the siconc dataset
    area_cube = fix_coordinates_cube(area_cube)
    return xr.DataArray.from_iris(area_cube)
