import numpy as np
import pandas as pd
import xarray as xr
from math import radians, sin
import timeit
from shapely.geometry import box
import geopandas as gpd

def lat_lon_cell_area(lat, lon, d_lat, d_lon):

    """
    Calculate the area of a cell, in km^2, on a lat/lon grid.

    This applies the following equation from Santinie et al. 2010.

    S = (λ_2 - λ_1)(sinφ_2 - sinφ_1)R^2

    S = surface area of cell on sphere
    λ_1, λ_2, = bands of longitude in radians
    φ_1, φ_2 = bands of latitude in radians
    R = radius of the sphere

    Santini, M., Taramelli, A., & Sorichetta, A. (2010). ASPHAA: A GIS‐Based
    Algorithm to Calculate Cell Area on a Latitude‐Longitude (Geographic)
    Regular Grid. Transactions in GIS, 14(3), 351-377.
    https://doi.org/10.1111/j.1467-9671.2010.01200.x

    Parameters
    ----------
    lat_lon_grid_cell
        A shapely box with coordinates on the lat/lon grid

    Returns
    -------
    float
        The cell area in km^2

    """
    R = 6371.0088  # Earth's radius in km
    lat_rad = np.radians(lat)
    
    a = np.sin(np.radians(d_lat) / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    lat_distance = R * c
    
    a = np.cos(lat_rad) * np.cos(lat_rad) * np.sin(np.radians(d_lon) / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    lon_distance = R * c
    
    return lat_distance * lon_distance

def calculate_cell_area(lat, lon, side_length):
    """
    Calculate the area of a single grid cell.
    """
    return lat_lon_cell_area(lat, lon, side_length, side_length)

def calculate_area_of_grid(da: xr.DataArray, side_length: float):
    """
    Calculate the area of each grid cell for a DataArray with dimensions (time, lat, lon). This code
    is now vectorized using xarray `apply_ufunc` and should be faster than the previous version.

    Parameters:
        da (xr.DataArray): The DataArray containing the grid data with dimensions (time, lat, lon).
        side_length (float): The side length of each grid cell in degrees. GLORYS=1/12 and ROMS=1/64

    Returns:
        xr.DataArray: A DataArray containing the grid cell areas with dimensions (time, lat, lon).
    """
    # Create meshgrid of latitudes and longitudes
    lons, lats = np.meshgrid(da.lon.values, da.lat.values)
    
    # Calculate areas for each grid cell
    areas = xr.apply_ufunc(
        calculate_cell_area,
        lats,
        lons,
        input_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        kwargs={'side_length': side_length}
    )
    
    # Create a DataArray with the same dimensions as the input
    area_da = xr.DataArray(
        areas,
        dims=['lat', 'lon'],
        coords={'lat': da.lat, 'lon': da.lon}
    )
    
    # Broadcast the area DataArray to match the time dimension of the input
    area_da = area_da.broadcast_like(da)
    
    # Mask the areas where the input data is NaN
    return xr.where(np.isnan(da), np.nan, area_da)
 