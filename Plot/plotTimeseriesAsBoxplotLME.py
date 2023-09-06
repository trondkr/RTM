
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import datetime as datetime
import pandas as pd
import glob
import os
import string
from matplotlib.pyplot import cm
import cmocean
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np, scipy.stats as st
import statsmodels.stats.api as sms

__author__   = 'Trond Kristiansen'
__email__    = 'me@trondkristiansen.com'
__created__  = datetime.datetime(2018, 5, 29)
__modified__ = datetime.datetime(2018, 5, 19)
__version__  = "1.0"
__status__   = "Development, 29.5.2015, 29.05.2018"


def create_plots_compare(ds2020, ds2050):  # only used for comparisons

    ds_diff = ds2050 - ds2020
    print(ds_diff)

    land_110m = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    proj = ccrs.PlateCarree()
    extent = [-20, 20, 50, 80]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), subplot_kw={'projection': proj})
    for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        ds2020.sel(season=season).where(pd.notnull).plot.pcolormesh(
            ax=axes[i, 0], cmap='Spectral_r', transform=ccrs.PlateCarree(),  # vmin=-30, vmax=30, cmap='Spectral_r',
            add_colorbar=True, extend='both')
      #  axes[i, 0].set_extent(extent, crs=proj)
        axes[i, 0].add_feature(land_110m, color="lightgrey")
        axes[i, 0].add_feature(cfeature.COASTLINE, edgecolor="black")
        axes[i, 0].add_feature(cfeature.BORDERS, linestyle=':')

        ds2050.sel(season=season).where(pd.notnull).plot.pcolormesh(
            ax=axes[i, 1], cmap='Spectral_r', transform=ccrs.PlateCarree(),  # vmin=-30, vmax=30, cmap='Spectral_r',
            add_colorbar=True, extend='both')
     #   axes[i, 1].set_extent(extent, crs=proj)
        axes[i, 1].add_feature(land_110m, color="lightgrey")
        axes[i, 1].add_feature(cfeature.COASTLINE, edgecolor="black")
        axes[i, 1].add_feature(cfeature.BORDERS, linestyle=':')

        ds_diff.sel(season=season).where(pd.notnull).plot.pcolormesh(
            ax=axes[i, 2], cmap='Spectral_r', transform=ccrs.PlateCarree(),  # vmin=-0.1, vmax=.1, cmap='RdBu_r',
            add_colorbar=True, extend='both')
      #  axes[i, 2].set_extent(extent, crs=proj)
        axes[i, 2].add_feature(land_110m, color="lightgrey")
        axes[i, 2].add_feature(cfeature.COASTLINE, edgecolor="black")
        axes[i, 2].add_feature(cfeature.BORDERS, linestyle=':')

        axes[i, 0].set_ylabel(season)
        axes[i, 1].set_ylabel('')
        axes[i, 2].set_ylabel('')

    for ax in axes.flat:
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.axis('tight')
        ax.set_xlabel('')

    axes[0, 0].set_title('ds2020')
    axes[0, 1].set_title('ds2050')
    axes[0, 2].set_title('Difference')

   # plt.tight_layout()

    fig.suptitle('Seasonal Chlorophyll', fontsize=16, y=1.02)
    plt.show()

def calculate_season_averages(ds:xr.Dataset):
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'.
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')
    return ds_weighted

if __name__ == '__main__':
    light = xr.open_dataset("../../oceanography/light/ncfiles/irradiance_ACCESS-ESM1-5_r1i1p1f1_1950-01-01-1950-12-01.nc")
    print(light)

    varname1 = "irradiance_vis_osa",
    varname2 = "irradiance_vis_osa"

    ds2020 = calculate_season_averages(light.irradiance_vis_osa.sel(time=slice("1950-01-01","1950-12-31")))
    ds2050 = calculate_season_averages(light.irradiance_vis_osa.sel(time=slice("1950-01-01", "1950-12-31")))

    create_plots_compare(ds2020, ds2050)
