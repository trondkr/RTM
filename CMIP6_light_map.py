from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns
import geopandas as gp
import regionmask
import matplotlib.pyplot as plt
from tabulate import tabulate
plt.style.use('default')
import cartopy.crs as ccrs
import os

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as colors

def plot_monthly_climatology(clim, varname,
                             baseURL_output,
                             prefix=""):
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    if varname in ["ghi", "msdwswrf"]:
        units = 'GHI (Wm$^{2}$)'
        lev=np.arange(np.around(clim.min().values,0),np.around(clim.max().values,0), 5)
        lev=[10,50,100,150,175,200,225,250,275,300,350, 400]

    elif varname == "bias":
        lev=np.arange(-15,15,1)
        cm = level_colormap(lev, cmap=plt.cm.get_cmap("RdBu_r"))
        units = 'Bias (Wm$^{2}$)'
    elif varname == "osa":
        lev=np.arange(0.01,0.9,0.001)
        cm = level_colormap(lev, cmap=plt.cm.get_cmap("RdBu_r"))
        units = 'Bias (Wm$^{2}$)'

    elif varname == "anomalies":
        lev = np.arange(-10, 10, 1)
        cm = level_colormap(lev, cmap=plt.cm.get_cmap("RdBu_r"))
        units = 'Anomalies (Wm$^{2}$)'
    else:
        units = ''
    fig = plt.figure(figsize=(12, 12))
    cm = level_colormap(lev, cmap=plt.cm.get_cmap("RdBu_r"))

    projection=ccrs.NorthPolarStereo()
    axes_class = (GeoAxes, dict(map_projection=projection))
    grids = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 4),
                    axes_pad=(0.6, 0.5),  # control padding separately for e.g. colorbar labels, axes titles, etc.
                    cbar_location='bottom',
                    cbar_mode="single",
                    cbar_pad="5%",
                    cbar_size='5%',
                    label_mode='')

    for i, grid_ax in enumerate(grids):

        # Get the regional domain to plot for the climatology plots and make sure that
        # we dont use too frequent ticks depending on the size of the region

        grid_ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

   #     grid_ax.set_xticks([*range(-180,180, delta_x)], crs=projection)
   #     grid_ax.set_yticks([*range(50,90, delta_y)], crs=projection)
      #  grid_ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
      #  grid_ax.yaxis.set_major_formatter(LatitudeFormatter())
      #  grid_ax.tick_params(axis="x", labelsize=8)
      #  grid_ax.tick_params(axis="y", labelsize=8)
     #  plt.grid(True, zorder=0, alpha=0.5)
        grid_ax.set_aspect(0.7)

        cs = grid_ax.contourf(clim.lon,
                     clim.lat,
                     clim[i,:,:],
                     levels=lev,
                    # locator=loc_ticker,
                     cmap=cm,
                     zorder=2,
                     alpha=1.0,
                     extend="both",
                     transform=ccrs.PlateCarree())

        # contour lines
        grid_ax.contour(clim.lon,
                        clim.lat,
                        clim[i,:,:], colors='k',
                        levels=lev, linewidths=0.1,
                        transform=ccrs.PlateCarree())

        grid_ax.set_title(months[i])
        grid_ax.add_feature(cfeature.LAND, color="grey", zorder=3)
        grid_ax.coastlines(resolution="110m", linewidth=0.2, color="black", alpha=1.0, zorder=4)

    cb = grids.cbar_axes[0].colorbar(cs, format='%.1f', label=units)

    if not os.path.exists(baseURL_output):
        os.makedirs(baseURL_output)


    plotfile = "{}/{}_{}_clim.png".format(baseURL_output, prefix, varname)
    print("[CMIP6_plot] Created plot {} ".format(plotfile))
    plt.savefig(plotfile, dpi=200, facecolor='w', transparent=False, bbox_inches='tight')
    plt.show()

def level_colormap(levels, cmap=None):
    """Make a colormap based on an increasing sequence of levels"""

    # Spread the colours maximally
    nlev = len(levels)
    S = np.arange(nlev, dtype='float') / (nlev - 1)
    A = cmap(S)

    # Normalize the levels to interval [0,1]
    levels = np.array(levels, dtype='float')
    L = (levels - levels[0]) / (levels[-1] - levels[0])

    # Make the colour dictionary
    R = [(L[i], A[i, 0], A[i, 0]) for i in range(nlev)]
    G = [(L[i], A[i, 1], A[i, 1]) for i in range(nlev)]
    B = [(L[i], A[i, 2], A[i, 2]) for i in range(nlev)]
    cdict = dict(red=tuple(R), green=tuple(G), blue=tuple(B))

    # Use
    return colors.LinearSegmentedColormap(
        '%s_levels' % cmap.name, cdict, 256)