import matplotlib
matplotlib.use('Agg')
from pylab import *
import os, datetime
import numpy as np
from netCDF4 import Dataset, num2date
import mpl_util
import sys
import pandas as pd
import xarray as xr

import calendar

__author__ = 'Trond Kristiansen'
__email__ = 'trond.kristiansen@niva.no'
__created__ = datetime.datetime(2017, 12, 20)
__modified__ = datetime.datetime(2017, 12, 20)
__version__ = "1.1"
__status__ = "Development, 20.12.2017, 05.04.2019"

"""This script calculates change in habitat from historical values. Habitat is defined
as a range of temperature and light in the water column for 4 seasons. 

This script requires the output from running the two scripts:
1. calculateMaxLightEntireArctic.py
2. removeSeasonality.sh

"""


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    # turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


def plotTimeseries(ts, myvar, season):
    ts_annual = ts.resample("A")
    ts_quarterly = ts.resample("Q")
    ts_monthly = ts.resample("M")

    # Write data to file
    mypath = "%s_annualaverages.csv" % (myvar)
    if os.path.exists(mypath): os.remove(mypath)
    ts.to_csv(mypath)
    print(("Wrote timeseries to file: %s" % (mypath)))

    red_purple = brewer2mpl.get_map('RdPu', 'Sequential', 9).mpl_colormap
    colors = red_purple(np.linspace(0, 1, 12))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # for mymonth in xrange(12):
    #    ts[(ts.index.month == mymonth + 1)].plot(marker='o', color=colors[mymonth], markersize=5, linewidth=0,
    #                                            alpha=0.8)
    # ts_annual.plot(marker='o', color="#FA9D04", linewidth=0, alpha=1.0, markersize=7, label="Annual")
    remove_border(top=False, right=False, left=True, bottom=True)
    ts.resample("M").mean().plot(style="r", marker='o', linewidth=1, label="Monthly")
    ts.resample("A").mean().plot(style="b", marker='o', linewidth=2, label="Annual")

    # legend(loc='best')
    if myvar == "light":
        ylabel(r'Light (W m$^{-2})$')

    if myvar == "temp":
        ylabel(r'Temp ($^{o}$C)')

    plotfile = 'figures/timeseries_' + str(season) + '_' + str(myvar) + '.png'
    plt.savefig(plotfile, dpi=300, bbox_inches="tight", pad_inches=0)


def getData(infile,infile_grid):
    if os.path.exists(infile):
        try:
            cdf = Dataset(infile)
            cdf_grid = Dataset(infile_grid)
            print(("Opened inputfile: %s" % (infile)))
        except:
            print(("Unable to  open file: %s" % (infile)))
            sys.exit()

        temp = cdf_grid.variables["tos"][:]
        temp_anomaly = cdf.variables["tos"][:]
        light = cdf.variables["light"][:]
        times = cdf.variables["time"][:]
        longitude = cdf_grid.variables["longitude"][:]
        latitude = cdf_grid.variables["latitude"][:]
        
        dates = num2date(times, "days since 1948-01-01 00:00:00", calendar="365_day")
        print("Extracted time-steps starting in %s and ending in %s" % (dates[0], dates[-1]))

        return temp,temp_anomaly,light,dates,longitude,latitude


def getStartAndEndIndex(startYear, endYear, dates):
    startIndex = -9;
    endIndex = -9
    for dateIndex, JD in enumerate(dates):

        if JD.year == startYear:
            startIndex = dateIndex
        if JD.year == endYear:
            endIndex = dateIndex
    if startIndex == -9 or endIndex == -9:
        print(("Unable to find indexes for start %s and end %s years", startYear, endYear))
        sys.exit()

    print(("=> Period %s to %s" % (dates[startIndex], dates[endIndex])))
    return startIndex, endIndex


def createDecadalAverages(seasonsArray, seasons, latitude, longitude, dates, periods):
    # New array to store values will contain seasonal values of tos and light as well as standard deviations
    # within each period of each.
    decadalArray = np.zeros((6, len(seasons), len(periods)-1, len(latitude), len(longitude)))
    decadalBatchSizes = np.zeros((len(periods)-1))


    for seasonIndex, season in enumerate(seasons):

        for periodIndex in range(len(periods)-1):
            startIndex, endIndex = getStartAndEndIndex(periods[periodIndex], periods[periodIndex + 1], dates)
            # 0= TOS, 1=STD(TOS), 2 = LIGHT, 3=STD(LIGHT), 4=Normalized tos, 5=normalized light

            temp_anomaly_period = np.squeeze(seasonsArray[0, seasonIndex, startIndex:endIndex, :, :])
            light_anomaly_period = np.squeeze(seasonsArray[1, seasonIndex, startIndex:endIndex, :, :])
            temp_period = np.squeeze(seasonsArray[2, seasonIndex, startIndex:endIndex, :, :])
           
            decadalArray[0, seasonIndex, periodIndex, :, :] = np.squeeze(
                np.ma.mean(temp_anomaly_period, axis=0))
            decadalArray[1, seasonIndex, periodIndex, :, :] = np.squeeze(
                np.ma.std(temp_anomaly_period, axis=0))
            decadalArray[2, seasonIndex, periodIndex, :, :] = np.squeeze(
                np.ma.mean(light_anomaly_period, axis=0))
            decadalArray[3, seasonIndex, periodIndex, :, :] = np.squeeze(
                np.ma.std(light_anomaly_period, axis=0))
            decadalArray[4, seasonIndex, periodIndex, :, :] = np.squeeze(
                np.ma.mean(temp_period, axis=0))
            decadalArray[5, seasonIndex, periodIndex, :, :] = np.squeeze(
                np.ma.std(temp_period, axis=0))
           
            decadalBatchSizes[periodIndex]=endIndex - startIndex
            decadalArray = ma.masked_invalid(decadalArray)
           
    return decadalArray, periods, decadalBatchSizes

def estimateChangedEcosystem(decadalArray, seasons, periods,decadalBatchSizes):
    # Size: [variables, seasons, period diffs, lat, long]
    # the masking of the array is required for defining the temperature range where 
    # polar cod is found (temp<3C)
    estimateChangeArray = np.ma.empty((3, np.shape(decadalArray[0, :, 0, 0, 0])[0],
                                    np.shape(decadalArray[0, 0, :, 0, 0])[0]-1,
                                    np.shape(decadalArray[0, 0, 0, :, 0])[0],
                                    np.shape(decadalArray[0, 0, 0, 0, :])[0]))

    for seasonIndex, season in enumerate(seasons):
        print("Calculating for season: ", season)
        x1 = np.squeeze(decadalArray[0,seasonIndex,0,:,:])
        y1 = np.squeeze(decadalArray[2,seasonIndex,0,:,:])
        x1_std = np.squeeze(decadalArray[1, seasonIndex,0,:,:])
        y1_std = np.squeeze(decadalArray[3,seasonIndex,0,:,:])
        s1=decadalBatchSizes[0]

        for periodIndex in range(1, np.shape(decadalArray[0, 0, :, 0, 0])[0]):
            s2=decadalBatchSizes[periodIndex]
            temp = np.squeeze(decadalArray[4,seasonIndex,periodIndex,:,:])

            x2 = np.squeeze(decadalArray[0,seasonIndex,periodIndex,:,:])
            y2 = np.squeeze(decadalArray[2,seasonIndex,periodIndex,:,:])
            x2_std = np.squeeze(decadalArray[1, seasonIndex,periodIndex,:,:])
            y2_std = np.squeeze(decadalArray[3,seasonIndex,periodIndex,:,:])
            
            # Calculate the combined effect of CHANGES  in light and temperature combined.
            # This identifies hotspots of change related to polar cod habitats.
            g1=calculateHedgesG(s1,s2,x1,x2,x1_std,x2_std)
            g2=calculateHedgesG(s1,s2,y1,y2,y1_std,y2_std)
            estimateChangeArray[2,seasonIndex,periodIndex-1,:,:]=np.sqrt(g1*g2)
            estimateChangeArray[0,seasonIndex,periodIndex-1,:,:]=x2
            estimateChangeArray[1,seasonIndex,periodIndex-1,:,:]=y2
       
            # Mask out areas not suitable for polar cod habitats. The nasking requires that 
            # the estimateChangeArray is defined using masked array: estimateChangeArray = np.ma.empty(....)
            maskedHabitats=np.squeeze(estimateChangeArray[2,seasonIndex,periodIndex-1,:,:])
            habitat_masked_by_temp=np.ma.masked_where(temp>3.0,maskedHabitats)
            estimateChangeArray[2,seasonIndex,periodIndex-1,:,:]=habitat_masked_by_temp
         
    estimateChangeArray = np.ma.masked_invalid(estimateChangeArray)

    return estimateChangeArray

def calculateHedgesG(s1,s2,x1,x2,x1_std,x2_std):
    # Calculate hedges g
    # https://en.wikipedia.org/wiki/Effect_size#Hedges'_g
    s = np.sqrt(((s2-1)*np.power(x2_std,2) + (s1-1)*np.power(x1_std,2))/(s1+s2-2.0))
    return (x2-x1)/s

def plotMap(lon, lat, mydata, period, qctype, season,hotspotsLon=None, hotspotsLat=None,hostpotsVals=None):
    plt.figure(figsize=(12, 12), frameon=False)
    mymap = Basemap(projection='npstere', lon_0=0, boundinglat=50)
    llat, llon = np.meshgrid(lat, lon)
 
    x, y = mymap(llon, llat)
    step=(np.max(mydata) - np.min(mydata))/10.
    
    if step >0:
        print("Plotting season: {} for period {}".format(season, period))
  
        levels = np.arange(np.min(mydata), np.max(mydata), step)
        levels = np.arange(-1,6,0.2)

        print("Number of masked points {}".format(np.sum(mydata.mask==True)))
        CS1 = mymap.contourf(x, y, mydata,levels,
                            cmap=mpl_util.LevelColormap(levels, cmap=cm.RdBu_r))
  
        mymap.drawparallels(np.arange(-90., 120., 15.), labels=[1, 0, 0, 0])  # draw parallels
        mymap.drawmeridians(np.arange(0., 420., 30.), labels=[0, 1, 0, 1])  # draw meridians

        mymap.drawcoastlines()
        mymap.drawcountries()
        if not (hotspotsLat is None and hotspotsLon is None):
            hlon,hlat = mymap(hotspotsLon, hotspotsLat)
            mymap.scatter(hlon, hlat, s=100, alpha=0.5, color='r', edgecolors='k')
        mymap.fillcontinents(color='grey', alpha=0.2)
        plt.colorbar(CS1, shrink=0.5)
        title('QC:' + str(period) + ' season:' + str(season))

        CS1.axis = 'tight'
        if not os.path.exists("Figures"):
            os.mkdir("Figures/")
        plotfile = 'Figures/map_qc_' + str(period) + '_season_' + str(season) + '.png'

        plt.savefig(plotfile, dpi=100)
        plt.clf()
        plt.close()

def createTimeseriesAtHotspots(Y,lon,lat):
    nhotspots=10
    lats, lons = np.meshgrid(lat, lon)
    # https://github.com/numpy/numpy/issues/9283

    YFlat=np.ravel(Y)
    LatsFlat=np.ravel(lats)
    LonsFlat=np.ravel(lons)
    
    ind = np.argpartition(YFlat, -nhotspots)[-nhotspots:]

  #  for c,i in enumerate(ind):
  #      print("Hotspot {}: index {} lat/lon:({},{}) value: {}".format(c,i,LatsFlat[i],LonsFlat[i],YFlat[i]))

    return LonsFlat[ind],LatsFlat[ind],YFlat[ind]

def getPointTimeseries(temp,light,time,lat,lon,hotspotsLat,hotspotsLon,periodName,season):
    
    daTemp = xr.DataArray(temp,
                  dims=['time','latitude','longitude'],
                  coords={'longitude': lon, 'latitude':lat,'time':time})
    daLight = xr.DataArray(light,
                  dims=['time','latitude','longitude'],
                  coords={'longitude': lon, 'latitude':lat,'time':time})
    for hs in range(len(hotspotsLat)):
        fig, axes = plt.subplots(ncols=2)
        print("=> Creating timeseries plot for location ({},{})".format(hotspotsLon[hs],hotspotsLat[hs]))
        dsT=daTemp.sel(latitude=hotspotsLat[hs], longitude=hotspotsLon[hs], method='nearest').resample(time='1Y').mean()
        dsL=daLight.sel(latitude=hotspotsLat[hs], longitude=hotspotsLon[hs], method='nearest').resample(time='1Y').mean()
    
        dsT.plot(color='purple', marker='.',ax=axes[0],lineWidth=0.3)
        dsL.plot(color='red', marker='.',ax=axes[1],lineWidth=0.3)
        if not os.path.exists("Timeseries"):os.mkdir("Timeseries/")
        plotfile='Timeseries/timeseries_{}_season_{}_hotspot({},{}).png'.format(periodName,season,hotspotsLon[hs],hotspotsLat[hs])
        if os.path.exists(plotfile):os.remove(plotfile)
        plt.savefig(plotfile, dpi=100, bbox_inches='tight')
        plt.clf()
    print
def createSeasonArrays(temp,temp_anomaly,light,dates,longitude,latitude):
    periods = [1950, 2000, 2025, 2050, 2075]

    winter = ["Jan", "Feb", "Mar"]
    spring = ["Apr", "May", "Jun"]
    summer = ["Jul", "Aug", "Sep"]
    autumn = ["Oct", "Nov", "Dec"]

    seasons = [winter, spring, summer, autumn]
    seasonNames = ["winter", "spring", "summer", "autumn"]

    count = 0
    last_year = -9
    years = []
    for d in dates:
        if d.year > last_year and d.month == 1:
            last_year = d.year

            years.append(datetime.datetime(d.year, d.month, d.day))
            count += 1
    count += 1
    print(("Timeseries contains %s years" % (count)))

    seasonsArray = np.zeros((3, len(seasons), count, len(latitude), len(longitude)))

    for seasonIndex, season in enumerate(seasons):
        tindex = 0
        seasonName = seasonNames[seasonIndex]
        tempT = np.zeros((len(latitude), len(longitude)))
        temp_anomalyT = np.zeros((len(latitude), len(longitude)))
        temp_anomalyL = np.zeros((len(latitude), len(longitude)))
        yearFinished = -9
        counter = 0

        for dateIndex, JD in enumerate(dates):

            if calendar.month_abbr[JD.month] in season:
                temp_anomalyT = temp_anomalyT + temp_anomaly[dateIndex, :, :]
                temp_anomalyL = temp_anomalyL + light[dateIndex, :, :]
                tempT = tempT + temp[dateIndex, :, :]
                counter += 1
            else:
                if yearFinished != JD.year:
                    if counter > 0:
                        seasonsArray[0, seasonIndex, tindex, :, :] = temp_anomalyT / counter * 1.0 
                        seasonsArray[1, seasonIndex, tindex, :, :] = temp_anomalyL / counter * 1.0
                        seasonsArray[2, seasonIndex, tindex, :, :] = tempT / counter * 1.0

                    tindex += 1
                    tempT = tempT * 0.0
                    temp_anomalyT = temp_anomalyT * 0.0
                    temp_anomalyL = temp_anomalyL * 0.0
                    counter = 0
                    yearFinished = JD.year

        createTimeseriesPlot = False
        if createTimeseriesPlot:
            ll = []
            tt = []

            for i in range(len(years)):
                tt.append(np.ma.mean(seasonsArray[0, seasonIndex, i, :, :]))
                ll.append(np.ma.mean(seasonsArray[1, seasonIndex, i, :, :]))

            tsl = pd.Series(ll, years)
            plotTimeseries(tsl, "light", seasonName)

            tst = pd.Series(tt, years)
            plotTimeseries(tst, "temp", seasonName)


    decadalArray,periods,decadalBatchSizes = createDecadalAverages(seasonsArray, seasons, latitude, longitude, years, periods)
    estimateChangeArray = estimateChangedEcosystem(decadalArray,seasons,periods,decadalBatchSizes)

    plotvars = ["tosQC", "lightQC", "combinedQC"]
   
    for i in range(len(plotvars)):
        for seasonIndex, season in enumerate(seasons):

            # If you have 4 reference dates 1950, 2000, 2050, 2100, then the periodIndex here refers to
            # the difference periods between each date. E.g. 0 = 1950-2000, 1=2000-2050 etc.
            # One less than perdiods definition.
            for periodIndex in range(np.shape(estimateChangeArray[0,0,:,0,0])[0]):
                
                mydata = np.squeeze(estimateChangeArray[i,seasonIndex,periodIndex,:,:])
                mydata = np.rot90(np.flipud(mydata), 3)
                hotspotsLon, hotspotsLat, hostpotsVals = createTimeseriesAtHotspots(mydata,longitude,latitude)

                periodName = "{}-{}-{}".format(plotvars[i], periods[periodIndex+1], periods[periodIndex + 2])
                print("Periodname {}".format(periodName))
                plotMap(longitude, latitude, mydata, periodName, plotvars[i], seasons[seasonIndex], hotspotsLon, hotspotsLat,hostpotsVals)
                if i==2:

                    print(hotspotsLon)
                    print(hotspotsLat)
                 #   getPointTimeseries(temp,light,dates,latitude,longitude,hotspotsLon,hotspotsLat,periodName,season)

print(('Python %s on %s' % (sys.version, sys.platform)))

infile = "Light_and_temperature_1850_2100_Arctic_delta.nc"
infile_grid = "Light_and_temperature_1850_2100_Arctic.nc"
temp,temp_anomaly,light,dates,longitude,latitude = getData(infile,infile_grid )

createSeasonArrays(temp,temp_anomaly,light,dates,longitude,latitude)
