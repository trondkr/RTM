
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pylab import *
from netCDF4 import Dataset
import datetime
import pandas as pd
import datetime as datetime
import pandas as pd
import glob
import string
from matplotlib.pyplot import cm 
#import seaborn as sns
import os

import seaborn as sns

__author__   = 'Trond Kristiansen'
__email__    = 'trond.kristiansen@imr.no'
__created__  = datetime.datetime(2015, 5, 29)
__modified__ = datetime.datetime(2017, 4, 26)
__version__  = "1.0"
__status__   = "Development, 29.5.2015, 08.07.2016, 26.04.2017"

def finalizePlot(myvar,LME,LMENAME,numberOfModels):
    plotfile='Figures/seasonal_boxplot_'+str(myvar)+'_LME_'+str(LMENAME)+'.pdf'
    plt.savefig(plotfile)
    print('Saved figure file {}'.format(plotfile))
    #plt.show()


def initializePlot(months,myvar,allData,LMENAME):
    plt.clf()
    figure(figsize=(12, 14))    
    matplotlib.rcParams.update({'font.size': 22})
    ax = subplot(111)  
    ax.spines["top"].set_visible(False)  
   # ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
   # ax.spines["left"].set_visible(False)
    # Ensure that the axis ticks only show up on the bottom and left of the plot.  
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
    #ax.get_xaxis().tick_bottom()  
    #ax.get_yaxis().tick_left() 
    # Limit the range of the plot to only where the data is.  
    # Avoid unnecessary whitespace.  
    if myvar=="tos":
        #ylabel('Temperature ($^\circ$C)')
        minimum = np.min(allData)
        maximum = np.max(allData)
  
    if myvar=="sos":
        minimum = -2.2
        maximum = 1.0
        
    if myvar=="intpp":
        #ylabel('PSU')
        minimum = -60
        maximum = 100
       
    if myvar=="sic":
        #ylabel('PSU')
        minimum = -20
        maximum = 120

   # import seaborn
   # seaborn.set(style='ticks')
    
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.  
    # plt.tick_params(axis="both", which="both", bottom="off", top="off",  
    #             labelbottom="on", left="off", right="off", labelleft="on")
    return ax

def getColorData():
    # These are the "Tableau 20" colors as RGB.  
    # http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(158, 200, 229),
             (158, 218, 130),(158, 218, 220),(58, 118, 229),(58, 118, 229),(58, 18, 129),(58, 18, 209),(58, 18, 2),(58, 118, 209)]  

    for i in range(len(tableau20)):  
            r, g, b = tableau20[i]  
            tableau20[i] = (r / 255., g / 255., b / 255.)
    return tableau20


"""" 

------------------------------------------------------------------
     MAIN
     Trond Kristiansen, 26.5.2015, 29.07.2015, 05.08.2015, 06.08.2015
     Trond.Kristiansen@imr.no
------------------------------------------------------------------

"""

dataDir="/Users/trondkr/Dropbox/Projects/Paper27/CMIP5/ESRL-DATA/RCP8.5/"

variables=["tos","sos","intpp","sic"]
#variables=["tos"]


LMES=[1,2,9,18,19,20,54,55,64]

LMENAMES=['East Bering Sea','Gulf of Alaska','Labrador - Newfoundland','Canadian Eastern Arctic - West Greenland','Greenland Sea',
        'Barents Sea','Northern Bering - Chukchi Seas','Beaufort Sea','Central Arctic']
LMEABBREVIATIONS=['EBS','GAK','LabN','WG','GS','BS','ChuckS','BeauS','ARC']
           
months=np.arange(0,12,1)
plotIndividualModels=False
runmn=1 #  runmn = 1, 5, 10, 20, 30 ;
historicalEndYear=2006


for LME, LMENAME in zip(LMES, LMENAMES):
    print("\n=> Exctrating data and plotting for LME {}".format(LMENAME)) 
    for variable in variables:
        first=True
        argument="%s*%s*.nc"%(dataDir,variable)

        allFiles = glob.glob(argument)
        allFiles.sort()
       
        allModels=[]
        savedENSMN=[]
        tableau20=getColorData()
    
        for myIndex,myFile in enumerate(allFiles):
           
            myCDF=Dataset(myFile)
            years=(myCDF.variables["year"][:])
            allYears=[]

            mydata = np.squeeze(myCDF.variables[variable][:,0:12,runmn,LME-1])
            myclim = np.squeeze(myCDF.variables["clim"][0:12,LME-1])
            #if variable in ['tos','sos','intpp']:
               # mydata = mydata  - myclim
           
            if (variable in ['intpp']):
                mydata = (mydata/myclim)*100.

            mydata = np.ma.masked_where(abs(mydata) > 1000, mydata)

            if first:
                for index,year in enumerate(years):
                    if int(year)==int(historicalEndYear):
                        historicalEndYearIndex=index
                        print("Found index {} as end for historical period".format(historicalEndYearIndex))
                        continue

                # no.models - no.year - no.months
                allDataHist = np.zeros((len(allFiles),12))
                allDataFuture = np.zeros((len(allFiles),12))
               
                first=False
            # Remove extreme outliers (NorESM shows crazy intpp results)
           # if (np.ma.max(mydata) < 400):
            
            # Calculate the monthly averages for each model
            allDataHist[myIndex,:] = np.mean(mydata[0:historicalEndYearIndex,:],0)
            allDataFuture[myIndex,:] = np.mean(mydata[historicalEndYearIndex:-1,:],0)
          
            head, tail = os.path.split(myFile)
            model=tail.split('.')
            modelName=model[1]

            allModels.append(modelName)
            
            if modelName=="ENSMN":
                numberOfModels = len(allModels)
                print("Extracting data from {} models".format(len(allModels)))
                savedENSMNHist = np.mean(mydata[0:historicalEndYearIndex,:],0)
                savedENSMNFuture = np.mean(mydata[historicalEndYearIndex:-1,:],0)

            myCDF.close()

        allDataHist = np.squeeze(np.ma.masked_where(abs(allDataHist) > 1000, allDataHist))
        allDataFuture = np.squeeze(np.ma.masked_where(abs(allDataFuture) > 1000, allDataFuture))

        # Convert the data to Pandas dataframe

        dtypes=[]
        for modelName in allModels:
            mytype="('%s','float32')"%(modelName)
            dtypes.append(mytype)

       # values = np.array(allDataFuture, dtype=dtypes)
        
        index = ['Model'+str(i) for i in range(1, len(allDataFuture[:,0])+1)]
      
        dffuture = pd.DataFrame(allDataFuture,index=index)
        dfhistory = pd.DataFrame(allDataHist,index=index)
     
       # ax = initializePlot(13,variable,allDataHist,LMENAME)
       
       # df.boxplot()
       # plt.show()
        #sns.set_style("whitegrid")
        clf()
        sns.set(style="ticks", palette="muted", color_codes=True)
        ax=sns.boxplot(data=dffuture, whis=np.inf, color="c")
       
        std = dfhistory.std()
        ax.fill_between(months, dfhistory.mean()-std, dfhistory.mean()+std, alpha=.1, color="grey")
       
        ax.plot(months,dfhistory.mean(),color="red",lw=None,marker='.')
        sns.boxplot(data=dffuture, whis=np.inf, color="c")
       
        labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Okt","Nov","Dec"]
      
        ax.annotate('N=%s'%(numberOfModels), xy=(years[0], np.min(allDataHist)), xytext=(3, 1.5))
        ax.set_xticklabels(labels)

        finalizePlot(variable,LME,LMENAME,numberOfModels)
        plt.show()
