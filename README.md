# RTM

# Google Storage 
This code assumes that the user will be using Google Cloud Storage (GCS) to store results and to retrieve input files for running the RTM. This requires you to define the GCS credential file or path. I fnot you will get the error message "Your default credentials were not found. To set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information."
 
## Preparing the forcing for the RTM
### Extract variables and interpolate to cartesian grid
Before you can run the Radiative Transfer Model you need to create the required files used as forcing. The model takes 11 variables as input which is interpolated values from CMIP6 models. You can create these forcing files using the script `CMIP6_light.py` and by setting the configuration to (`CMIP6_config.py`):

```
self.use_local_CMIP6_files = False
self.write_CMIP6_to_file = True
self.perform_light_calculations = False
```
This will extract the required variables (11) necessary for each CMIP6 model and ensemble member (e.g., r1i1p1f1) 
and climate scenario (e.g. SSP245, SSP585) specified. For this paper we use the following combinations.
```
CMCC-ESM2: ["r1i1p1f1"]
CanESM5: ["r1i1p2f1" "r2i1p2f1" "r9i1p2f1" "r10i1p2f1" "r7i1p2f1" "r6i1p2f1" "r3i1p2f1"]
MPI-ESM1-2-LR: ["r10i1p1f2" "r1i1p1f1" "r2i1p1f1" "r4i1p1f2" "r5i1p1f1" "r6i1p1f2"]
UKESM1-0-LL: ["r1i1p1f2" "r2i1p1f2" "r3i1p1f2" "r4i1p1f2" "r8i1p2f1"]
MPI-ESM1-2-HR: ["r1i1p1f1" "r2i1p1f1"]
```
For each model, scenario, ensemble member, and variable we extract the data defined by the latitudinal and longitudinal boundaries defined in the configuration:
```
self.min_lat = 60
self.max_lat = 85
self.min_lon = 0
self.max_lon = 360
```
The variables we consider (11 in total) includes:
```
"prw","clt","uas","vas","chl","sithick","siconc","sisnthick","sisnconc","tas","tos"
```
Once you have run all of the combinations the forcing files will be ready. The files are automatically stored on Google Cloud storage (you have to define your own setup for this with permissions to upload to your own buckets).

## Running the RTM
### Note to run without disconnecting 
The code was run on Google VM instances which can occassionally dicsonnect to the SSH VScode window. To avoid disrupting
the run of the program when this happens run teh script using `nohup`:
`nohup /home/sam/miniconda3/envs/actea-3.9/bin/python CMIP6_light.py > output.txt &`
To start all simulations simulatenously, with a lag of 120 seconds between starts to avoid all processes reading the same
fields at the same time, use the script `run_all.sh`. This scripts creates individual log files for each CMIP6 `source_id` and `member_id` combination at path `logs/SOURCE_ID_MEMBER_ID.txt`. Its reccomended to run on a machine with +64MB RAM. 

### Calculate ocean surface albedo (OSA).
The recent study introduces an advanced methodology for upcoming climate models, focusing on the quantification of the impacts of solar zenith angle, oceanic wave dynamics, and chlorophyll levels on ocean surface albedo (OSA). This innovative approach aims to diminish the uncertainties associated with climate sensitivity concerning radiative energy flow. The newly proposed OSA model, incorporated into the Radiative Transfer Model (RTM) discussed in this document, performs spectral calculations of the contributions from the ocean surface to both direct and diffuse shortwave radiation. This enhancement allows for a more accurate representation of the shortwave radiation reflected by the ocean surface.

Here we use the new OSA approach by Seferian et al. 2018 to spectrally calculate the albedo at each 
grid point accounting for solar angle, wind/waves (surface roughness) and chlorophyll. The output provides OSA for 
direct and diffuse light for wavelengths 200-4000 nm. The OSA is then split into UV and VIS components 
based on wavelengths to be used in function `calculate_radiation`.

### Calculate irradiance
Using the output from OSA we can estimate the total incomin irradiance. This is done in multiple steps 
using main function `radiation` (`CMIP6_light.py`). This function returns an array of calculated diffuse and direct light for each longitude index around the globe (fixed latitude). The function takes latitude, time, pvlib system settings, cloud cover, albedo, and ozone as input to the calculations.

The radiation reaching the earth's surface can be represented in a number of different ways. Global Horizontal Irradiance (GHI) is the total amount of shortwave radiation received from above by a surface horizontal to the ground. This value is of particular interest to photovoltaic installations and includes both Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DIF). DNI is solar radiation that comes in a straight line from the direction of the sun at its current position in the sky. DIF is solar radiation that does not arrive on a direct path from the sun, but has been scattered by molecules and particles in the atmosphere and comes equally from all directions.

The impact of clouds on the amount of light through teh atmosphere is calculated using `pvlib.irradiance.campbell_norman` where `transmittance = (1.0 - cloud_covers) * 0.75`. The calculated `ghi` value is bias-corrected using the difference to `ERA5` atmospheric reanalysis of shortwave radiation.

The DISC algorithm (`pvlib.irradiance._disc_kn`) converts the GHI to direct normal irradiance (DNI) through empirical relationships between the global and direct clearness indices (`pvlib.irradiance.clearness_index`).

The DNI and DHI values are converted to a plane with tilt zero horizontal to the earth. This is done applying tilt=0 to POA calculations using the output from campbell_norman. The POA calculations include calculating sky and ground diffuse light where specific models can be selected (we use default). Here, the albedo is used to calculate ground diffuse irradiance.

Finally we account for cloud opacity on the spectral radiation in a function called `cloud_opacity_factor`.

### Total irradiance calculations
It is possible to add more accurate models for extra terrestrial light using various models when 
calculating the following:
```python
dni_extra = pvlib.irradiance.get_extra_radiation(time)

total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt,
                                                            surface_azimuth,
                                                            apparent_zenith,
                                                            azimuth,
                                                            irrads['dni'],
                                                            irrads['ghi'],
                                                            irrads['dhi'],
             dni_extra=dni_extra_array,
             model='haydavies')
```

If you need the angle of incidence:
```python
aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
                                      solpos['apparent_zenith'].to_numpy(), 	   solpos['azimuth'].to_numpy())
```
To reference the use of pvlib for light calculations use:
Holmgren, W., C. Hansen and M. Mikofski (2018). “pvlib Python: A python package for modeling solar energy systems.” 
Journal of Open Source Software 3(29): 884.

### Create weights for the ensembles
To weight the performance and the indepdendence of each model contribution to the overall ensemble we applied the ClimWIP package (https://github.com/lukasbrunner/ClimWIP). We 
compared how well each model was able to replicate the observed values of ocean surface temperature (`tos`), as well as the independence of
each model.  We compared the surface temperature with observations from two different datasets: 1) the NOAA Extended Reconstructed SST V5
 (ERSSTv5) which is a global monthly SST analysis from 1854 to the present derived from ICOADS data and 2) the Coriolis Ocean database for ReAnalysis (CORA5.2). CORA5.2 is a "...dataset of delayed time mode 
validated temperature and salinity in-situ measurements provided by the Coriolis datacenter and distributed by the Copernicus Marine service (https://www.coriolis.eu.org/Data-Products/Products/CORA)".

The final weights file should be in directory and named `data/Calculated_weights.nc`.

### Unittests
Several unittests has been written to verify that the functions provide the expected results. These are all written as `pytest` and can be run simply as: `pytest`. As of May 14th 2024 there are 33 unittests.

### Create weighted ensembles
After running the RTM for a set of models and scenarios we can calculate the weighted ensembles of the model outputs. This is done running the Python notebook `CMIP6_create_ensembles.ipynb`. You can select to create ensembles with and without weights for comparison.

### Create final plots
The final weighted ensembles and statistics can be plotted using two main Python notebooks: `CMIP6_plot_seaonal_light.ipynb` and `CMIP6_plot_light_results.ipynb`. Other scripts like `CMIP6_plot_forcing.ipynb` plots the forcing for running the RTM, `CMIP6_plot_optimal_growth.ipynb` calculates and plots the impact of the environment on eggs and juveniel fish.

### Useful links
http://www.matteodefelice.name/post/aggregating-gridded-data/
https://cds.climate.copernicus.eu/toolbox/doc/index.html
https://www.toptal.com/python/an-introduction-to-mocking-in-python
https://esmtools.readthedocs.io/en/latest/examples/pco2.html
earthsystemmodeling.org/esmf\_releases/last\_built/esmpy\_doc/html/examples.html
https://github.com/Quick/Nimble#truthiness
https://csdms.colorado.edu/w/images/CICE_documentation_and_software_user's_manual.pdf

[image-1]:	https://badge.buildkite.com/998b597662a8db957ab524d2660958105de691cc0bc1753594.svg
[image-2]:	https://codebeat.co/badges/8bf4f052-6579-47fa-a552-b221154549c0
[image-3]:	https://codecov.io/gh/trondkr/cmip6-albedo/branch/master/graph/badge.svg