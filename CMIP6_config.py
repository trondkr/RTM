import logging
import gcsfs
import numpy as np
import pandas as pd
import os

class Config_albedo:
    """
    Class that is passed to the CMIP6 calculations containing the configuration.
    """
    def __init__(self, create_forcing:bool = False, use_gcs: bool = False):
        """
        This function initialized the configuration for the CMIP6 calculations.
        """

        logging.info("[CMIP6_config] Defining the config file for the calculations")

        self.fs = gcsfs.GCSFileSystem(token="anon", access="read_only")
        self.grid_labels = ["gn"]  # Can be gr=grid rotated, or gn=grid native
        self.experiment_ids = ["ssp245"] #, "ssp585"] #, "ssp585"]
        self.source_id = None
        self.member_id = None
        self.create_forcing = create_forcing

        self.variable_ids = [
            "prw",
            "clt",
            "uas",
            "vas",
            "chl",
            "sithick",
            "siconc",
            "sisnthick",
            "sisnconc",
            "tas",
            "tos",
        ]  # ,"toz"]
        
        self.table_ids = [
            "Amon",
            "Amon",
            "Amon",
            "Amon",
            "Omon",
            "SImon",
            "SImon",
            "SImon",
            "SImon",
            "Amon",
            "Omon",
        ]
      #  self.table_ids = ["Amon","Amon"]
      #  self.variable_ids = ["rsus","rsds"]
        
        self.use_gcs = use_gcs
        self.bias_correct_ghi = False
        self.sensitivity_run = False
        if self.sensitivity_run:
            self.experiment_ids = ["ssp245"]
        self.dset_dict = {}
        self.start_date = "1979-01-01"
        self.end_date = "1982-12-16" #"2099-12-16"
        
        if self.sensitivity_run:
            # For sensitivity runs we do 40 year periods to 
            # evaluate the sensitivity from individual factors.
            self.end_date = "1989-01-16"
            self.scenarios = ["osa", "no_ice", "no_chl", "no_wind", "no_osa", "no_meltpond", "snow_sensitivity", "no_clouds"]
                           
        else:
            self.scenarios = ["osa"]
            
        # Change these to False if you want to download the CMIP6 data from the cloud
        # and write the files to disk.
        if self.create_forcing:
            logging.info("[CMIP6_config] Creating forcing files")
        else:
            logging.info("[CMIP6_config] Running light calculations")
        
        print("create_forcing", "True" if self.create_forcing else "False")
        self.use_local_CMIP6_files = False if self.create_forcing else True
        self.perform_light_calculations = False if self.create_forcing else True
        
        if not self.use_local_CMIP6_files and not self.perform_light_calculations:
            # This is used to create the input files for the light calculations.
            # On first run turn use_local_CMIP6_files and perform_light_calculations 
            # off and create files.
            self.write_CMIP6_to_file = True
        else:
            self.write_CMIP6_to_file = False
            
        self.cmip6_netcdf_dir = "../results"
        self.cmip6_outdir = "../results"
        if not self.bias_correct_ghi:
            self.cmip6_outdir = "../results/nobias"
        if self.sensitivity_run:
            self.cmip6_outdir = "../results/light_sensitivity"
        if os.path.exists(self.cmip6_outdir):
            os.makedirs(self.cmip6_outdir, exist_ok=True)
            
        # Cut the region of the global data to these longitude and latitudes
        if self.write_CMIP6_to_file:
            # We want to save the entire northern hemisphere for possible use later
            # while calculations are done north of 50N
            self.min_lat = 60
            self.start_date = "1950-01-01"
        else:
            self.min_lat = 60
        self.max_lat = 70
        self.min_lon = 180
        self.max_lon = 190

        # ESMF and Dask related
        self.interp = "bilinear"
        #self.outdir = f"/mnt/disks/actea-disk-1/{self.cmip6_outdir}"
        self.outdir = f"{self.cmip6_netcdf_dir}"
        if os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
            
        self.selected_depth = 0
        self.models = {}
        
        # Define the range of wavelengths that constitue the different parts of the spectrum. 
        self.start_index_uv = len(np.arange(200, 200, 10))
        self.end_index_uv = len(np.arange(200, 410, 10))
        self.start_index_uvb = len(np.arange(200, 280, 10))
        self.end_index_uvb = len(np.arange(200, 320, 10))
        self.start_index_uva = len(np.arange(200, 320, 10))
        self.end_index_uva = len(np.arange(200, 400, 10))
        self.start_index_visible = len(np.arange(200, 400, 10))
        self.end_index_visible = len(np.arange(200, 710, 10))
        self.start_index_nir = len(np.arange(200, 800, 10))
        self.end_index_nir = len(np.arange(200, 2500, 10))
        
        self.setup_erythema_action_spectrum()

    def setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def read_cmip6_repository(self):
        self.df = pd.read_csv(
            "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
        )

    def setup_parameters(self):
        # These are values of reflection from the ocean surface at various wavelengths
        # These are used to calculate the ocean surface albedo.
        wl = pd.read_csv(
            "../data/Wavelength/Fresnels_refraction.csv", header=0, sep=";", decimal=","
        )
        self.wavelengths = wl["λ"].values
        self.refractive_indexes = wl["n(λ)"].values
        self.alpha_chl = wl["a_chl(λ)"].values
        self.alpha_w = wl["a_w(λ)"].values
        self.beta_w = wl["b_w(λ)"].values
        self.alpha_wc = wl["a_wc(λ)"].values
        self.solar_energy = wl["E(λ)"].values
      
        self.fractions_shortwave_uv = self.solar_energy[self.start_index_uv:self.end_index_uv]
        self.fractions_shortwave_vis = self.solar_energy[
            self.start_index_visible:self.end_index_visible
        ]
        self.fractions_shortwave_nir = self.solar_energy[self.start_index_nir:self.end_index_nir]

        logging.info(
            "[CMIP6_config] Energy fraction UV ({} to {}): {:3.3f}".format(
                self.wavelengths[self.start_index_uv],
                self.wavelengths[self.end_index_uv],
                np.sum(self.fractions_shortwave_uv),
            )
        )

        logging.info(
            "[CMIP6_config] Energy fraction PAR ({} to {}): {:3.3f}".format(
                self.wavelengths[self.start_index_visible],
                self.wavelengths[self.end_index_visible],
                np.sum(self.fractions_shortwave_vis),
            )
        )

        #  logging.info("[CMIP6_config] Energy fraction NIR ({} to {}): {:3.3f}".format(self.wavelengths[self.start_index_nir],
        #                                                                               self.wavelengths[self.end_index_nir],
        #                                                                               np.sum(
        #                                                                                   self.fractions_shortwave_nir)))

        # Read in the ice parameterization for how ice absorbs irradiance as a function of wavelength.
        # Based on Perovich 1996
        ice_wl = pd.read_csv(
            "../data/ice-absorption/sea_ice_absorption_perovich_and_govoni_interpolated.csv",
            header=0,
            sep=",",
            decimal=".",
        )

        self.wavelengths_ice = ice_wl["wavelength"].values
        self.absorption_ice_pg = ice_wl["k_ice_pg"].values

    def setup_erythema_action_spectrum(self):
        # Spectrum suggested by:
        # A.F. McKinlay, A.F. and B.L. Diffey,
        # "A reference action spectrum for ultraviolet induced erythema in human skin",
        # CIE Research Note, 6(1), 17-22, 1987
        # https://www.esrl.noaa.gov/gmd/grad/antuv/docs/version2/doserates.CIE.txt
        # A = 	1		for  250 <= W <= 298
        # A = 	10^(0.094(298- W))	for 298 < W < 328
        # A = 	10^(0.015(139-W-))	for 328 < W < 400
        wavelengths = np.arange(200, 410, 10)
        self.erythema_spectrum = np.zeros(len(wavelengths))

        # https://www.nature.com/articles/s41598-018-36850-x
        for i, wavelength in enumerate(wavelengths):
            if 250 <= wavelength <= 298:
                self.erythema_spectrum[i] = 1.0
            elif 298 <= wavelength <= 328:
                self.erythema_spectrum[i] = 10.0 ** (0.094 * (298 - wavelength))
            elif 328 < wavelength < 400:
                self.erythema_spectrum[i] = 10.0 ** (0.015 * (139 - wavelength))
        logging.info(
            "[CMIP6_config] Calculated erythema action spectrum for wavelengths 290-400 at 10 nm increment"
        )

    def setup_ozone_uv_spectrum(self):
        # Data collected from Figure 4
        # http://html.rhhz.net/qxxb_en/html/20190207.htm#rhhz
        infile = "../data/ozone-absorption/O3_UV_absorption_edited.csv"
        df = pd.read_csv(infile, sep="\t")

        # Get values from dataframe
        o3_wavelength = df["wavelength"].values
        o3_abs = df["o3_absorption"].values

        wavelengths = np.arange(200, 410, 10)

        # Do the linear interpolation
        o3_abs_interp = np.interp(wavelengths, o3_wavelength, o3_abs)

        logging.info(
            "[CMIP6_config] Calculated erythema action spectrum for wavelengths 290-400 at 10 nm increment"
        )
      
        return o3_abs_interp, wavelengths

    def setup_absorption_chl(self):
        # Data exported from publication Matsuoka et al. 2007 (Table. 3)
        # Data are interpolated to a fixed wavelength grid that fits with the wavelengths of
        # Seferian et al. 2018
        infile = "../data/chl-absorption/Matsuoka2007-chla_wavelength_absorption.csv"
        df = pd.read_csv(infile, sep=" ")

        # Get values from dataframe
        chl_abs_A = df["A"].values
        chl_abs_B = df["B"].values
        chl_abs_wavelength = df["wavelength"].values

        # Interpolate to 10 nm wavelength bands - only visible
        # This is because all other wavelength calculations are done at 10 nm bands.
        # Original Matsuoka et al. 2007 operates at 5 nm bands.
        wavelengths = np.arange(400, 710, 10)

        # Do the linear interpolation
        A_chl_interp = np.interp(wavelengths, chl_abs_wavelength, chl_abs_A)
        B_chl_interp = np.interp(wavelengths, chl_abs_wavelength, chl_abs_B)

        return A_chl_interp, B_chl_interp, wavelengths

    # import matplotlib.pyplot as plt
    # plt.plot(self.wavelengths,self.solar_energy)
    # plt.title("Energy contributions from wavelengths 200-4000")

    # plt.savefig("energy_fractions.png", dpi=150)
