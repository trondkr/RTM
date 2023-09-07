import unittest
import CMIP6_ccsm3
import numpy as np
import logging


class MyCMIP6CCSM3(unittest.TestCase):
    def setUp(self) -> None:
        self.lons = np.arange(10, 20, 1)
        self.lats = np.arange(75, 80, 1)
        self.ice_thickness = np.ones(
            (len(self.lats), len(self.lons))
        )  # np.random.random((len(self.lats), len(self.lons)))
        self.snow_thickness = self.ice_thickness / 10.0

        # Constant ocean albedo of 0.06
        self.ocean_albedo = np.ones(np.shape(self.snow_thickness)) * 0.06
        # slightly warmer than melting so we get the effect of melt ponds
        self.air_temp = np.ones(np.shape(self.snow_thickness)) - 10.0
        self.sea_ice_concentration = np.ones(np.shape(self.snow_thickness))
        self.snow_concentration = self.sea_ice_concentration

        self.direct_nir = np.ones(np.shape(self.snow_thickness)) * 1.0
        self.direct_vis = np.ones(np.shape(self.snow_thickness)) * 10.0
        self.diffuse_nir = np.ones(np.shape(self.snow_thickness)) * 0.5
        self.diffuse_vis = np.ones(np.shape(self.snow_thickness)) * 5

        self.cmip6_ccsm3 = CMIP6_ccsm3.CMIP6_CCSM3()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def test_init_correct(self):
        expected_formatted_name = "ccsm3"
        self.assertEqual(self.cmip6_ccsm3.shortwave, expected_formatted_name)

    def test_albedo_correct_when_sea_ice_concentration_is_zero(self):
        self.sea_ice_concentration = np.zeros(np.shape(self.snow_thickness))
        self.snow_thickness = np.zeros(np.shape(self.snow_thickness))
        self.ice_thickness = np.zeros(np.shape(self.snow_thickness))
        self.air_temp = np.ones(np.shape(self.snow_thickness)) * (-20.0)
        albo_dr = self.cmip6_ccsm3.direct_and_diffuse_albedo_from_snow_and_ice(
            self.ocean_albedo,
            self.snow_concentration,
            self.snow_thickness,
            self.sea_ice_concentration,
            self.ice_thickness,
            self.air_temp
        )

        np.testing.assert_almost_equal(albo_dr, self.ocean_albedo, decimal=2)

    def test_albedo_correct_when_sea_air_temp_is_very_cold_and_full_ice_snow_cover(
        self,
    ):
        # Very thick ice
        self.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.snow_concentration = self.sea_ice_concentration
        # Make sure the effect of pond melting is absolutely none
        self.air_temp = np.ones(np.shape(self.snow_thickness)) - 20.0
        albo_dr = self.cmip6_ccsm3.direct_and_diffuse_albedo_from_snow_and_ice(
            self.ocean_albedo,
            self.snow_concentration,
            self.snow_thickness,
            self.sea_ice_concentration,
            self.ice_thickness,
            self.air_temp
        )
        
        np.testing.assert_almost_equal(
            albo_dr, np.ones(np.shape(self.snow_thickness)) * 0.96, decimal=2
        )

    def test_albedo_correct_when_sea_air_temp_is_very_cold_and_full_ice_no_snow_cover(
        self,
    ):
        # Very thick ice
        self.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.snow_concentration = self.ice_thickness * 0.0
        self.snow_thickness = self.ice_thickness * 0.0

        # Make sure the effect of pond melting is absolutely none
        self.air_temp = np.ones(np.shape(self.snow_thickness)) - 2.0
        albo_dr = self.cmip6_ccsm3.direct_and_diffuse_albedo_from_snow_and_ice(
            self.ocean_albedo,
            self.snow_concentration,
            self.snow_thickness,
            self.sea_ice_concentration,
            self.ice_thickness,
            self.air_temp
        )

        np.testing.assert_almost_equal(
            albo_dr, np.ones(np.shape(self.snow_thickness)) * 0.73, decimal=2
        )
      

    def test_albedo_correct_when_sea_air_temp_is_very_warm_and_full_ice_no_snow_cover(
        self,
    ):
        # https://csdms.colorado.edu/w/images/CICE_documentation_and_software_user's_manual.pdf
        # Very thick ice, but very warm weather creating melt ponds that drastically decreases
        # the albedo from sea-ice (to 0.06)
        self.ice_thickness = np.ones(np.shape(self.snow_thickness))
        self.snow_concentration = self.snow_thickness*0.0
        self.snow_thickness = self.snow_concentration*0.0

        # Make sure the effect of pond melting is absolutely none
        self.air_temp = np.ones(np.shape(self.snow_thickness)) + 20.0
        albo_dr = self.cmip6_ccsm3.direct_and_diffuse_albedo_from_snow_and_ice(
            self.ocean_albedo,
            self.snow_concentration,
            self.snow_thickness,
            self.sea_ice_concentration,
            self.ice_thickness,
            self.air_temp
        )

        np.testing.assert_almost_equal(
            albo_dr, np.ones(np.shape(self.snow_thickness)) * 0.66, decimal=2
        )
       
    def test_albedo_correct_when_very_thin_ice_no_snow_cover_converge_to_ocean_osa(
        self,
    ):
        # https://csdms.colorado.edu/w/images/CICE_documentation_and_software_user's_manual.pdf
        # Very thick ice, but very warm weather creating melt ponds that drastically decreases
        # the albedo from sea-ice (to 0.06)
        self.ice_thickness = np.zeros(np.shape(self.snow_thickness)) + 0.002
        self.snow_concentration = self.snow_thickness*0.0
        self.snow_thickness = self.snow_concentration*0.0

        # Make sure the effect of pond melting is absolutely none
        self.air_temp = np.ones(np.shape(self.snow_thickness))-10
        albo_dr = self.cmip6_ccsm3.direct_and_diffuse_albedo_from_snow_and_ice(
            self.ocean_albedo,
            self.snow_concentration,
            self.snow_thickness,
            self.sea_ice_concentration,
            self.ice_thickness,
            self.air_temp
        )

        np.testing.assert_almost_equal(
            albo_dr, np.ones(np.shape(self.snow_thickness)) * 0.066, decimal=2
        )

    def test_albedo_correct_when_thin_ice_but_more_than_10cm_no_snow_cover(
        self,
    ):
        # https://csdms.colorado.edu/w/images/CICE_documentation_and_software_user's_manual.pdf
        # Very thick ice, but very warm weather creating melt ponds that drastically decreases
        # the albedo from sea-ice (to 0.06)
        self.ice_thickness = np.zeros(np.shape(self.snow_thickness)) + 0.1
        self.snow_concentration = self.snow_thickness*0.0
        self.snow_thickness = self.snow_concentration*0.0

        # Make sure the effect of pond melting is absolutely none
        self.air_temp = np.ones(np.shape(self.snow_thickness))-10
        albo_dr = self.cmip6_ccsm3.direct_and_diffuse_albedo_from_snow_and_ice(
            self.ocean_albedo,
            self.snow_concentration,
            self.snow_thickness,
            self.sea_ice_concentration,
            self.ice_thickness,
            self.air_temp
        )

        np.testing.assert_almost_equal(
            albo_dr, np.ones(np.shape(self.snow_thickness)) * 0.29, decimal=2
        )

    def test_calculate_diffuse_albedo_per_grid_point_all_snow(self):
        # Thick ice, full area cover, full snow cover
        self.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.sea_ice_concentration = np.ones(np.shape(self.sea_ice_concentration))
        self.snow_concentration = np.ones(np.shape(self.snow_thickness))
        self.snow_thickness = self.snow_concentration

        albedo = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(
            self.snow_concentration, self.sea_ice_concentration
        )

        np.testing.assert_almost_equal(
            albedo, np.ones(np.shape(self.snow_thickness)) * 0.96, decimal=2
        )

    def test_calculate_diffuse_albedo_per_grid_point_all_ice_no_snow(self):
        # Thick ice, full area cover, full snow cover
        self.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.sea_ice_concentration = np.ones(np.shape(self.sea_ice_concentration))
        self.snow_concentration = self.ice_thickness * 0.0
        self.snow_thickness = self.snow_concentration

        albedo = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(
            self.snow_concentration, self.sea_ice_concentration
        )

        np.testing.assert_almost_equal(
            albedo, np.ones(np.shape(self.snow_thickness)) * 0.73, decimal=2
        )

    def test_calculate_diffuse_albedo_per_grid_point_all_ice_except_one_grid_cell(self):
        # Thick ice, full area cover, full snow cover with the exception
        # of one grid cell that should return normal ocean albedo (0.06)
        self.ice_thickness = np.ones(np.shape(self.snow_thickness))
        self.sea_ice_concentration = np.ones(np.shape(self.sea_ice_concentration))
        self.snow_concentration = self.sea_ice_concentration * 0.0
        self.snow_thickness = self.snow_concentration

        self.snow_thickness[0, 0] = 0
        self.snow_concentration[0, 0] = 0
        self.ice_thickness[0, 0] = 0
        self.sea_ice_concentration[0, 0] = 0

        albedo = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(
            self.snow_concentration, self.sea_ice_concentration
        )

        result_should_be = np.ones(np.shape(self.snow_thickness)) * 0.73
        result_should_be[0, 0] = 0.06
        np.testing.assert_almost_equal(albedo, result_should_be, decimal=2)

    def test_calculate_chl_attenuated_shortwave(self):
        wavelengths = np.arange(400, 710, 10)
        dr = np.ones((len(wavelengths), 3, 2))
        chl = np.zeros((3, 2))
        depth = 0.1
        res = self.cmip6_ccsm3.calculate_chl_attenuated_shortwave(dr, chl, depth)
    
        np.testing.assert_almost_equal(res, dr, decimal=2)

    def test_calculate_attenuation_from_chl_with_high_chl(self):
        wavelengths = np.arange(400, 710, 10)
        dr = np.ones((len(wavelengths), 3, 2))
        chl = np.zeros((3,2)) + 2.0* 1.0e-6  # have to convert to kg/m3
        depth = 0.1
        res = self.cmip6_ccsm3.calculate_chl_attenuated_shortwave(dr, chl, depth)
        result_should_be = dr * 0.99

        np.testing.assert_almost_equal(res, result_should_be, decimal=2)


if __name__ == "__main__":
    unittest.main()
