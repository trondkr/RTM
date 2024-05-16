import unittest
import CMIP6_light
import CMIP6_config
import CMIP6_model
import CMIP6_IO
from datetime import datetime
import numpy as np
import xarray as xr
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from argparse import Namespace

# Unittest for ``CMIP6_light` setup

class TestCMIP6_light(unittest.TestCase):
    def setUp(self):

        self.cmip6 = CMIP6_light.CMIP6_light(Namespace(source_id="MPI-ESM1-2-LR", member_id="r4i1p1f1"))
        self.cmip6_model=CMIP6_model.CMIP6_MODEL(name="test")
        self.cmip6_IO = CMIP6_IO.CMIP6_IO()
        self.query_string = "source_id=='ACCESS-ESM1-5'and table_id=='Amon' and grid_label=='gn' \
                and experiment_id=='historical' and variable_id=='uas'"

class TestMethods(TestCMIP6_light):

    def test_create_data_array(self):
        self.assertIsNotNone(self.cmip6)
        self.assertIsNotNone(self.cmip6.config)

class TestIO(TestCMIP6_light):

    def test_query_returns_dataset_with_timesteps(self):
        # We avoid calling the read_cmip6_repository in other tests as it is time consuming
        try:
            self.cmip6.config.df
        except:
            self.cmip6.config.read_cmip6_repository()
        ds_hist = self.cmip6_IO.perform_cmip6_query(self.cmip6.config, self.query_string)
        self.assertIsNotNone(ds_hist)
        self.assertIsInstance(ds_hist, xr.Dataset)
        self.assertTrue(len(ds_hist["time"].values) > 0)

class TestModel(TestCMIP6_light):

    def test_access_to_variable_id_using_keys(self):
        # Make sure dictionary is empty at init
        self.assertFalse(self.cmip6_model.ds_sets)

class TestInit(TestCMIP6_light):

    def test_verify_equal_length_variable_and_table_ids(self):
        self.assertEqual(len(self.cmip6.config.variable_ids), len(self.cmip6.config.table_ids))

    def test_setup_parameters(self):
        self.cmip6.config.setup_parameters()
        self.assertIsNotNone(self.cmip6.config.wavelengths)
        self.assertIsNotNone(self.cmip6.config.alpha_chl)
        self.assertIsNotNone(self.cmip6.config.alpha_w)
        self.assertIsNotNone(self.cmip6.config.beta_w)
        self.assertIsNotNone(self.cmip6.config.alpha_wc)
        self.assertIsNotNone(self.cmip6.config.solar_energy)

    def test_shape_of_arrays_equal_after_setup_parameters(self):
        self.cmip6.config.setup_parameters()
        self.assertEqual(np.shape(self.cmip6.config.wavelengths), np.shape(self.cmip6.config.alpha_chl))
        self.assertEqual(np.shape(self.cmip6.config.wavelengths), np.shape(self.cmip6.config.alpha_w))
        self.assertEqual(np.shape(self.cmip6.config.wavelengths), np.shape(self.cmip6.config.beta_w))
        self.assertEqual(np.shape(self.cmip6.config.wavelengths), np.shape(self.cmip6.config.alpha_wc))
        self.assertEqual(np.shape(self.cmip6.config.wavelengths), np.shape(self.cmip6.config.solar_energy))

    def test_initial_models_empty(self):
        self.assertFalse(self.cmip6.cmip6_models)

    def test_initial_config_not_null(self):
        self.assertIsNotNone(self.cmip6.config)

    def test_initial_start_and_end_dates(self):
        self.assertIsNotNone(self.cmip6.config.start_date)
        self.assertIsNotNone(self.cmip6.config.end_date)

    def test_inital_start_and_end_dates_correct_format(self):
        start_date = datetime.strptime(self.cmip6.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.cmip6.config.end_date, '%Y-%m-%d')
        self.assertIsInstance(start_date, datetime,
                              "Make sure that start and end date in `CMIP6_config` is of correct format `YYYY-mm-dd`")
        self.assertIsInstance(end_date, datetime,
                              "Make sure that start and end date in `CMIP6_config` is of correct format `YYYY-mm-dd`")

    def test_initial_variable_and_table_ids_equal_length(self):
        self.assertTrue(len(self.cmip6.config.table_ids) == len(self.cmip6.config.variable_ids),
                        "Make sure that you initialize CMIP6_config using equal length of variable ids and table ids")


if __name__ == "__main__":
    unittest.main()
