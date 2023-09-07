import unittest
import CMIP6_IO

class MyCMIP6IOCase(unittest.TestCase):
    def test_formatting_of_netcdf_filename(self):
        cmip6_io = CMIP6_IO.CMIP6_IO()

        expected_formatted_name = "test_dir/ssp245/model_name/CMIP6_model_name_member_id_ssp245_key.nc"
        formatted_name = cmip6_io.format_netcdf_filename(dir="test_dir", model_name="model_name", member_id="member_id", key="key",current_experiment_id="ssp245")
        print(expected_formatted_name)
        self.assertEqual(expected_formatted_name, formatted_name)

if __name__ == '__main__':
    unittest.main()
