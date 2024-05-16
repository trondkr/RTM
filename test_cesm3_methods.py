import pytest
import numpy as np
import xarray as xr
from CMIP6_cesm3 import CMIP6_cesm3

class TestCMIP6_cesm3:
    def setup_class(self):
        self.cmip6 = CMIP6_cesm3()

    def test_calculate_bare_ice_albedo(self):
        osa = xr.DataArray(np.random.rand(5, 5))
        fh = xr.DataArray(np.random.rand(5, 5))
        ice_thickness = xr.DataArray(np.random.rand(5, 5))
        snow_concentration = xr.DataArray(np.zeros((5, 5)))

        result = self.cmip6.calculate_bare_ice_albedo(osa, fh, 0.78, ice_thickness, snow_concentration, 0.02)

        assert 0 <= result.min() <= result.max() <= 1  # Resulting albedo values should be between [0, 1]

    def test_calculate_bare_ice_albedo_with_zero_ice_thickness(self):
        osa = xr.DataArray(np.random.rand(5, 5))
        fh = xr.DataArray(np.zeros((5, 5)))
        ice_thickness = xr.DataArray(np.zeros((5, 5)))
        snow_concentration = xr.DataArray(np.zeros((5, 5)))

        result = self.cmip6.calculate_bare_ice_albedo(osa, fh, 0.78, ice_thickness, snow_concentration, 0.02)

        assert (result == osa).all()  # When ice thickness is 0, resultant albedo should be equal to osa
