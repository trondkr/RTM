from datetime import datetime
import xarray as xr
import logging
import CMIP6_model
import iris
import warnings
import os
import git
import cf_units
from pathlib import Path


class CMIP6_downscale_iris():

    # Add the correct units to coordinates in cube which fails when
    # converting xarray to iris cube
    def fix_coordinates_cube(self, cube):

        for coord in cube.coords():
            if coord.name() == "lat":
                lat = cube.coord("lat")
                cube.remove_coord("lat")
            if coord.name() == "latitude":
                lat = cube.coord("latitude")
                cube.remove_coord("latitude")
            if coord.name() == "lon":
                lon = cube.coord("lon")
                cube.remove_coord("lon")
            if coord.name() == "longitude":
                lon = cube.coord("longitude")
                cube.remove_coord("longitude")

        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat.units = "degrees"
        lon.units = "degrees"

        # Depth is ndim=2 and thetao has ndim=3
        cube.add_dim_coord(lat, cube.ndim - 2)
        cube.add_dim_coord(lon, cube.ndim - 1)

        if not cube.coord("latitude").has_bounds():
            cube.coord("latitude").guess_bounds()
            cube.coord("longitude").guess_bounds()

        return cube

    def fix_calendar(self, cube):

        tcoord = cube.coord('time')
        cube.coord('time').units = cf_units.Unit(tcoord.units.origin, calendar='proleptic_gregorian')

        return cube

    def get_git_attributes(self):
        # pip install gitpython
        repo = git.Repo(search_parent_directories=True)

        assert not repo.bare
        sha = repo.head.object.hexsha
        repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]
        return sha, repo_name

    # Add some attributes to the iris cube
    def add_attributes_to_cube(self, cube,
                               cmip6_model: str,
                               project_name: str,
                               prefix=''):


        a = cube.attributes
        a[prefix + 'version'] = 'ISIMIP3BASD v2.4.1'
        a[prefix + 'model'] = str(cmip6_model)
        a['model_info'] = "None"

        a['author'] = "Test"
        a['institution'] = "Test"
        a['git' + '_sha'] = "None"
        a['git_url'] = "https://github.com/trondkr/CMIP6-downscale"
        a['project'] = str(project_name)
        a['date'] = str(datetime.now())

        cube.attributes = a
        return cube

    # Fix the units of the cube variable
    def add_variable_units(self, cube, units):

        cube.units = units
        return cube

    def ds_to_iris(self, ds: xr.Dataset,
                   var_name: str,
                   model_obj: CMIP6_model,
                   project_name: str,
                   prefix: str = "ba_"):

        ds_iris = ds[var_name].to_iris()
        ds_iris = self.fix_coordinates_cube(ds_iris)

        ds_iris = self.add_attributes_to_cube(ds_iris, model_obj.name, project_name, prefix=prefix)
        if var_name == "thetao":
            ds_iris = self.add_variable_units(ds_iris, 'celsius')
        if var_name == "depth":
            ds_iris = self.add_variable_units(ds_iris, 'meter')
        if var_name == "o2":
            ds_iris = self.add_variable_units(ds_iris, 'ml/l')
        if var_name == "areacello":
            ds_iris = self.add_variable_units(ds_iris, 'kilometers^2')
        if var_name == "siconc":
            ds_iris = self.add_variable_units(ds_iris, '1')
        return ds_iris

    def create_directory_for_path(self, name: str):
        # Make sure directory exists - if not create
        # logging.info("[CMIP6_downscale_iris] Creating directory for output {}".format(name))
        if name.endswith(".nc") or name.endswith(".npy"):
            name = Path(name).parent
        Path(name).mkdir(parents=True, exist_ok=True)

    def save_iris_to_netcdf(self, cube, name, chunksizes):
        logging.info("[CMIP6_downscale_iris] save iris to netcdf : {}".format(name))

        if os.path.exists(name):
            os.remove(name)

        self.create_directory_for_path(name)

        with warnings.catch_warnings():
            #  warnings.simplefilter('ignore', UserWarning)
            iris.save(cube, name,
                      saver=iris.fileformats.netcdf.save,
                      unlimited_dimensions=['time'],
                      fill_value=1.e20, zlib=True,
                      complevel=1, chunksizes=chunksizes)