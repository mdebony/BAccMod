import os

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from gammapy.data import DataStore
from gammapy.irf import Background3D, Background2D
from gammapy.maps import MapAxis
from regions import CircleSkyRegion

from baccmod import RadialAcceptanceMapCreator, Grid3DAcceptanceMapCreator

import gammapy
gammapy_version = gammapy.__version__
gammapy_ver_major = int(gammapy_version.split('.')[0])
gammapy_ver_minor = int(gammapy_version.split('.')[1])
gammapy_ver_patch = 0
if len(gammapy_version.split('.')) > 2:
    gammapy_ver_patch = int(gammapy_version.split('.')[2])



class TestIntegrationClass:
    datastore = DataStore.from_dir(os.path.join(os.environ['GAMMAPY_DATA'], 'hess-dl3-dr1'))

    coordinate_pks_2155 = SkyCoord.from_name('PKS 2155-304')
    exclude_region_PKS_2155 = [CircleSkyRegion(coordinate_pks_2155, 0.4 * u.deg)]
    separation_obs_pks_2155 = SkyCoord(ra=datastore.obs_table['RA_PNT'], dec=datastore.obs_table['DEC_PNT']).separation(
        coordinate_pks_2155)
    id_obs_pks_2155 = datastore.obs_table['OBS_ID'][separation_obs_pks_2155 < 2. * u.deg]
    obs_collection_pks_2155 = datastore.get_observations(obs_id=id_obs_pks_2155, required_irf='all-optional')

    # Inject HESS site information in the run
    for i in obs_collection_pks_2155:
        if gammapy_ver_minor < 3:
            obs_collection_pks_2155[i].obs_info['GEOLON'] = 16.50004902672975
            obs_collection_pks_2155[i].obs_info['GEOLAT'] = -23.271584051253615
            obs_collection_pks_2155[i].obs_info['GEOALT'] = 1800
        obs_collection_pks_2155[i]._location = EarthLocation.from_geodetic(lon=16.50004902672975 * u.deg,
                                                                           lat=-23.271584051253615 * u.deg,
                                                                           height=1800. * u.m)

    energy_axis = MapAxis.from_energy_bounds(100. * u.GeV, 10. * u.TeV, nbin=5, per_decade=True, name='energy')
    energy_axis_computation = MapAxis.from_energy_edges((list(np.geomspace(0.1, 1, 6)) + list(np.geomspace(1, 10, 3)[1:])) * u.TeV, name='energy')
    offset_axis = MapAxis.from_bounds(0. * u.deg, 2. * u.deg, nbin=6, name='offset')

    absolute_tolerance = 1e-15
    #TODO when issue with spatial fit is resolved, need to be lowered
    relative_tolerance = 5e-2

    def test_integration_3D(self):
        bkg_maker = Grid3DAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background3D
        reference = Background3D.read('ressource/test_data/reference_model/pks_2155_3D.fits')
        assert np.all(np.isclose(background_model.data, reference.data,
                                 atol=self.absolute_tolerance,
                                 rtol=self.relative_tolerance))

    def test_integration_spatial_fit(self):
        bkg_maker = Grid3DAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155,
                                               method='fit')
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background3D
        reference = Background3D.read('ressource/test_data/reference_model/pks_2155_spatial_fit_bkg.fits')
        for i in range(background_model.data.shape[0]):
            print(np.sum((np.abs(background_model.data[i, : ,:] - reference.data[i, : ,:]) / reference.data[i, : ,:]) > 1e-3))
        assert np.all(np.isclose(background_model.data, reference.data,
                                 atol=self.absolute_tolerance,
                                 rtol=self.relative_tolerance))

    def test_integration_spatial_fit(self):
        bkg_maker = Grid3DAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155,
                                               method='fit')
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background3D

    def test_integration_2D(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background2D
        reference = Background2D.read('ressource/test_data/reference_model/pks_2155_2D.fits')
        assert np.all(np.isclose(background_model.data, reference.data,
                                 atol=self.absolute_tolerance,
                                 rtol=self.relative_tolerance))

    def test_integration_3D_unregular_computation_axis(self):
        bkg_maker = Grid3DAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               energy_axis_computation=self.energy_axis_computation,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background3D

    def test_integration_spatial_fit_unregular_computation_axis(self):
        bkg_maker = Grid3DAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               energy_axis_computation=self.energy_axis_computation,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155,
                                               method='fit')
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background3D

    def test_integration_2D_unregular_computation_axis(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               energy_axis_computation=self.energy_axis_computation,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background2D

    def test_integration_zenith_binned_model(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map_cos_zenith_binned(observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D
            reference = Background2D.read(f'ressource/test_data/reference_model/pks_2155_{id_obs}_zenith_binned.fits')
            assert np.all(np.isclose(background_model[id_obs].data, reference.data,
                                     atol=self.absolute_tolerance,
                                     rtol=self.relative_tolerance))

    def test_integration_zenith_interpolated_model(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map_cos_zenith_interpolated(
            observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D
            reference = Background2D.read(f'ressource/test_data/reference_model/pks_2155_{id_obs}_zenith_interpolated.fits')
            assert np.all(np.isclose(background_model[id_obs].data, reference.data,
                                     atol=self.absolute_tolerance,
                                     rtol=self.relative_tolerance))

    def test_integration_zenith_interpolated_model_mini_irf_and_run_splitting(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155,
                                               use_mini_irf_computation=True,
                                               zenith_binning_run_splitting=True)
        background_model = bkg_maker.create_acceptance_map_cos_zenith_interpolated(
            observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D
            reference = Background2D.read(f'ressource/test_data/reference_model/pks_2155_{id_obs}_zenith_interpolated_run_splitting_mini_irf.fits')
            assert np.all(np.isclose(background_model[id_obs].data, reference.data,
                                     atol=self.absolute_tolerance,
                                     rtol=self.relative_tolerance))

    def test_integration_norm_per_run(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map_per_observation(observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D
            reference = Background2D.read(f'ressource/test_data/reference_model/pks_2155_{id_obs}_run_normalisation.fits')
            assert np.all(np.isclose(background_model[id_obs].data, reference.data,
                                     atol=self.absolute_tolerance,
                                     rtol=self.relative_tolerance))
