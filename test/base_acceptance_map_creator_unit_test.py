import unittest

import astropy.units as u
import numpy as np
from gammapy.maps import MapAxis
from baccmod.base_acceptance_map_creator import BaseAcceptanceMapCreator

class DummyBaseAcceptanceMapCreator(BaseAcceptanceMapCreator):
    """ Dummy class to test base functionnalities of BaseAcceptanceMapCreator. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_base_computation_map(self):
        return None

    def create_model(self):
        return None

class TestBaseAcceptanceClass(unittest.TestCase):

    numerical_r_tolerance = 1e-5
    numerical_a_tolerance = 1e-8


    def test_compute_dynamic_energy_axis(self):
        """ Test `BaseAcceptanceMapCreator._compute_dynamic_energy_axis` """
        errors_msg=[]
        errors=[]

        out_energy_axis = MapAxis.from_energy_bounds(100.*u.GeV, 10.*u.TeV, nbin=15, per_decade=False, name='energy')

        bamc = DummyBaseAcceptanceMapCreator(energy_axis_out=out_energy_axis,
                                             max_offset=1*u.deg,
                                             spatial_resolution=0.5*u.deg,
                                             use_dynamic_energy_axis=True,)

        # test 1 : basic run with known output
        data=[0,0,2,20,15,4,6,3,2,1,1,0,0,0,0]
        nb_spatial_bin = 4
        bamc.dynamic_energy_axis_target_statistics = 3
        bamc.dynamic_energy_axis_maximum_wideness_bin = 2 # remove the maximum bin wideness criteria
        # expected output : more than 12 counts per bin which would be [0,0,22,19,13,0,0,0,0] counts per energy bins
        # Number of merged bins : [0,0,2,2,4,0,0,0,0]
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)

        try:
            assert len(bamc.energy_axis_computation.edges) == 10
            assert np.all(bamc.energy_axis_computation.edges == out_energy_axis.edges[[0,1,2,4,6,11,12,13,14,15]])
        except Exception as e:
            errors_msg.append('Error in the simple usage without maximum bin wideness.\n')
            errors.append(e)

        bamc.dynamic_energy_axis_maximum_wideness_bin = 0.5 # default maximum bin wideness criteria: limiting once
        # we have 15 bin for 2 decades so this corresponds to a limit of 3.75 bins merged at most
        # expected output : [0,0,22,15,13,4,0,0,0,0] counts per energy bins
        # Number of merged bins : [0,0,2,1,3,3,0,0,0,0]
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)


        try:
            assert len(bamc.energy_axis_computation.edges) == 11
            assert np.all(bamc.energy_axis_computation.edges == out_energy_axis.edges[[0,1,2,4,5,8,11,12,13,14,15]])
        except Exception as e:
            errors_msg.append('Error in the simple usage with maximum bin wideness.\n')
            errors.append(e)

        if len(errors)>0:
            raise ExceptionGroup('Errors in the test of BaseAcceptanceMapCreator._compute_dynamic_energy_axis',errors)
