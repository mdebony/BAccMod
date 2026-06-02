import unittest
import numpy as np

from astropy.modeling.core import ModelDefinitionError

from baccmod.model import (CustomModels,
                           Gaussian2DLinearGradientsSpatial,
                           PowerLawCutOffsEnergy,
                           PLCutOffsEnergyxGaussGradSpatial)



class TestModelClass(unittest.TestCase):

    numerical_r_tolerance = 1e-5
    numerical_a_tolerance = 1e-8

    def test_powerlaw_cutoffs(self):
        model = PowerLawCutOffsEnergy()
        e1 = model.evaluate(1, amplitude=100.0, index=2.2, eref=1.0, b=2.1, ecl=0.1, ech=100.0)
        e2 = model.evaluate(np.array([1, 2, 3, 4, 5]), amplitude=100.0, index=2.2, eref=1.0, b=2.1, ecl=0.1, ech=100.0)
        assert np.isclose(e1, 98.22167398064686,
                          rtol=self.numerical_r_tolerance,
                          atol=self.numerical_a_tolerance)
        assert np.all(np.isclose(e2, [98.22167398, 21.29332307,  8.64890211,  4.54892255,  2.75698111],
                                 rtol=self.numerical_r_tolerance,
                                 atol=self.numerical_a_tolerance))

    def test_2dmodel_gaussgrad(self):
        model = Gaussian2DLinearGradientsSpatial()
        e = model.evaluate(x=0, y=2)
        assert np.isclose(e, 0.1353352832366127,
                          rtol=self.numerical_r_tolerance,
                          atol=self.numerical_a_tolerance)

    def test_3dmodel_plcuts_gaussgrad(self):
        model = PLCutOffsEnergyxGaussGradSpatial()
        e = model.evaluate(e=5, x=0, y=2)
        assert np.isclose(e, 0.005147336796951535,
                          rtol=self.numerical_r_tolerance,
                          atol=self.numerical_a_tolerance)

    def test_CustomModels_specialcases(self):
        with self.assertRaises(NotImplementedError):
            CustomModels.evaluate(None)
        with self.assertRaises(ModelDefinitionError):
            CustomModels(5)
