import unittest
import numpy as np

from astropy.modeling.core import ModelDefinitionError

from baccmod.model import (CustomModels,
                           Gaussian2DLinearGradientsSpatial,
                           PowerLawCutOffsEnergy,
                           PLCutOffsEnergyxGaussGradSpatial)


class TestModelClass(unittest.TestCase):

    def test_powerlaw_cutoffs(self):
        model = PowerLawCutOffsEnergy()
        model.evaluate(1, amplitude=100.0, index=2.2, eref=1.0, b=2.1, ecl=0.1, ech=100.0)
        model.evaluate(np.array([1, 2, 3, 4, 5]), amplitude=100.0, index=2.2, eref=1.0, b=2.1, ecl=0.1, ech=100.0)

    def test_2dmodel_gaussgrad(self):
        model = Gaussian2DLinearGradientsSpatial()
        model.evaluate(x=0, y=2)

    def test_3dmodel_plcuts_gaussgrad(self):
        model = PLCutOffsEnergyxGaussGradSpatial()
        model.evaluate(e=5, x=0, y=2)

    def test_CustomModels_specialcases(self):
        with self.assertRaises(NotImplementedError):
            CustomModels.evaluate(None)
        with self.assertRaises(ModelDefinitionError):
            CustomModels(5)
