# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: modeling.py
# Purpose: Introduce a class for users to create custom models (astropy.modeling.FittableModel) to fit on the background
#          using the baccmod.FitAcceptanceMapCreator.
#          A number of models, creating using this class, are also provided.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------

__all__ = ['CustomModels',
           'bilinear_gradient',
           'gaussian2d_bilinear_gradient',
           'powerlawenergy_gaussian2dspatial',
           'Gaussian2DxLinearGradientsSpatial',
           'PowerLawEnergyxGaussian2DSpatial']
import logging
from astropy.modeling.core import ModelDefinitionError, _custom_model_inputs
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling import FittableModel
from astropy.modeling.parameters import Parameter
from astropy.utils import find_current_module
logger = logging.getLogger(__name__)


class CustomModels(FittableModel):
    """
    Creates an `astropy.modeling.FittableModel` compatible with baccmod (and astropy) fitting procedures.
    The FittableModel is created from any provided function and automatically detect the number of inputs.
    The function REQUIRES inputs to be without default values and model parameters to have a default value as this is
    used to identify the difference between the two.

    BAccMod assumes that the inputs are ordered as (energy, x, y) with x and y the offset in camera coordinates.
    One input is assumed to be energy, two inputs camera coordinates, and three inputs energy and camera coordinates.

    CustomModels behaves similarly to `astropy.modeling.core.custom_model`. It can be used in two way:

        - To create a new class to instanciate later.
            ```
            class MyCustomModel(CustomModels):
                def __init__(self):
                    super().__init__(fnc = my_function)
            ...
            my_fittable_model = MyCustomModel()

           ```
        - To directly initialise a model with the required function.
           ```
           my_fittable_model = CustomModels(fnc = my_function)
           ```

    Parameters
    ----------
    fnc: callable
        Function to be fitted. Defined with inputs without values and parameters with default value.
    """
    def __init__(self, fnc):
        super().__init__()
        if not callable(fnc):
            raise ModelDefinitionError(
                "fnc is not callable; it must be a function or other callable object"
            )
        inputs, special_params, settable_params, params = _custom_model_inputs(fnc)
        params = {
            param: Parameter(param, default=default) for param, default in params.items()
        }

        self.param_names = list(params.keys())
        mod = find_current_module(2)
        if mod:
            modname = mod.__name__
        else:
            modname = "__main__"
        members = {
            "_name":fnc.__name__,
            "__module__": str(modname),
            "__doc__": fnc.__doc__,
            "_inputs": inputs,
            "n_outputs": special_params.pop("n_outputs", 1),
            "evaluate": fnc,
            "_settable_properties": settable_params,
        }

        members.update(params)
        self.__dict__.update(members)
        self._initialize_parameters(args=[], kwargs=params)
        self._initialize_slices()
        self._initialize_unit_support()
        self._separable = len(inputs) == 1
        logger.info(f"Initialised custom model with function {fnc.__name__} and {self.n_inputs:d} inputs")

    def evaluate(self, *args, **kwargs):
        pass


# Below are some models function and classes created using CustomModels.

def bilinear_gradient(x, y, x_gradient=0, y_gradient=0):
    """ Function used to create a linear gradient defined along 2 coordinates. """
    return (1 + x * x_gradient) * (1 + y * y_gradient)

def gaussian2d_bilinear_gradient(x, y, amplitude=1.0, x_mean=0.0, y_mean=0.0, x_stddev=1.0, y_stddev=1.0,
                                        theta=0.0, x_gradient=0, y_gradient=0):
    """ Function convoluting a two dimensionnal Gaussian with a linear gradient """
    return Gaussian2D.evaluate(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta) * bilinear_gradient(
        x, y, x_gradient, y_gradient)


def powerlawenergy_gaussian2dspatial(e, x, y, amplitude=1.0, x_0=1, alpha=2.0, x_mean=0.0, y_mean=0.0,
                                     x_stddev=1.0, y_stddev=1.0, theta=0.0):
    """ Function convoluting a powerlaw model along one dimension (assumed energy) with a two dimensional Gaussian along
        two different dimensions (assumed spatial) """
    return amplitude * PowerLaw1D.evaluate(e, 1, x_0, alpha) * Gaussian2D.evaluate(x, y, 1, x_mean, y_mean, x_stddev,
                                                                                   y_stddev, theta)

class Gaussian2DxLinearGradientsSpatial(CustomModels):
    """ 2D FIttableModel using a two dimensional spatial distribution convoluting a two dimensionnal Gaussian with a
        linear gradient"""
    def __init__(self):
        super().__init__(gaussian2d_bilinear_gradient)

class PowerLawEnergyxGaussian2DSpatial(CustomModels):
    """ 3D FittableModel using a powerlaw enegy distribution and two dimensional Gaussian spatial distribution."""
    def __init__(self):
        super().__init__(powerlawenergy_gaussian2dspatial)
