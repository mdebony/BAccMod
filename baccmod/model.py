# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: modeling.py
# Purpose: Collection of function for fitting analytic model of background
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


"""
Define the background model functions for fits and associated seeds and bounds.
"""
import numpy as np

__all__ = ['BilinearGradient']

from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.functional_models import Gaussian2D


class BilinearGradient(Fittable2DModel):
    x_gradient = Parameter()
    y_gradient = Parameter()

    @staticmethod
    def evaluate(x, y, x_gradient, y_gradient):
        return (1 + x * x_gradient) * (1 + y * y_gradient)

    @staticmethod
    def fit_deriv(x, y, x_gradient, y_gradient):
        """
        Partial derivatives as function of the parameters of the model
        """
        d_x_gradient = x * (1 + y * y_gradient)
        d_y_gradient = y * (1 + x * x_gradient)
        return [d_x_gradient, d_y_gradient]
