# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: fitting.py
# Purpose: Implement an astropy fitter that used poisson statistics likelihood and iminuit
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------
import logging
from typing import List

import numpy as np
from astropy.modeling import Model
from iminuit import Minuit

logger = logging.getLogger(__name__)

class PoissonFitter():
    supported_constraints = ['fixed', 'bounds', 'tied']
    supports_uncertainties = False

    def __init__(self):
        pass

    def __call__(self,
                 model: Model,
                 *coords: List[np.ndarray],
                 data: np.ndarray,
                 exposure_correction: np.ndarray = None,
                 maxiter: int = 1000) -> Model:
        """
        Fit the model to the data following poisson likelihood statistics and using iminuit

        Parameters
        ----------
            model : astropy.modeling.Model
                model to use for the fitting
            coords: list of np.array
                N arrays of shape = data.shape giving the N coordinate grids
            data : np.array
                Integer counts array of N dimension
            exposure_correction: np.array
                Floating point value to correct for difference of exposure in the data
            maxiter : int
                maximum number of iteration for fitting, as the fitting is performed in two steps, could be the double of this value in practice

        Returns
        -------
            model : astropy.modeling.Model
                the fitted model
        """
        # work on a copy
        model_copy = model.copy()

        if exposure_correction is None:
            exposure_correction = np.ones_like(data, dtype=np.float64)

        # flatten coords & data
        flat_coords = [c.ravel() for c in coords]
        flat_data   = data.ravel().astype(int)
        flat_exposure_correction = exposure_correction.ravel()

        # precompute log‑factorial
        log_fact = self._log_factorial(flat_data)

        # gather constraint info
        tied = model_copy.tied

        # build list of free parameters
        all_params = list(model_copy.param_names)
        free_params = [p for p in all_params if (p not in tied) or (not tied[p])]

        # initial seeds, bounds, fixed flags
        seeds  = {p: getattr(model_copy, p).value for p in free_params}
        bounds = {p: model_copy.bounds.get(p, (None, None)) for p in free_params}
        fixed  = {p: model_copy.fixed.get(p, False)         for p in free_params}

        # helper to apply parameters and tied relation
        def apply_params_and_tied(pars):
            # update free params
            for name, val in pars.items():
                setattr(model_copy, name, val)
            # update tied params
            for name, rule in tied.items():
                if callable(rule):
                    val = rule(model_copy)
                elif isinstance(rule, str):
                    ctx = {p: getattr(model_copy, p).value for p in all_params}
                    val = eval(rule, {}, ctx)
                else:
                    continue
                setattr(model_copy, name, val)

        # negative log‑likelihood
        def neg_logL(**pars):
            apply_params_and_tied(pars)
            mu = model_copy(*flat_coords) * flat_exposure_correction
            return -np.sum(self._log_poisson(mu, flat_data, log_fact))

        # wrapper to accept either positional arguments or keyword arguments (exclusives)
        def fcn_wrapper(*args, **kwargs):
            if args:
                pars = dict(zip(free_params, args))
            else:
                pars = kwargs
            return neg_logL(**pars)

        # set up Minuit
        m = Minuit(fcn_wrapper, name=free_params, **seeds)
        m.errordef = Minuit.LIKELIHOOD

        # apply bounds & fixed
        for p in free_params:
            m.limits[p] = bounds[p]
            if fixed[p]:
                m.fixed[p] = True

        # minimize
        m.simplex(ncall=maxiter).migrad(ncall=maxiter)
        if not m.valid:
            logger.warning("PoissonFitter: fit may not have converged.")

        # write best‑fit back to model (free + tied)
        best = m.values.to_dict()
        apply_params_and_tied(best)

        return model_copy

    @staticmethod
    def _log_factorial(x: np.ndarray) -> np.ndarray:
        """
        Returns the log of the factorial of elements of `count_map` while computing each value only once.
        Parameters
        ----------
        x: Array-like of integers
            Input for which we want the factorial of all elements
        Returns
        -------
            The factorial of x in log scale
        """
        max_x = x.max()
        all_int = np.arange(0, max_x + 1)
        all_int[0] = 1
        log_int = np.log(all_int)
        log_int_factorials = np.cumsum(log_int)
        log_factorial_x = log_int_factorials[x]
        return log_factorial_x

    @staticmethod
    def _log_poisson(mu: np.ndarray, x: np.ndarray, log_factorial_x: np.ndarray) -> np.ndarray:
        return -mu + x * np.log(mu) - log_factorial_x
