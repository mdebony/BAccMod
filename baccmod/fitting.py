# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: fitting.py
# Purpose: Implement an astropy fitter that used poisson statistics likelihood and iminuit
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------
import numpy as np
import logging
from typing import List
from astropy.modeling import Model
from astropy.modeling.fitting import Fitter
from iminuit import Minuit

logger = logging.getLogger(__name__)

class PoissonFitter(Fitter):
    supported_constraints = ['fixed', 'bounds', 'tied']
    supports_uncertainties = False

    upsampling_integration = 10

    def __init__(self):
        super().__init__(optimizer=self._optimizer, statistic=self._statistic)

    @staticmethod
    def _optimizer():
        raise NotImplementedError()

    @staticmethod
    def _statistic(self):
        raise NotImplementedError()

    def __call__(self,
                 model: Model,
                 *coords: List[np.ndarray],
                 data: np.ndarray,
                 exposure_correction: np.ndarray = None,
                 integrated_data: bool = False,
                 log_scaling: bool = False,
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
            integrated_data: bool
                If true the data are considered as integrated and coords are the edges of the bins
            log_scaling: bool
                A logarithmic scaled binning is used for integration
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
        flat_data = data.ravel().astype(int)
        flat_exposure_correction = exposure_correction.ravel()

        # Determination of the evaluation points and weights for the integrated case
        if integrated_data:
            evaluation_points = []
            weights = []
            for i, c in enumerate(coords):
                # Add a dimension after each individual one for integration, the dimension of the current dimension is extending to store values for the integration
                dim_expansion = (slice(None), None) * (c.ndim)
                new_c = np.repeat(c[dim_expansion], repeats=self.upsampling_integration+1, axis=2*i+1)

                # Create slice for width determination
                us, ls = [slice(None)] * new_c.ndim, [slice(None)] * new_c.ndim
                us[2*i] = slice(1, new_c.shape[2*i])
                ls[2*i] = slice(0, new_c.shape[2*i] - 1)
                for j in range(len(coords)):
                    if i != j:
                        # Take into account that the other axis will be reduced by one also due to transformation edges -> bin
                        us[2*j] = slice(0, new_c.shape[2*j] - 1)
                        ls[2*j] = slice(0, new_c.shape[2*j] - 1)
                us, ls = tuple(us),  tuple(ls)

                # Slice dim expansion of the evaluation points
                sd = [None] * new_c.ndim
                sd[2*i+1] = slice(None)
                sd = tuple(sd)

                if log_scaling:
                    width = np.log10(new_c[us]) - np.log10(new_c[ls])
                    pts = 10**(np.log10(new_c[ls]) + width * np.linspace(0, 1, self.upsampling_integration + 1)[sd])
                else:
                    width = new_c[us] - new_c[ls]
                    pts = new_c[ls] + width * np.linspace(0, 1, self.upsampling_integration+1)[sd]
                evaluation_points.append(pts)

                # Create slice for integration width determination
                usi, lsi = [slice(None)] * pts.ndim, [slice(None)] * pts.ndim
                usi[2 * i + 1] = slice(1, pts.shape[2 * i + 1])
                lsi[2 * i + 1] = slice(0, pts.shape[2 * i + 1] - 1)
                usi, lsi = tuple(usi),  tuple(lsi)
                width_integration = evaluation_points[usi] - evaluation_points[lsi]

                # Compute weights associated with each points for the integration
                w = np.ones_like(pts)
                w[lsi] += 0.5 * width_integration
                w[usi] += 0.5 * width_integration
                weights.append(w)
        else:
            evaluation_points = [c.ravel() for c in coords]

        # precompute log‑factorial
        log_fact = self._log_factorial(flat_data)

        # gather constraint info
        tied = model_copy.tied or {}

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
            if integrated_data:
                val = model_copy(*evaluation_points)
                for i in range(len(coords)-1, -1, -1):
                    val = np.sum(val*weights[i], axis=2*i+1)
                mu = val.ravel() * flat_exposure_correction
            else:
                mu = model_copy(*evaluation_points) * flat_exposure_correction
            return -np.sum(self._log_poisson(mu, flat_data, log_fact))

        # wrapper to accept both positional and keyword args
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
            lo, hi = bounds[p]
            if fixed[p]:
                m.fixed[p] = True
            elif lo is not None or hi is not None:
                m.limits[p] = (lo, hi)

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
