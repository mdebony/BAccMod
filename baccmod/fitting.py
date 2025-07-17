import numpy as np
from astropy.modeling.fitting import Fitter
from iminuit import Minuit

class PoissonFitter(Fitter):
    supported_constraints = ['fixed', 'bounds']
    supports_uncertainties = False

    def __init__(self):
        super().__init__()

    def __call__(self, model, *coords, data, maxiter=None):
        """
        model : any astropy.modeling.Model whose call signature is
                model(*coords) → intrinsic rate array
        coords: N arrays of shape = data.shape giving the N coordinate grids
        data : integer counts array
        """
        model_copy = model.copy()

        # 1) flatten each coordinate array
        coord1 = [c.ravel() for c in coords]
        data1 = data.ravel().astype(int)

        # 2) precompute log-factorials
        log_fact = self._log_factorial(data1)

        # 3) gather parameter names, initial guesses, bounds, fixed flags
        pnames = list(model_copy.param_names)
        seeds  = {p: getattr(model_copy, p).value for p in pnames}
        bounds = {p: model_copy.bounds.get(p, (None, None)) for p in pnames}
        fixed  = {p: model_copy.fixed.get(p, False)         for p in pnames}

        # 4) define negative log‐likelihood
        def neg_logL(**pars):
            # update model parameters in‐place
            for name, val in pars.items():
                setattr(model_copy, name, val)
            # expected counts = intrinsic_rate * (e_masked/e_tot)
            mu = model_copy(*coord1)
            mu = np.clip(mu, 1e-12, None)
            return -np.sum(self._log_poisson(mu, data1, log_fact))

        # 5) set up Minuit
        m = Minuit(neg_logL, name=pnames, **seeds)
        m.errordef = Minuit.LIKELIHOOD

        # 6) apply bounds & fixed flags
        for p in pnames:
            lo, hi = bounds[p]
            if fixed[p]:
                m.fixed[p] = True
            elif lo is not None or hi is not None:
                m.limits[p] = (lo, hi)

        # 7) minimize
        m.simplex(ncall=maxiter).migrad(ncall=maxiter)

        # 8) write best‐fit back into the model
        for name, val in m.values.to_dict().items():
            setattr(model_copy, name, val)

        return model_copy

    @staticmethod
    def _log_factorial(x):
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
    def _log_poisson(mu, x, log_factorial_x):
        return -mu + x * np.log(mu) - log_factorial_x
