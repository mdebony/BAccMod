# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: spatial_fit_acceptance_map_creator.py
# Purpose: Concrete class implementing “fit” for a spatial 3D grid acceptance model
# ---------------------------------------------------------------------

import logging
import numpy as np

import astropy.units as u
import gammapy
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis

from .base_fit_acceptance_map_creator import BaseFitAcceptanceMapCreator

logger = logging.getLogger(__name__)


class SpatialFitAcceptanceMapCreator(BaseFitAcceptanceMapCreator):
    """
    Implements the “fit” method for a 3D grid acceptance model.
    Relies on BaseFitAcceptanceMapCreator.fit_background(...) for each energy slice.
    """

    def __init__(
        self,
        energy_axis: MapAxis,
        offset_axis: MapAxis,
        oversample_map: int = 10,
        exclude_regions=None,
        cos_zenith_binning_method: str = 'min_livetime',
        cos_zenith_binning_parameter_value: int = 3600,
        initial_cos_zenith_binning: float = 0.01,
        max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
        zenith_binning_run_splitting: bool = False,
        max_fraction_pixel_rotation_fov: float = 0.5,
        time_resolution: u.Quantity = 0.1 * u.s,
        use_mini_irf_computation: bool = False,
        mini_irf_time_resolution: u.Quantity = 1. * u.min,
        interpolation_type: str = 'linear',
        activate_interpolation_cleaning: bool = False,
        interpolation_cleaning_energy_relative_threshold: float = 1e-4,
        interpolation_cleaning_spatial_relative_threshold: float = 1e-2,
        fit_fnc='gaussian2d',
        fit_seeds=None,
        fit_bounds=None,
    ) -> None:
        """
        Spatial “fit” acceptance creator: splits each energy slice, calls BaseFitAcceptanceMapCreator.fit_background(),
        then normalizes into [counts / (MeV‐sr s)] and builds a Background3D.

        Parameters
        ----------
        energy_axis, offset_axis, oversample_map, etc.  (same as BaseFit, plus fit_fnc, fit_seeds, fit_bounds)
        """
        super().__init__(
            energy_axis=energy_axis,
            offset_axis=offset_axis,
            oversample_map=oversample_map,
            exclude_regions=exclude_regions,
            cos_zenith_binning_method=cos_zenith_binning_method,
            cos_zenith_binning_parameter_value=cos_zenith_binning_parameter_value,
            initial_cos_zenith_binning=initial_cos_zenith_binning,
            max_angular_separation_wobble=max_angular_separation_wobble,
            zenith_binning_run_splitting=zenith_binning_run_splitting,
            max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
            time_resolution=time_resolution,
            use_mini_irf_computation=use_mini_irf_computation,
            mini_irf_time_resolution=mini_irf_time_resolution,
            interpolation_type=interpolation_type,
            activate_interpolation_cleaning=activate_interpolation_cleaning,
            interpolation_cleaning_energy_relative_threshold=interpolation_cleaning_energy_relative_threshold,
            interpolation_cleaning_spatial_relative_threshold=interpolation_cleaning_spatial_relative_threshold,
            fit_fnc=fit_fnc,
            fit_seeds=fit_seeds,
            fit_bounds=fit_bounds,
        )

    def create_acceptance_map(self, observations) -> Background3D:
        """
        Calculate a 3D grid acceptance map by fitting a 2D function in each energy slice.

        Steps:
        1) call self._create_base_computation_map(...) to get (counts, exp_map, exp_map_total, livetime)
        2) downsample exposures to the “model resolution”
        3) for each energy bin: data_e = self.fit_background(counts[e], exp_total_ds[e], exp_ds[e])
        4) normalize to [counts / (sr MeV s)]
        5) pack into a Background3D
        """
        # 1) gather fine-binned counts + exposures + livetime
        count_background, exp_map_background, exp_map_background_total, livetime = self._create_base_computation_map(
            observations
        )

        # 2) downsample exposures
        exp_ds = exp_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_total_ds = exp_map_background_total.downsample(self.oversample_map, preserve_counts=True)

        # 3) for each energy slice, fit 2D function → “corrected counts” on [x,y] grid
        n_energy = count_background.shape[0]
        corrected_counts = np.empty(count_background.shape)
        self.sq_rel_residuals = {"mean": [], "std": []}

        for e in range(n_energy):
            logger.info(
                f"Fitting background, energy bin [{self.energy_axis.edges[e]:.2f}, "
                f"{self.energy_axis.edges[e + 1]:.2f}]"
            )
            corrected_counts[e] = self.fit_background(
                count_map=count_background[e].astype(int),
                exp_map_total=exp_total_ds.data[e],
                exp_map=exp_ds.data[e],
            )

        logger.info(
            "Average event sqrt‐residuals per energy: "
            f"{np.round(self.sq_rel_residuals['mean'], 2)}, std = {np.round(self.sq_rel_residuals['std'], 2)}"
        )

        # 4) build final offset axes (matching Grid3DAcceptanceMapCreator)
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        ext_off_x = MapAxis.from_edges(extended_edges, name="fov_lon")
        ext_off_y = MapAxis.from_edges(extended_edges, name="fov_lat")

        bin_width_x = np.repeat(ext_off_x.bin_width[:, np.newaxis], ext_off_x.nbin, axis=1)
        bin_width_y = np.repeat(ext_off_y.bin_width[np.newaxis, :], ext_off_y.nbin, axis=0)
        solid_angle = 4.0 * (np.sin(bin_width_x / 2) * np.sin(bin_width_y / 2)) * u.steradian

        # 5) normalize: data_background[e, :, :] = corrected_counts[e] / (solid_angle * energy_bin_width * livetime)
        data_background = (
            corrected_counts
            / solid_angle[np.newaxis, :, :]
            / self.energy_axis.bin_width[:, np.newaxis, np.newaxis]
            / livetime
        )

        # 6) instantiate Background3D
        gammapy_major = int(gammapy.__version__.split(".")[0])
        gammapy_minor = int(gammapy.__version__.split(".")[1])
        if gammapy_major == 1 and gammapy_minor >= 3:
            data_final = np.flip(data_background.to(u.Unit("s-1 MeV-1 sr-1")), axis=1)
            acceptance_map = Background3D(
                axes=[self.energy_axis, ext_off_x, ext_off_y],
                data=data_final,
                fov_alignment=FoVAlignment.ALTAZ,
            )
        else:
            acceptance_map = Background3D(
                axes=[self.energy_axis, ext_off_x, ext_off_y],
                data=data_background.to(u.Unit("s-1 MeV-1 sr-1")),
                fov_alignment=FoVAlignment.ALTAZ,
            )

        return acceptance_map
