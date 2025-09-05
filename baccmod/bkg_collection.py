# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: bkg_collection.py
# Purpose: Class for storing model with background zenith binning
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from gammapy.irf import FoVAlignment
from gammapy.irf.background import BackgroundIRF, Background2D, Background3D
import astropy.units as u
from scipy.interpolate import interp1d

from .exception import BackgroundModelFormatException
from .toolbox import compute_neighbour_condition_validation

logger = logging.getLogger(__name__)

class BackgroundCollection(ABC):

    def __init__(self,
                 interpolation_type: str='linear',
                 threshold_value_log_interpolation: float=np.finfo(np.float64).tiny,
                 activate_interpolation_cleaning: bool=False,
                 interpolation_cleaning_energy_relative_threshold: bool=1e-4,
                 interpolation_cleaning_spatial_relative_threshold: bool=1e-2):
        """
        interpolation_type: str, optional
            Select the type of interpolation to be used, could be either "log" or "linear", log tend to provided better results be could more easily create artefact that will cause issue
        activate_interpolation_cleaning: bool, optional
            If true, will activate the cleaning step after interpolation, it should help to eliminate artefact caused by interpolation
        interpolation_cleaning_energy_relative_threshold: float, optional
            To be considered value, the bin in energy need at least one adjacent bin with a relative difference within this range
        interpolation_cleaning_spatial_relative_threshold: float, optional
            To be considered value, the bin in space need at least one adjacent bin with a relative difference within this range
        """
        self.interpolation_type = interpolation_type
        self.threshold_value_log_interpolation = threshold_value_log_interpolation
        self.activate_interpolation_cleaning = activate_interpolation_cleaning
        self.interpolation_cleaning_energy_relative_threshold = interpolation_cleaning_energy_relative_threshold
        self.interpolation_cleaning_spatial_relative_threshold = interpolation_cleaning_spatial_relative_threshold
        self.max_cleaning_iteration = 50

        self.interpolation_function_exist = False

    @abstractmethod
    def get_zenith(self, azimuth:u.Quantity):
        """
        Return zenith of the available models
        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want to have the list of available zenith

        Returns
        -------
        keys : np.array
            The zenith angle available in degree
        """
        pass

    @abstractmethod
    def get_model_from_collection(self, zenith:u.Quantity, azimuth:u.Quantity):
        """
        Return zenith of the available models
        Parameters
        ----------
        zenith: u.Quantity
            the zenith of the model, must exist in the return of get_zenith
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        pass

    @abstractmethod
    def generate_interpolation_function(self):
        """
        Generate the interpolation functions
        In the implementation, could use self.interpolation_function_exist to avoid multiple generation
        """
        pass

    @abstractmethod
    def get_interpolation_function(self, azimuth: u.Quantity):
        """
        Retrieve the appropriate interpolation function

        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        interp_func : scipy.interpolate.interp1d
            The object that could be call directly for performing the interpolation
        """
        pass

    def get_binned_model(self, zenith:u.Quantity, azimuth:u.Quantity):
        """
        Return the binned model for a given zenith and east/west pointing
        Parameters
        ----------
        zenith: u.Quantity
            the zenith for which a model is requested
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        cos_zenith_observation = np.cos(zenith)
        zenith_model = self.get_zenith(azimuth)
        cos_zenith_model = np.cos(np.deg2rad(zenith_model))
        key_closest_model = zenith_model[np.abs(cos_zenith_model - cos_zenith_observation).argmin()]
        return self.get_model_from_collection(key_closest_model, azimuth)

    def get_interpolated_model(self, zenith: u.Quantity, azimuth: u.Quantity):
        """
        Return the interpolated model for a given zenith and east/west pointing
        Parameters
        ----------
        zenith: u.Quantity
            the zenith for which a model is requested
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """

        # Determine model properties
        ref_model = self.get_model_from_collection(self.get_zenith(azimuth)[0], azimuth)
        type_model = type(ref_model)
        axes_model = ref_model.axes
        unit_model = ref_model.unit

        # Perform the interpolation
        interp_func = self.get_interpolation_function(azimuth)
        if interp_func is None:
            return self.get_model_from_collection(self.get_zenith(azimuth)[0], azimuth)
        else:
            if self.interpolation_type == 'log':
                interp_bkg = (10. ** interp_func(np.cos(zenith)))
                interp_bkg[interp_bkg < 100 * self.threshold_value_log_interpolation] = 0.
            elif self.interpolation_type == 'linear':
                interp_bkg = interp_func(np.cos(zenith))
                interp_bkg[interp_bkg < 0.] = 0.
            else:
                raise Exception("Unknown interpolation type")
            if self.activate_interpolation_cleaning:
                interp_bkg = self._background_cleaning(interp_bkg)

            # Return the model
            if type_model is Background2D:
                return Background2D(axes=axes_model, data=interp_bkg * unit_model)
            elif type_model is Background3D:
                return Background3D(axes=axes_model, data=interp_bkg * unit_model, fov_alignment=FoVAlignment.ALTAZ)
            else:
                raise Exception('Unknown background format')

    def _background_cleaning(self, background_model):
        """
            Is cleaning the background model from suspicious values not compatible with neighbour pixels.

            Parameters
            ----------
            background_model : numpy.array
                The background model to be cleaned

            Returns
            -------
            background_model : numpy.array
                The background model cleaned
        """

        base_model = background_model.copy()
        final_model = background_model.copy()
        i = 0
        while (i < 1 or not np.allclose(base_model, final_model)) and (i < self.max_cleaning_iteration):
            base_model = final_model.copy()
            i += 1

            count_valid_neighbour_condition_energy = compute_neighbour_condition_validation(base_model, axis=0,
                                                                                            relative_threshold=self.interpolation_cleaning_energy_relative_threshold)
            count_valid_neighbour_condition_spatial = compute_neighbour_condition_validation(base_model, axis=1,
                                                                                             relative_threshold=self.interpolation_cleaning_spatial_relative_threshold)
            if base_model.ndim == 3:
                count_valid_neighbour_condition_spatial += compute_neighbour_condition_validation(base_model, axis=2,
                                                                                                  relative_threshold=self.interpolation_cleaning_spatial_relative_threshold)

            mask_energy = count_valid_neighbour_condition_energy > 0
            mask_spatial = count_valid_neighbour_condition_spatial > (1 if base_model.ndim == 3 else 0)
            mask_valid = np.logical_and(mask_energy, mask_spatial)
            final_model[~mask_valid] = 0.

        return final_model

    def _create_interpolation_function_from_zenith_collection(self, base_model: Dict[float, BackgroundIRF]) -> interp1d:
        """
            Create the function that will perform the interpolation

            Parameters
            ----------
            base_model : dict of gammapy.irf.background.BackgroundIRF
                The binned base model
                Each key of the dictionary should correspond to the zenith in degree of the model

            Returns
            -------
            interp_func : scipy.interpolate.interp1d
                The object that could be call directly for performing the interpolation
        """

        # Reshape the base model
        binned_model = []
        cos_zenith_model = []
        for k in np.sort(list(base_model.keys())):
            binned_model.append(base_model[k])
            cos_zenith_model.append(np.cos(np.deg2rad(k)))
        cos_zenith_model = np.array(cos_zenith_model)

        data_cube = np.zeros(tuple([len(binned_model), ] + list(binned_model[0].data.shape))) * binned_model[0].unit
        for i in range(len(binned_model)):
            data_cube[i] = binned_model[i].data * binned_model[i].unit
        if self.interpolation_type == 'log':
            interp_func = interp1d(x=cos_zenith_model,
                                   y=np.log10(data_cube.to_value(
                                       binned_model[0].unit) + self.threshold_value_log_interpolation),
                                   axis=0,
                                   fill_value='extrapolate')
        elif self.interpolation_type == 'linear':
            interp_func = interp1d(x=cos_zenith_model,
                                   y=data_cube.to_value(binned_model[0].unit),
                                   axis=0,
                                   fill_value='extrapolate')
        else:
            raise Exception("Unknown interpolation type")

        return interp_func

    @staticmethod
    def _check_entry(key, v):
        error_message = ''
        if key > 90.0 or key < 0.0:
            error_message += ('Invalid key : The zenith associated with the model should be between 0 and 90 in degree,'
                              ' ') + str(key) + ' provided.\n'
        if not isinstance(v, BackgroundIRF):
            error_message += 'Invalid type : model should be a BackgroundIRF.'
        if error_message != '':
            raise BackgroundModelFormatException(error_message)


class BackgroundCollectionZenith(BackgroundCollection):

    def __init__(self, bkg_dict: dict[float, BackgroundIRF] = None, **kwargs):
        """
            Create the class for storing a collection of model for different zenith angle

            Parameters
            ----------
            bkg_dict : dict of gammapy.irf.BackgroundIRF
                The collection of model in a dictionary with as key the zenith angle (in degree) associated to the model
            **kwargs:
                Arguments for the base class, see docstring of BackgroundCollection
        """
        super().__init__(**kwargs)
        self.interpolation_function = None
        bkg_dict = bkg_dict or {}
        self.bkg_dict = {}
        for k, v in bkg_dict.items():
            key = float(k)
            self._check_entry(key, v)
            self.bkg_dict[key] = v

    def get_zenith(self, azimuth:u.Quantity=None):
        """
        Return zenith of the available models
        Parameters
        ----------
        azimuth: u.Quantity
            ignored

        Returns
        -------
        keys : np.array
            The zenith angle available in degree
        """
        return np.sort(np.array(list(self.bkg_dict.keys())))*u.deg

    def get_model_from_collection(self, zenith: u.Quantity, azimuth: u.Quantity = None):
        """
        Return zenith of the available models
        Parameters
        ----------
        zenith: u.Quantity
            the zenith of the model, must exist in the return of get_zenith
        azimuth: u.Quantity
            ignored

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        return self.bkg_dict[zenith.to_value(u.deg)]

    def generate_interpolation_function(self):
        """
        Generate the interpolation functions
        """
        if not self.interpolation_function_exist:
            self.interpolation_function = self._create_interpolation_function_from_zenith_collection(self.bkg_dict) if len(self.bkg_dict) > 1 else None
            self.interpolation_function_exist = True
        if self.interpolation_function is None:
            logger.warning('Only one zenith bin, zenith interpolation deactivated')

    def get_interpolation_function(self, azimuth: u.Quantity=None):
        """
        Retrieve the appropriate interpolation function

        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        interp_func : scipy.interpolate.interp1d
            The object that could be call directly for performing the interpolation
        """
        self.generate_interpolation_function()
        return self.interpolation_function


class BackgroundCollectionZenithSplitAzimuth(BackgroundCollection):

    def __init__(self,
                 bkg_dict_east: dict[float, BackgroundIRF] = None,
                 bkg_dict_west: dict[float, BackgroundIRF] = None,
                 **kwargs):
        """
            Create the class for storing a collection of model for different zenith angle

            Parameters
            ----------
            bkg_dict_east : dict of gammapy.irf.BackgroundIRF
                The collection of model in a dictionary with as key the zenith angle (in degree) associated to the model pointing east
            bkg_dict_west : dict of gammapy.irf.BackgroundIRF
                The collection of model in a dictionary with as key the zenith angle (in degree) associated to the model pointing west
            **kwargs:
                Arguments for the base class, see docstring of BackgroundCollection
        """
        super().__init__(**kwargs)
        self.interpolation_function_west = None
        self.interpolation_function_east = None
        bkg_dict_east = bkg_dict_east or {}
        self.bkg_dict_east = {}
        for k, v in bkg_dict_east.items():
            key = float(k)
            self._check_entry(key, v)
            self.bkg_dict_east[key] = v
        bkg_dict_west = bkg_dict_west or {}
        self.bkg_dict_west = {}
        for k, v in bkg_dict_west.items():
            key = float(k)
            self._check_entry(key, v)
            self.bkg_dict_west[key] = v

    def get_zenith(self, azimuth:u.Quantity=None):
        """
        Return zenith of the available models
        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want to have the list of available zenith

        Returns
        -------
        keys : np.array
            The zenith angle available in degree
        """
        if azimuth.to_value(u.deg)%360 <= 180:
            return np.sort(np.array(list(self.bkg_dict_east.keys())))
        else:
            return np.sort(np.array(list(self.bkg_dict_west.keys())))

    def get_model_from_collection(self, zenith: u.Quantity, azimuth: u.Quantity = None):
        """
        Return zenith of the available models
        Parameters
        ----------
        zenith: float
            the zenith of the model, must exist in the return of get_zenith
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        if azimuth.to_value(u.deg)%360 <= 180:
            return self.bkg_dict_east[zenith]
        else:
            return self.bkg_dict_west[zenith]

    def generate_interpolation_function(self):
        """
        Generate the interpolation functions
        """
        if not self.interpolation_function_exist:
            self.interpolation_function_east = self._create_interpolation_function_from_zenith_collection(self.bkg_dict_east) if len(self.bkg_dict_east) > 1 else None
            self.interpolation_function_west = self._create_interpolation_function_from_zenith_collection(self.bkg_dict_west) if len(self.bkg_dict_west) > 1 else None
            self.interpolation_function_exist = True
            if self.interpolation_function_east is None:
                logger.warning('Only one zenith bin, zenith interpolation deactivated for east pointing')
            if self.interpolation_function_west is None:
                logger.warning('Only one zenith bin, zenith interpolation deactivated for west pointing')

    def get_interpolation_function(self, azimuth: u.Quantity):
        """
        Retrieve the appropriate interpolation function

        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------
        interp_func : scipy.interpolate.interp1d
            The object that could be call directly for performing the interpolation
        """
        self.generate_interpolation_function()
        if azimuth.to_value(u.deg)%360 <= 180:
            return self.interpolation_function_east
        else:
            return self.interpolation_function_west
