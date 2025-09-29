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
from gammapy.irf.background import BackgroundIRF
import astropy.units as u
from scipy.interpolate import interp1d

from .exception import BackgroundModelFormatException
from .toolbox import compute_neighbour_condition_validation

logger = logging.getLogger(__name__)

class BackgroundCollection(ABC):

    def __init__(self,
                 bkg,
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
        self.bkg = bkg
        self.type_model = None
        self.axes_model = None
        self.unit_model = None
        self.fov_alignment = None
        self.consistent_bkg = True
        self.check_bkg()
        self.interpolation_type = interpolation_type
        self.threshold_value_log_interpolation = threshold_value_log_interpolation
        self.activate_interpolation_cleaning = activate_interpolation_cleaning
        self.interpolation_cleaning_energy_relative_threshold = interpolation_cleaning_energy_relative_threshold
        self.interpolation_cleaning_spatial_relative_threshold = interpolation_cleaning_spatial_relative_threshold
        self.max_cleaning_iteration = 50

        self.interpolation_function_exist = False

    @abstractmethod
    def get_model(self, *args, **kwargs):
        """

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        pass

    @abstractmethod
    def get_binned_model(self, *args, **kwargs):
        """

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        pass

    @abstractmethod
    def _generate_interpolation_function(self, *args, **kwargs):
        """
        Generate the interpolation functions
        """
        pass

    def generate_interpolation_function(self, *args, **kwargs):
        """
        Generate the interpolation functions
        In the implementation, could use self.interpolation_function_exist to avoid multiple generation
        """
        if not self.interpolation_function_exist:
            self._generate_interpolation_function(*args, **kwargs)
            self.interpolation_function_exist = True

    @abstractmethod
    def _interpolate(self, *args, **kwargs):
        """
        Returns
        -------
        interp_bkg : gammapy.irf.BackgroundIRF
        """
        pass

    def get_interpolated_model(self, *args, **kwargs):
        """
        Return the interpolated model

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        if not self.consistent_bkg:
            raise BackgroundModelFormatException("Interpolation impossible with inconsistent background models.")

        # Perform the interpolation
        interp_bkg = self._interpolate(*args, **kwargs)
        if self.activate_interpolation_cleaning:
            interp_bkg = self._background_cleaning(interp_bkg)

        # Return the model
        return self.type_model(axes=self.axes_model,
                               data=interp_bkg,
                               fov_alignment=self.fov_alignment)

    def _background_cleaning(self, background_model):
        """
            Cleans the background model from suspicious values not compatible with neighbour pixels.

            Parameters
            ----------
            background_model : numpy.array
                The background model to be cleaned

            Returns
            -------
            background_model : numpy.array
                The background model cleaned
        """

        base_model = background_model.copy().value
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


    @staticmethod
    def _check_model(v, error_message=''):
        if not isinstance(v, BackgroundIRF):
            error_message += 'Invalid type : model should be a BackgroundIRF.'
        return error_message

    def _check_ref(self, v):
        if self.consistent_bkg:
            warning_message = ''
            if not isinstance(v, self.type_model):
                warning_message += f'Inconsistent type, not all {self.type_model}. '
                self.consistent_bkg = False
            if v.axes != self.axes_model:
                warning_message += f'Inconsistent axes.'
                self.consistent_bkg = False
            if v.unit != self.unit_model:
                warning_message += f'Inconsistent units,not all {self.unit_model}.'
                self.consistent_bkg = False
            if v.fov_alignment != self.fov_alignment:
                warning_message += f'Inconsistent fov_alignment, not all {self.fov_alignment}.'
                self.consistent_bkg = False
            if not self.consistent_bkg:
                logger.warning(warning_message)


    def check_bkg(self, **kwargs):
        pass


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
        super().__init__(bkg = bkg_dict, **kwargs)
        self.interpolation_function = None

    def get_zenith(self, *args, **kwargs):
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
        return np.sort(np.array(list(self.bkg.keys())))*u.deg

    def __getitem__(self, item):
        return self.bkg[item]

    def get_model(self, zenith: u.Quantity, *args, **kwargs):
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
        return self[zenith.to_value(u.deg)]

    def get_binned_model(self, zenith: u.Quantity, *args, **kwargs):
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
        zenith_model = self.get_zenith()
        cos_zenith_model = np.cos(np.deg2rad(zenith_model))
        key_closest_model = zenith_model[np.abs(cos_zenith_model-cos_zenith_observation).argmin()]
        return self.get_model(key_closest_model)

    def _create_interpolation_function_from_zenith_collection(self, base_model: Dict[float, BackgroundIRF]):
        """
            Create the function that will perform the interpolation

            Parameters
            ----------
            base_model : dict of gammapy.irf.background.BackgroundIRF
                The binned base model
                Each key of the dictionary should correspond to the zenith in degree of the model

            Returns
            -------
            interp_func : wrapper for scipy.interpolate.interp1d
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

        def inter_wrapper(zenith):
            interp_bkg = interp_func(np.cos(zenith))
            if self.interpolation_type == 'log':
                interp_bkg = (10. ** interp_bkg)
                interp_bkg[interp_bkg < 100 * self.threshold_value_log_interpolation] = 0.
            elif self.interpolation_type == 'linear':
                interp_bkg[interp_bkg < 0.] = 0.
            return interp_bkg * self.unit_model

        return inter_wrapper

    def _generate_interpolation_function(self):
        """
        Generate the interpolation functions
        """
        if len(self.bkg) <=1:
            logger.warning('Only one zenith bin, zenith interpolation deactivated')
            self.interpolation_function = lambda x: self.get_model(self.get_zenith()[0]).data
        else:
            self.interpolation_function = self._create_interpolation_function_from_zenith_collection(self.bkg)


    def _interpolate(self, zenith: u.Quantity, *args, **kwargs):
        """


        """
        self.generate_interpolation_function()
        return self.interpolation_function(zenith)


    def _check_entry(self, key, v, error_message=''):
        if key > 90.0 or key < 0.0:
            error_message += ('Invalid key : The zenith associated with the model should be between 0 and 90 in degree,'
                              ' ')+str(key)+' provided.\n'
        self._check_model(v, error_message=error_message)

    def check_bkg(self, error_message='', extra_context=''):
        ref_bkg = next(iter(self.bkg.values()))
        self.type_model = type(ref_bkg)
        self.axes_model = ref_bkg.axes
        self.unit_model = ref_bkg.unit
        self.fov_alignment = ref_bkg.fov_alignment
        for k, v in self.bkg.items():
            key = float(k)
            self._check_ref(v)
            self._check_entry(key, v, error_message=error_message)
            if error_message != '':
                raise BackgroundModelFormatException(extra_context+error_message)


class BackgroundCollectionZenithSplitAzimuth(BackgroundCollection):

    def __init__(self,
                 bkg_east: BackgroundCollectionZenith = None,
                 bkg_west: BackgroundCollectionZenith = None,
                 **kwargs):
        """
            Create the class for storing a collection of model for different zenith angle

            Parameters
            ----------
            bkg_east : BackgroundCollectionZenith
                The collection of model associated to the model pointing east
            bkg_west : BackgroundCollectionZenith
                The collection of model associated to the model pointing west
            **kwargs:
                Arguments for the base class, see docstring of BackgroundCollection
        """
        super().__init__(bkg = {'east':bkg_east,
                                'west':bkg_west}
                         ,**kwargs)

    @staticmethod
    def eastwest(azimuth:u.Quantity):
        if azimuth.to_value(u.deg)%360 <= 180:
            return 'east'
        else:
            return 'west'

    def interpolation_functions(self, az_key):
        return self.bkg[az_key].interpolation_function

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
        return self.bkg[self.eastwest(azimuth)].get_zenith()

    def get_binned_model(self, zenith: u.Quantity, azimuth: u.Quantity, *args, **kwargs):
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
        return self.bkg[self.eastwest(azimuth)].get_binned_model(zenith)

    def get_model(self, zenith: u.Quantity, azimuth: u.Quantity, *args, **kwargs):
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
        return self.bkg[self.eastwest(azimuth)][zenith.to_value(u.deg)]

    def _generate_interpolation_function(self):
        """
        Generate the interpolation functions
        """
        self.bkg['east'].generate_interpolation_function()
        self.bkg['west'].generate_interpolation_function()
        if self.interpolation_functions('east') is None:
            logger.warning('Only one zenith bin, zenith interpolation deactivated for east pointing')
        if self.interpolation_functions('west') is None:
            logger.warning('Only one zenith bin, zenith interpolation deactivated for west pointing')

    def _interpolate(self, azimuth: u.Quantity, zenith: u.Quantity, *args, **kwargs):
        """
        Retrieve the appropriate interpolation function

        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want the model
        zenith: u.Quantity

        Returns
        -------

        """
        self.generate_interpolation_function()
        return self.interpolation_functions(self.eastwest(azimuth))(zenith)

    def check_bkg(self, **kwargs):
        for k, v in self.bkg.items():
            v.check_bkg(extra_context=k+': ')
        self.type_model = self.bkg['east'].type_model
        self.axes_model = self.bkg['east'].axes_model
        self.unit_model = self.bkg['east'].unit_model
        self.fov_alignment = self.bkg['east'].fov_alignment

