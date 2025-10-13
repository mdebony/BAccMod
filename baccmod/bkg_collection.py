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
from copy import copy
from typing import Dict, Union

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
        Base class for background irf collections, defining basic verification and access structure as well as
        interpolation settings.

        Parameters
        ----------
        bkg:
            The attribute containing the background IRFs
        interpolation_type: str, optional
            Select the type of interpolation to be used, could be either "log" or "linear",
            log tend to provided better results be could more easily create artefact that will cause issue
        activate_interpolation_cleaning: bool, optional
            If true, will activate the cleaning step after interpolation,
            it should help to eliminate artefact caused by interpolation
        interpolation_cleaning_energy_relative_threshold: float, optional
            To be kept, each bin in energy needs at least one adjacent bin with a relative difference within this range
        interpolation_cleaning_spatial_relative_threshold: float, optional
            To be kept, each bin in space needs at least one adjacent bin with a relative difference within this range
        """
        self.bkg = bkg
        self.type_model = None
        self.axes_model = None
        self.unit_model = None
        self.shape_model = None
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
        Given a set of identification parameters, return the associated BackgroundIRF.

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        pass

    @abstractmethod
    def get_closest_model(self, *args, **kwargs):
        """
        In collection with multiple model, find the one closest to the requested parameters.

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        pass

    @abstractmethod
    def _generate_interpolation_function(self, *args, **kwargs):
        """
        Generate interpolation functions.
        """
        pass

    def generate_interpolation_function(self, *args, **kwargs):
        """
        Generate the interpolation functions
        Use self.interpolation_function_exist to avoid multiple generation
        """
        if not self.interpolation_function_exist:
            self._generate_interpolation_function(*args, **kwargs)
            self.interpolation_function_exist = True

    @abstractmethod
    def _interpolate(self, *args, **kwargs):
        """
        Perform interpolation between available IRFs in the collection.

        Returns
        -------
        interp_bkg : astropy.units.Quantity
            Array of interpolated data
        """
        pass

    def get_interpolated_model(self, *args, **kwargs):
        """
        Perform interpolation between available IRFs in the collection to create a new BackgroundIRF.

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
            The interpolated IRF
        """
        if not self.consistent_bkg:
            raise BackgroundModelFormatException("Interpolation impossible with inconsistent background models.")

        # Perform the interpolation
        interp_bkg = self._interpolate(*args, **kwargs)
        if self.activate_interpolation_cleaning:
            interp_bkg = self._background_cleaning(interp_bkg)

        # Return the model
        return self.type_model(axes=self.axes_model,
                               data=interp_bkg * self.unit_model,
                               fov_alignment=self.fov_alignment)

    def _background_cleaning(self, background_model):
        """
            Cleans the background model from suspicious values not compatible with neighbour pixels.

            Parameters
            ----------
            background_model : numpy.ndarray
                The background model to be cleaned

            Returns
            -------
            background_model : numpy.ndarray
                The background model cleaned
        """

        base_model = copy(background_model)
        final_model = copy(background_model)
        i = 0
        while (i < 1 or not np.allclose(base_model, final_model)) and (i < self.max_cleaning_iteration):
            base_model = copy(final_model)
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
        """
        Verify that the type of the objects in the collection are BackgroundIRF.

        Parameters
        ----------
        v: BackgroundIRF
            Value whose type will be checked.
        error_message: str, optional
            Current content of the error message.

        Returns
        -------
        error_message: str
            Updated error message.
        """
        if not isinstance(v, BackgroundIRF):
            error_message += 'Invalid type : model should be a BackgroundIRF.'
        return error_message

    def _check_ref(self, v):
        """
        Verify that all BackgroundIRF have identical properties.
        Required for interpolation.

        Parameters
        ----------
        v: BackgroundIRF
            Object whose properties will be compared to previously setup expectations.
        """
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
            if v.data.shape != self.shape_model:
                warning_message += f'Inconsistent shape,not all {self.shape_model}.'
                self.consistent_bkg = False
            if v.fov_alignment != self.fov_alignment:
                warning_message += f'Inconsistent fov_alignment, not all {self.fov_alignment}.'
                self.consistent_bkg = False
            if not self.consistent_bkg:
                logger.warning(warning_message)

    @abstractmethod
    def check_bkg(self, **kwargs):
        """
        Perform checks to verify that the data received is of the correct type and properties.
        Typically, should include calls to _check_model and _check_ref.
        """
        pass


class BackgroundCollectionZenith(BackgroundCollection):

    def __init__(self, bkg_dict: dict[float, BackgroundIRF] = None, **kwargs):
        """
            Create the class for storing a collection of model for different zenith angles.

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

        Returns
        -------
        keys : astropy.units.Quantity
            The zenith angle available in degree
        """
        return np.sort(np.array(list(self.bkg.keys())))*u.deg

    def __getitem__(self, item):
        return self.bkg[item]

    def get_model(self, zenith: u.Quantity, *args, **kwargs):
        """
        Return model at the requested zenith angle.
        Parameters
        ----------
        zenith: u.Quantity
            the zenith of the model, must be a valid entry.

        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        return self[zenith.to_value(u.deg)]

    def get_closest_model(self, zenith: u.Quantity, *args, **kwargs):
        """
        Return the model closest to a given zenith.
        Parameters
        ----------
        zenith: u.Quantity
            the zenith for which a model is requested
        Returns
        -------
        model : gammapy.irf.BackgroundIRF
        """
        cos_zenith_observation = np.cos(zenith)
        zenith_model = self.get_zenith()
        cos_zenith_model = np.cos(np.deg2rad(zenith_model))
        key_closest_model = zenith_model[np.abs(cos_zenith_model-cos_zenith_observation).argmin()]
        return self.get_model(key_closest_model)

    def _create_interpolation_function(self, base_model: Dict[float, BackgroundIRF]):
        """
            Create the function that will perform the interpolation from a dictionary of BackgroundIRF.

            Parameters
            ----------
            base_model : dict of gammapy.irf.background.BackgroundIRF
                Each key of the dictionary should correspond to the zenith in degree of the model

            Returns
            -------
            interp_func : wrapper for scipy.interpolate.interp1d
                The object that could be called directly to perform the interpolation
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
            return interp_bkg

        return inter_wrapper

    def _generate_interpolation_function(self):
        """
        Generate the interpolation function. Support both the single model case and multi models case.
        """
        if len(self.bkg)==1:
            logger.warning('Only one zenith bin, zenith interpolation deactivated')
            self.interpolation_function = lambda x: self.get_model(self.get_zenith()[0]).data
        else:
            self.interpolation_function = self._create_interpolation_function(self.bkg)

    def _interpolate(self, zenith: u.Quantity, *args, **kwargs):
        """
        Performs the zenith interpolation, after generating the interpolation function if it was not done before.

        Parameters
        ----------
        zenith: u.Quantity
            The targeted zenith for the interpolation.
        """
        self.generate_interpolation_function()
        return self.interpolation_function(zenith)

    def _check_entry(self, key, v, error_message=''):
        """
        Verify that the items of the provided dictionary are valid zenith angles, BackgroundIRF pairs.
        Parameters
        ----------
        key: float
            zenith angle
        v: BackgroundIRF
        error_message: str, optional
        """
        if key > 90.0 or key < 0.0:
            error_message += ('Invalid key : The zenith associated with the model should be between 0 and 90 in degree,'
                              ' ')+str(key)+' provided.\n'
        self._check_model(v, error_message=error_message)

    def check_bkg(self, error_message='', extra_context=''):
        """
        Perform checks to verify that the data received is of the correct type and properties.

        Parameters
        ----------
        error_message: str, optional
            Starting error message.
        extra_context: str, optional
            Can be used to extend a potential error message without triggering it if no issues are found.
        """
        ref_bkg = next(iter(self.bkg.values()))
        self.type_model = type(ref_bkg)
        self.axes_model = ref_bkg.axes
        self.unit_model = ref_bkg.unit
        self.shape_model = ref_bkg.data.shape
        self.fov_alignment = ref_bkg.fov_alignment
        for k, v in self.bkg.items():
            key = float(k)
            self._check_ref(v)
            self._check_entry(key, v, error_message=error_message)
            if error_message != '':
                raise BackgroundModelFormatException(extra_context+error_message)


class BackgroundCollectionZenithSplitAzimuth(BackgroundCollection):

    def __init__(self,
                 bkg_east: Union[BackgroundCollectionZenith,BackgroundIRF],
                 bkg_west: Union[BackgroundCollectionZenith,BackgroundIRF],
                 **kwargs):
        """
            Create the class for storing a collection of model split in azimuth and for different zenith angle

            Parameters
            ----------
            bkg_east : BackgroundCollectionZenith or BackgroundIRF
                The collection of model associated to the model pointing east
            bkg_west : BackgroundCollectionZenith or BackgroundIRF
                The collection of model associated to the model pointing west
            **kwargs:
                Arguments for the base class, see docstring of BackgroundCollection
        """
        if isinstance(bkg_east, BackgroundIRF):
            bkg_east = BackgroundCollectionZenith({45.0:bkg_east}, **kwargs)
        if isinstance(bkg_west, BackgroundIRF):
            bkg_west = BackgroundCollectionZenith({45.0:bkg_west}, **kwargs)
        super().__init__(bkg = {'east':bkg_east,
                                'west':bkg_west},
                         **kwargs)

    @staticmethod
    def eastwest(azimuth:u.Quantity):
        """
        Associate an azimuth angle value to an east or west pointing.
        Parameters
        ----------
        azimuth: astropy.Quantity

        Returns
        -------
        str : 'east' or 'west'
        """
        if azimuth.to_value(u.deg)%360 <= 180:
            return 'east'
        else:
            return 'west'

    def interpolation_functions(self, az_key:str):
        """
        Gives access to the east and west zenith interpolation functions.
        Parameters
        ----------
        az_key: str
            'east' or 'west'

        Returns
        -------
            The interpolation function
        """
        return self.bkg[az_key].interpolation_function

    def get_zenith(self, azimuth:u.Quantity=None):
        """
        Return zenith of the available models for a given east/west pointing
        Parameters
        ----------
        azimuth: u.Quantity
            the azimuth for which you want to have the list of available zenith

        Returns
        -------
        keys : astropy.units.Quantity
            The zenith angle available in degree
        """
        return self.bkg[self.eastwest(azimuth)].get_zenith()

    def get_closest_model(self, zenith: u.Quantity, azimuth: u.Quantity, *args, **kwargs):
        """
        Return the closest model for a given zenith and east/west pointing
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
        return self.bkg[self.eastwest(azimuth)].get_closest_model(zenith)

    def get_model(self, zenith: u.Quantity, azimuth: u.Quantity, *args, **kwargs):
        """
        Return model at the requested zenith angle.
        Parameters
        ----------
        zenith: u.Quantity
            the zenith of the model, must be a valid entry.
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

    def _interpolate(self, zenith: u.Quantity, azimuth: u.Quantity, *args, **kwargs):
        """
        Performs the zenith interpolation for the correct east/west pointing,
        after generating the interpolation function if it was not done before.

        Parameters
        ----------
        zenith: u.Quantity
            The targeted zenith for the interpolation.
        azimuth: u.Quantity
            the azimuth for which you want the model

        Returns
        -------

        """
        self.generate_interpolation_function()
        return self.interpolation_functions(self.eastwest(azimuth))(zenith)

    def check_bkg(self, **kwargs):
        """
        Perform checks to verify that the data received is of the correct type and properties.
        """
        for k, v in self.bkg.items():
            v.check_bkg(extra_context=k+': ')
        self.type_model = self.bkg['east'].type_model
        self.axes_model = self.bkg['east'].axes_model
        self.unit_model = self.bkg['east'].unit_model
        self.shape_model = self.bkg['east'].shape_model
        self.fov_alignment = self.bkg['east'].fov_alignment

        if (self.type_model != self.bkg['west'].type_model or
            self.axes_model != self.bkg['west'].axes_model or
            self.unit_model != self.bkg['west'].unit_model or
            self.shape_model != self.bkg['west'].shape_model or
            self.fov_alignment != self.bkg['west'].fov_alignment):
            logger.warning("BackgroundIRF east and west have different properties.")


