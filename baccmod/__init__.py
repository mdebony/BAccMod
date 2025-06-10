# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: __init__.py
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------

from importlib import import_module

__all__ = [
    "base_acceptance_map_creator",
    "base_fit_acceptance_map_creator",
    "grid3d_acceptance_map_creator",
    "radial_acceptance_map_creator",
    "spatial_fit_acceptance_map_creator",
    "bkg_collection",
    "exception",
    "logging",
    "modeling",
    "toolbox",
]

_alias_map = {
    "BaseAcceptanceMapCreator": "baccmod.base_acceptance_map_creator",
    "BaseFitAcceptanceMapCreator": "baccmod.base_fit_acceptance_map_creator",
    "Grid3DAcceptanceMapCreator": "baccmod.grid3d_acceptance_map_creator",
    "RadialAcceptanceMapCreator": "baccmod.radial_acceptance_map_creator",
    "SpatialFitAcceptanceMapCreator": "baccmod.spatial_fit_acceptance_map_creator",
    "BackgroundCollectionZenith": "baccmod.bkg_collection",
    "BackgroundCollectionZenithSplitAzimuth": "baccmod.bkg_collection",
}

def __getattr__(name: str):
    mod = _alias_map.get(name)
    if mod is None:
        raise AttributeError(f"module 'baccmod' has no attribute {name!r}")
    return getattr(import_module(mod), name)
