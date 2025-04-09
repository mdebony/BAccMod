# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: logging.py
# Purpose: Setup logging and related functions.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------
import logging
import sys

# Define custom logging levels
MOREINFO=15
logging.addLevelName(MOREINFO, "MOREINFO")

def set_log_level(level, module='baccmod'):
    if not 'baccmod' in module:
        module = 'baccmod.' + module
    logging.getLogger(module).setLevel(level)

def setup_logging_output(output_file=None, use_terminal=True,
                         handlers=None, module='baccmod',
                         fmode='w', term_level=logging.NOTSET,
                         log_format='%(levelname)s:%(name)s: %(message)s'):
    """
    Setup output streams to file. Also allows for more complex logging options,
    including directly providing handlers or a new message format.

    Parameters
    ----------
    output_file: str
        Path of the output file.
    use_terminal: bool
        Whether to use a terminal handler or not.
    handlers: list
        If provided, ignore other parameters and use the provided handlers.
    fmode: str
        Mode used to open the output file. Default overwrites existing file.
    term_level: int or str
        Logging level for the terminal output. Useful to have different logging
        level in the output file and terminal.
    module: str
        (Sub)module whose logger we configure. Cn only be a submodule of BAccMod.
    log_format: str
        Logging format.

    """
    if not 'baccmod' in module:
        module = 'baccmod.' + module
    logger = logging.getLogger(module)
    formatter = logging.Formatter(log_format)
    if handlers is not None:
        for h in handlers:
            h.setFormatter(formatter)
    else:
        handlers=[]
        if output_file is not None:
            h = logging.FileHandler(output_file, mode=fmode)
            h.setFormatter(formatter)
            handlers.append(h)
        if use_terminal:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(formatter)
            h.setLevel(term_level)
            handlers.append(h)
    logger.handlers = handlers
