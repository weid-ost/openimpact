"""Support for logger creation and logging to file."""
import pdb

# Copyright (C) 2023 OST Ostschweizer Fachhochschule
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Author: Juan Pablo Carbajal <juanpablo.carbajal@ost.ch>

from typing import Union
from pathlib import Path
import logging

from rich.logging import RichHandler

from . import __name__ as pkg_name

default_logger_name = pkg_name
""" Default loger name. """

default_log_filename = f"{default_logger_name}.log"
""" Default filename for logging. """


def _replace_handlers_safe_sideeffect(
    logger: logging.Logger, handlers: list[logging.Handler]
):
    # Avoid that multiple handlers are piled up on the logger
    # that is, though the logger is unique, the handlers can be repeated
    # e.g. printing twice to stdout, etc.
    # Add handlers only if the logger has none
    # if not len(logger.handlers):
    # remove all handlers and close them to avoid ResourceWarning
    hdls_ = logger.handlers[:]
    for h in hdls_:
        logger.removeHandler(h)
        h.close()

    for hdl in handlers:
        logger.addHandler(hdl)


def default_formatter() -> logging.Formatter:
    """Create default logger formatter.

    Returns
    -------
    formatter:
        A logger formatter which adds a timestamp to the log message.

    """
    formatter = logging.Formatter(
        "{asctime} {name}:{lineno:d} {levelname} - {message}",
        datefmt="%d.%m.%Y %H:%M:%S %Z",
        style="{",
    )
    return formatter


# create console handler
def default_handler() -> logging.Handler:
    """Create default logger handler.

    Returns
    -------
    handler:
        A logger handler which logs to stdout.

    """
    handler = logging.StreamHandler()

    # add formatter to handler
    handler.setFormatter(default_formatter())
    return handler


def default_logger(name: str = default_logger_name) -> logging.Logger:
    """Create default logger.

    Returns
    -------
    logger:
        A logger which logs to stdout with severity INFO.

    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.INFO)
    _replace_handlers_safe_sideeffect(logger_, [default_handler()])
    return logger_


def default_stdout_handler(level: str = "INFO"):
    """A handler which logs to stdout.

    Parameters
    ---------
    level:
        Logging level.
    """
    handler = RichHandler(
        level=level, show_level=False, show_time=False, rich_tracebacks=True
    )
    return handler


def default_filtered_stdout_handler(*, name: str, level: str = "INFO"):
    """Add a filter to :func:`~.default_stdout_handler`.

       The filter picks only records from loggers of the given name and
       of the save level as the handler returned by :func:`~.default_stdout_handler`.

    Parameters
    ---------
    name:
        Name of the logger from which records will be logged.
    level:
        Logging level.
    """
    handler = default_stdout_handler(level=level)
    f_level = handler.level if handler.level != logging.NOTSET else None

    def std_filter(x):
        ret = (x.levelno == f_level) if f_level is not None else True
        return (name in x.name) and ret

    handler.addFilter(std_filter)
    return handler


def file_logger(
    filename: Union[str, Path] = default_log_filename,
    *,
    name: str = default_logger_name,
    stdout: Union[logging.Handler, bool] = False,
    level: str = "DEBUG",
    stdout_level: str = "INFO",
) -> logging.Logger:
    """Create file logger.

    Parameters
    ----------
    filename:
        Filename to write to. Contents will be overwritten.
    name:
        Name of the logger to attach to.
    stdout:
        Handler to use for logs to stdout. If True, the output of
        :func:`~.default_filtered_stdout_handler`
        will be used.
    level:
        Level of the logging.
    stdout_level:
        Level of the logging for stdout.

    Returns
    -------
    logger:
        A logger which also logs to filename, overwriting the file every time.

    """
    _logger = logging.getLogger(name)
    lvl = getattr(logging, level.upper())
    _logger.setLevel(lvl)
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(default_formatter())
    handlers = [file_handler]
    if stdout:
        if isinstance(stdout, bool):
            stdout_handler = default_filtered_stdout_handler(
                name=name, level=stdout_level
            )
            handlers.append(stdout_handler)
        elif isinstance(stdout, logging.Handler):
            handlers.append(stdout)
        else:
            raise TypeError(f"Wrong type for stdout parameter: {type(stdout)}")

    _replace_handlers_safe_sideeffect(_logger, handlers)
    return _logger
