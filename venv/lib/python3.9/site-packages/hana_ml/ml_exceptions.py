"""
This module contains exception and error classes for the API.

The following classes are available:

    * :class:`Error`
    * :class:`FitIncompleteError`
    * :class:`BadSQLError`
"""
class Error(Exception):
    """Base class for hana_ml exceptions."""

class FitIncompleteError(Error):
    """Exception class raised by performing predict or score without fit first."""
    pass

class BadSQLError(Error):
    """Raised if malformed tokens or unexpected comments are detected in SQL."""

class PALUnusableError(Error):
    """Raised if hana_ml cannot access a compatible version of PAL."""
