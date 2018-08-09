import logging
from collections import MutableSequence

import astropy.units as u
import numpy as np
from astropy.nddata import NDIOMixin, NDUncertainty

from .spectrum1d import Spectrum1D

__all__ = ['SpectrumCollection']


class SpectrumCollection(MutableSequence, NDIOMixin):
    """
    A container class for :class:`~specutils.Spectrum1D` objects. This allows 
    for operations to be performed over a set of spectrum objects. This class 
    behaves as if it were a :class:`~specutils.Spectrum1D` and supports
    pythonic collection operations like slicing, deleting, and inserting.

    Note
    ----
    Items in this collection must currently be the same shape. Items are not
    automatically resampled onto a shared grid. Currently, users should resample
    their spectra prior to creating a :class:`~specutils.SpectrumCollection`
    object.

    Parameters
    ----------
    items : list, ndarray
        A list of :class:`~specutils.Spectrum1D` objects to be held in the
        collection
    """
    def __init__(self, items):
        # Enforce that the shape of each item must be the same
        if not all((x.shape == items[0].shape for x in items)):
            raise Exception("Shape of all elements must be the same.")

        self._items = items

    def __setitem__(self, index, value):
        self._items[index] = value

    def __delitem__(self, index):
        del self._items[index]

    def insert(self, index, value):
        self._items.insert(index, value)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def _get_property(self, name):
        val = [getattr(x, name) for x in self._items]

        if name == 'uncertainty':
            val = [x.array if val is not None else 0 for x in val]

            logging.info(
                "`SpectrumCollection` assumes that all "
                "spectra have the same uncertainty type.")

            if self._items[0].uncertainty is not None:
                val = self._items[0].uncertainty.__class__(val)

        elif hasattr(val[0], 'unit'):
            val = np.vstack(np.array(val)) * val[0].unit

        return val

    def __getattr__(self, name):
        """
        This is a proxy function that forces `SpectrumCollection` to behave
        like a `Spectrum1D` object without having to manually provide
        transient attributes that match the `Spectrum1D` API.
        """
        # TODO: currently, this method assumes that all uncertainties share
        # the same uncertainty type.
        if hasattr(Spectrum1D, name):
            # First, check to see if this property is cached
            val = self._get_property(name)

            return val

        return object.__getattribute__(self, name)

    def __repr__(self):
        return """<SpectrumCollection(size={})>""".format(len(self))
