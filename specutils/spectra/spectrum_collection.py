import logging
from collections import MutableSequence

import astropy.units as u
import numpy as np
from astropy.nddata import NDIOMixin, NDUncertainty

from .spectrum1d import Spectrum1D

__all__ = ['SpectrumArray', 'SpectrumCollection', 'ResampleMixin']


class SpectrumArray(MutableSequence):
    """
    An array class to hold a set of :class:`~specutils.Spectrum1D` objects.
    This class does not implicitly attempt to resample onto a new wavelength
    grid.
    """
    def __init__(self, items):
        self._items = list(items)

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


class ResampleMixin:
    """
    Implements resampling behavior as a mixin class.
    """
    def deposit(self, grid):
        """
        Deposit spectrum list onto a new dispersion grid.

        Parameters
        ----------
        grid : str, array-like, tuple
            Define the dispersion grid to which all spectra in the collection
            will be resampled. If 'fine', the smallest dispersion bin in all
            spectra will be user; if 'coarse', then the largest will be used.
            Users can also supply a custom dispersion array, or a tuple of the
            form (start_bin, end_bin, bin_shape).
        """
        if isinstance(grid, np.ndarray):
            resample_grid = grid
        elif isinstance(grid, list):
            resample_grid = np.array(grid)
        elif isinstance(grid, tuple):
            resample_grid = np.arange(*grid)
        else:
            if isinstance(grid, str) and len(self._items) > 0:
                # Get the left bin edges of the start and end bins
                start_bin = min(
                    [x.spectral_axis[0] -
                     np.abs((x.spectral_axis[1] - x.spectral_axis[0]) * 0.5)
                     for x in self._items]).value
                end_bin = max(
                    [x.spectral_axis[-1] -
                     np.abs((x.spectral_axis[-1] - x.spectral_axis[-2]) * 0.5)
                     for x in self._items]).value

                if grid == 'coarse':
                    largest_bin = max([np.max(np.diff(x.spectral_axis))
                                       for x in self._items])

                    resample_grid = np.arange(
                        start_bin + largest_bin.value * 0.5,
                        end_bin + largest_bin.value * 0.5,  # Ensure the inclusion of the last bin
                        largest_bin.value)
                elif grid == 'fine':
                    smallest_bin = min([np.min(np.diff(x.spectral_axis))
                                        for x in self._items])

                    resample_grid = np.arange(
                        start_bin + smallest_bin.value * 0.5,
                        end_bin + smallest_bin.value * 0.5,  # Ensure the inclusion of the last bin
                        smallest_bin.value)
                elif grid == 'same':
                    bin_sizes = [np.mean((x.spectral_axis[1:] - x.spectral_axis[:-1]).value)
                                 for x in self._items]

                    if not np.allclose(bin_sizes[0], bin_sizes):
                        raise Exception(
                            "Bin sizes of all input spectra must be equal "
                            "when using the 'same' output grid type.")

                    resample_grid = np.sort(np.unique(np.concatenate(
                        [x.spectral_axis.value for x in self._items])))
            else:
                resample_grid = None

        # Ensure the unit information of the resample grid
        resample_grid = u.Quantity(resample_grid,
                                   self._items[0].spectral_axis_unit)

        if resample_grid is not None:
            resampled_spectra = [x.resample(resample_grid) for x in self._items]
        else:
            resampled_spectra = None

        # Next, resample the spectrum objects to this grid
        return resample_grid, resampled_spectra


class SpectrumCollection(SpectrumArray, ResampleMixin, NDIOMixin):
    """
    A container class for spectra which themselves do not share a single
    dispersion solution. :class:`~specutils.Spectrum1D` objects added to this
    collection are automatically rebinned onto the user-specified dispersion
    grid.

    Parameters
    ----------
    output_grid : str, array-like, tuple
        See the docstring in
        :class:`~specutils.spectra.spectrum_collection.ResampleMixin`.
    """
    def __init__(self, items=[], output_grid=None):
        super(SpectrumCollection, self).__init__(items)

        self._output_grid = output_grid if output_grid is not None else 'same'
        self._resampled_grid, self._resampled_items = None, None
        self._evaluate_grids()

    def __setitem__(self, *args, **kwargs):
        super(SpectrumCollection, self).__setitem__(*args, **kwargs)
        self._evaluate_grids()

    def __delitem__(self, *args, **kwargs):
        super(SpectrumCollection, self).__delitem__(*args, **kwargs)
        self._evaluate_grids()

    def insert(self, *args, **kwargs):
        super(SpectrumCollection, self).insert(*args, **kwargs)
        self._evaluate_grids()

    def append(self, *args, **kwargs):
        super(SpectrumCollection, self).append(*args, **kwargs)
        self._evaluate_grids()

    @property
    def output_grid(self):
        return self._output_grid

    @output_grid.setter
    def output_grid(self, value):
        """
        If the output grid value is changed, force the `SpectrumCollection` to
        re-evaluate the resampled grid.
        """
        self._output_grid = value
        self._evaluate_grids()

    def _evaluate_grids(self):
        self._resampled_grid, self._resampled_items = \
            self.deposit(self.output_grid)

    def with_output_grid(self, output_grid):
        """
        Re-deposit the spectra onto a new output grid.
        """
        return SpectrumCollection(self._items, output_grid=output_grid)

    def __getattr__(self, name):
        """
        This is a proxy function that forces `SpectrumCollection` to behave
        like a `Spectrum1D` object without having to manually provide
        transient attributes that match the `Spectrum1D` API.
        """
        # TODO: currently, this method assumes that all uncertainties share
        # the same uncertainty type.
        if hasattr(Spectrum1D, name):
            val = [getattr(x, name) for x in self._resampled_items]

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

        return object.__getattr__(self, name)

    def __repr__(self):
        return """<SpectrumCollection(size={})>""".format(len(self))
