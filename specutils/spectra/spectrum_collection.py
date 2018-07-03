import numpy as np
from .spectrum1d import Spectrum1D
from astropy.units import Quantity
from collections import MutableSequence
from astropy.nddata import NDIOMixin


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
        self._items[index] = value

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


class ResampleMixin:
    """
    Implements resampling behavior as a mixin class.
    """
    def deposit(self, output_grid):
        """
        Deposit spectrum list onto a new dispersion grid.

        Parameters
        ----------
        output_grid : str, array-like, tuple
            Define the dispersion grid to which all spectra in the collection
            will be resampled. If 'fine', the smallest dispersion bin in all
            spectra will be user; if 'coarse', then the largest will be used.
            Users can also supply a custom dispersion array, or a tuple of the
            form (start_bin, end_bin, bin_shape).
        """
        # First define the output dispersion grid
        if isinstance(output_grid, np.ndarray):
            self._output_grid = output_grid
        elif isinstance(output_grid, tuple):
            self._output_grid = np.arange(*output_grid)
        else:
            start_bin = min([x.spectral_axis[0] for x in self._items]).value
            end_bin = max([x.spectral_axis[-1] for x in self._items]).value

            if output_grid is None or output_grid == 'coarse':
                largest_bin = max([np.max(np.diff(x.spectral_axis))
                                   for x in self._items])
                self._output_grid = np.arange(
                    start_bin, end_bin, largest_bin.value) * largest_bin.unit
            elif output_grid == 'fine':
                smallest_bin = min([np.min(np.diff(x.spectral_axis))
                                    for x in self._items])
                self._output_grid = np.arange(
                    start_bin, end_bin, smallest_bin.value) * smallest_bin.unit

        # Next, resample the spectrum objects to this grid
        return [x.resample(self.output_grid) for x in self._items]

    @property
    def output_grid(self):
        return self._output_grid

    @property
    def shape(self):
        return (len(self._items), self.output_grid.size)


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
    def __init__(self, items, output_grid=None):
        super(SpectrumCollection, self).__init__(items)

        # Deposit the spectrum1d list onto a unified dispersion solution
        self._resampled_items = self.deposit(output_grid)

    def __getitem__(self, index):
        return self._resampled_items[index]

    @property
    def spectral_axis(self):
        """
        Two-dimensional :class:`~astropy.units.Quantity` object containing
        spectral axes of all collection items.
        """
        return self._get_attr('spectral_axis')

    @property
    def wavelength(self):
        """
        Two-dimensional :class:`~astropy.units.Quantity` object containing
        wavelength of all collection items.
        """
        return self._get_attr('wavelength')

    @property
    def frequency(self):
        """
        Two-dimensional :class:`~astropy.units.Quantity` object containing
        frequencies of all collection items.
        """
        return self._get_attr('frequency')

    @property
    def velocity(self):
        """
        Two-dimensional :class:`~astropy.units.Quantity` object containing
        velocities of all collection items.
        """
        return self._get_attr('velocity')

    @property
    def flux(self):
        """
        Two-dimensional :class:`~astropy.units.Quantity` object containing
        fluxes of all collection items.
        """
        return self._get_attr('flux')

    def _get_attr(self, name):
        """
        This is a proxy function that forces `SpectrumCollection` to behave
        like a `Spectrum1D` object without having to manually provide
        transient attributes that match the `Spectrum1D` API.
        """
        a = [getattr(x, name) for x in self._resampled_items]
        val = np.vstack(np.array(a)) * a[0].unit

        return val

    # def __getattribute__(self, name):
    #     """
    #     This is a proxy function that forces `SpectrumCollection` to behave
    #     like a `Spectrum1D` object without having to manually provide
    #     transient attributes that match the `Spectrum1D` API.
    #     """
    #     if hasattr(Spectrum1D, name) and name in ('flux', 'wavelength',
    #                                               'frequency', 'velocity'):
    #         a = [getattr(x, name) for x in self._resampled_items]
    #         val = np.vstack(np.array(a)) * a[0].unit

    #         return val

    #     return object.__getattribute__(self, name)

    def __repr__(self):
        return """<SpectrumCollection(size={}, shape={}>""".format(len(self),
                                                                   self.shape)
