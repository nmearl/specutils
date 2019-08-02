from astropy.utils.decorators import lazyproperty
import numpy as np
import astropy.units as u


class SpectralAxisMixin:
    @property
    def spectral_axis_index(self):
        """
        Finds the spectral axis within the wcs by searching for the associated
        VO UCD1+ vocabulary. If more than more exists, it takes the first
        instance.
        """
        physical_types = np.array(self.wcs.world_axis_physical_types)

        res = np.where((physical_types == 'em.wl') |
                       (physical_types == 'em.energy') |
                       (physical_types == 'em.freq') |
                       (physical_types == 'em.wavenumber') |
                       (physical_types == 'spect.dopplerVeloc') |
                       (physical_types == 'spect.dopplerVeloc.opt') |
                       (physical_types == 'spect.dopplerVeloc.radio'))

        return zip(*res)[0]

    @lazyproperty
    def spectral_axis(self):
        """
        Returns a `SpectralCoord` object with the values of the spectral axis.
        """
        if len(self.flux) > 0:
            spectral_axis = self.wcs.pixel_to_world(
                np.arange(self.flux.shape[self.spectral_axis_index]))
        else:
            # After some discussion it was suggested to create the empty
            # spectral axis this way to better use the WCS infrastructure.
            # This is to prepare for a future where pixel_to_world might yield
            # something more than just a raw Quantity, which is planned for
            # the mid-term in astropy and possible gwcs.  Such changes might
            # necessitate a revisit of this code.
            dummy_spectrum = self.__class__(
                spectral_axis=[1, 2]*self.wcs.spectral_axis_unit,
                flux=[1, 2]*self.flux.unit)
            spectral_axis = dummy_spectrum.wcs.pixel_to_world([0])[1:]

        return spectral_axis

    @lazyproperty
    def wavelength(self):
        self.spectral_axis.


class SpectralDataMixin:
    @property
    def flux(self):
        """
        Converts the stored data and unit information into a quantity.
        Returns
        -------
        `~astropy.units.Quantity`
            Spectral data as a quantity.
        """
        return u.Quantity(self.data, unit=self.unit, copy=False)

    def with_flux_unit(self, unit, equivalencies=None, suppress_conversion=False):
        """
        Converts the flux data to the specified unit.  This is an in-place
        change to the object.
        Parameters
        ----------
        unit : str or `~astropy.units.Unit`
            The unit to conver the flux array to.
        equivalencies : list of equivalencies
            Custom equivalencies to apply to conversions.
            Set to spectral_density by default.
        suppress_conversion : bool
            Set to true if updating the unit without
            converting data values.

        Returns
        -------
        `~specutils.Spectrum1D`
            A new spectrum with the converted flux array
        """
        new_spec = self.copy()

        if not suppress_conversion:
            if equivalencies is None:
                equivalencies = u.spectral_density(self.spectral_axis)

            new_data = self.flux.to(
                unit, equivalencies=equivalencies)

            new_spec._data = new_data.value
            new_spec._unit = new_data.unit
        else:
            new_spec._unit = u.Unit(unit)

        return new_spec

    @property
    def velocity_convention(self):
        return self._velocity_convention

    def with_velocity_convention(self, velocity_convention):
        return self.__class__(flux=self.flux, wcs=self.wcs, meta=self.meta,
                              velocity_convention=velocity_convention)

    @property
    def rest_value(self):
        return self._rest_value

    @rest_value.setter
    def rest_value(self, value):
        if not hasattr(value, 'unit') or not value.unit.is_equivalent(u.Hz, u.spectral()):
            raise u.UnitsError(
                "Rest value must be energy/wavelength/frequency equivalent.")

        self._rest_value = value