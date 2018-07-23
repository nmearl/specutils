import glob
import logging
import os

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy.wcs import WCS

from ...spectra.spectrum_collection import Spectrum1D, SpectrumCollection
from ..registers import data_loader


def identify_stis_fits(origin, *args, **kwargs):
    if isinstance(args[0], str) and (os.path.splitext(args[0].lower())[1] == '.fits'):
        with fits.open(args[0]) as f:
            return f[0].header.get('INSTRUME', '') == 'STIS'
    return False


@data_loader("stis-fits", identifier=identify_stis_fits,
             dtype=SpectrumCollection)
def stis_fits(file_name, ext=1, sdqflags=None, weights=None, output_grid='fine', **kwargs):
    """
    Loads a STIS file containing multiple `Spectrum1D` objects and stores them
    in a `SpectrumCollection`.

    Parameters
    ----------
    output_grid : str {'fine', 'coarse'}
        The grid onto which the `Spectrum1D` objects will be deposited.
    """
    name = os.path.basename(file_name.rstrip(os.sep)).rsplit('.', 1)[0]

    with fits.open(file_name) as hdulist:
        header = hdulist[0].header

        if sdqflags is None:
            sdqflags = hdulist[ext].header.get('SDQFLAGS', 31743)

        tab = Table.read(file_name, hdu=ext)

        wav_unit = hdulist[1].data.columns['WAVELENGTH'].unit.replace('Angstroms', 'Angstrom')
        data_unit = hdulist[1].data.columns['FLUX'].unit

    # Determine the FLT file matching the input X1D/SX1 file (look in same directory):
    dirname, basename = os.path.split(file_name)

    search_for = [
        os.path.join(dirname, basename.rsplit('_x1c.fits',1)[0] + '_flt.fits'),
        os.path.join(dirname, basename.rsplit('_s1c.fits',1)[0] + '_flt.fits'),
        os.path.join(dirname, basename.rsplit('_x1d.fits',1)[0] + '_flt.fits'),
        os.path.join(dirname, basename.rsplit('_sx1.fits',1)[0] + '_flt.fits')
    ]

    search_for.sort(key=len)
    search_for = search_for[0]
    flt_file = glob.glob(search_for)

    if len(flt_file) != 1:
        logging.warning("Unique FLT file not found in X1D's directory: {}".format(search_for))
        wcs = None
    else:
        flt_file = flt_file[0]
        # WCS must be generated from the FLT file:
        wcs = WCS(fits.getheader(flt_file, ext=('SCI', ext)))

    if weights is not None:
        # Require weight array to be 2D:
        weights = np.array(weights).squeeze()

        if len(np.shape(weights)) == 1:
            weights = np.array([weights])

        if np.shape(tab['WAVELENGTH']) != np.shape(weights):
            raise ValueError("Dimensionality of (optional) 'weights' array must match "
                            "WAVELENGTH/FLUX/ERROR/DQ arrays.")

    spectra = []

    for i in range(len(tab)):
        flags = tab[i]['DQ'] & sdqflags
        uncertainty = StdDevUncertainty(tab[i]['ERROR']) #* u(header[1].data.columns['ERROR'].unit)
        wave = tab[i]['WAVELENGTH'] * u.Unit(wav_unit)
        flux = tab[i]['FLUX'] * u.Unit(data_unit)
        meta = {'header': header, 'sdqflags': sdqflags, 'flags': flags, 'dq': tab[i]['DQ'],
                'sporder': tab[i]['SPORDER']}

        if weights is not None:
            meta['weight'] = weights[i,:]

        # WCS shouldn't overtake spectral_axis here!  Leave out WCS for now:
        spectra.append(
            Spectrum1D(spectral_axis=wave, flux=flux,
                       uncertainty=uncertainty, meta=meta))

    return SpectrumCollection(spectra)
