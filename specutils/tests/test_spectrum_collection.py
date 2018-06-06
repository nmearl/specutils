import astropy.units as u
import numpy as np

from ..spectra.spectrum1d import Spectrum1D
from ..spectra.spectrum_collection import SpectrumCollection


def test_create_spectrum_collection():
    spec = Spectrum1D(spectral_axis=np.linspace(0, 50, 50) * u.AA,
                      flux=np.random.randn(50) * u.Jy)
    spec1 = Spectrum1D(spectral_axis=np.linspace(0, 50, 25) * u.AA,
                       flux=np.random.randn(25) * u.Jy)

    # By default, the output grid is defined by the coarsest bin
    sc = SpectrumCollection([spec, spec1])

    assert len(sc) == 2
    assert isinstance(sc[0], Spectrum1D)
    assert sc.flux.shape == (2, 24)
    assert isinstance(sc.flux, u.Quantity)

    # Force the output grid to be the finest available bin
    sc = SpectrumCollection([spec, spec1], output_grid='fine')

    assert sc.flux.shape == (2, 50)
