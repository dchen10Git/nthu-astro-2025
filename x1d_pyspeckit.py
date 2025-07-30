from astropy.io import fits
import pyspeckit
import numpy as np
import matplotlib.pyplot as plt

hdul = fits.open('fits/1x1d.fits')
#hdul.info()

data = hdul['EXTRACT1D'].data
#print(hdul['EXTRACT1D'].columns)

wavelength = data['WAVELENGTH']  # 1D array of wavelengths (μm)
flux = data['FLUX']              # 1D array of fluxes (Jy)
error = data['FLUX_ERROR']       # 1D array of uncertainties

# Now build Spectrum object
spec = pyspeckit.Spectrum(xarr=wavelength,
                          data=flux,
                          error=error,
                          xarrkwargs={'unit': 'μm'})
spec.unit = 'Jy'

spec.plotter()
spec.plotter.savefig('1x1dspec.png')
