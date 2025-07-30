from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Open the x1d FITS file
hdul = fits.open('fits/1x1d.fits')
data = hdul['EXTRACT1D'].data

# Extract the first row of the table
wavelength_um = np.array(data['WAVELENGTH'])       
flux_Jy = np.array(data['FLUX'])                  
error_Jy = np.array(data['FLUX_ERROR'])           

# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(wavelength_um, flux_Jy, label='Flux', color='darkorange')
#plt.fill_between(wavelength_um, flux_Jy - error_Jy, flux_Jy + error_Jy, color='orange', alpha=0.3, label='1σ Error')
plt.xlabel("Wavelength (μm)")
plt.ylabel("Flux (Jy)")
plt.title("JWST Extracted Spectrum")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
