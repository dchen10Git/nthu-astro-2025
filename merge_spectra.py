from astropy.io import fits
import numpy as np
from spectral_cube import SpectralCube
from astropy.table import Table
from helpers import get_error_spectrum

def read_spectrum(fits_file, subtracted_fits_file, x, y):
    subtracted_cube = SpectralCube.read(subtracted_fits_file)
    wave = subtracted_cube.spectral_axis
    flux = subtracted_cube.unmasked_data[:, y, x]
    error = get_error_spectrum(fits_file, subtracted_cube, x, y)
    return wave, flux, error

def merge_overlap(w1, f1, e1, w2, f2, e2):
    middle = (max(w1[0], w2[0]) + min(w1[-1], w2[-1])) / 2
    left_w = w1[w1 < middle]
    left_f = f1[w1 < middle]
    left_e = e1[w1 < middle]
    right_w = w2[w2 > middle]
    right_f = f2[w2 > middle]
    right_e = e2[w2 > middle]

    wave_combined = np.concatenate([left_w, right_w])
    flux_combined = np.concatenate([left_f, right_f])
    error_combined = np.concatenate([left_e, right_e])

    return wave_combined, flux_combined, error_combined

x, y = 25, 34
w1, f1, e1 = read_spectrum('fits/4s3d.fits', 'fits/subtracted_cube_full4.fits', x, y)
w2, f2, e2 = read_spectrum('fits/5s3d.fits', 'fits/subtracted_cube_full5.fits', x, y)
w3, f3, e3 = read_spectrum('fits/6s3d.fits', 'fits/subtracted_cube_full6.fits', x, y)

# === Merge step-by-step ===
w12, f12, e12 = merge_overlap(w1, f1, e1, w2, f2, e2)
w123, f123, e123 = merge_overlap(w12, f12, e12, w3, f3, e3)

table = Table([w123, f123, e123], names=('wavelength', 'flux', 'error'))
table.write(f'fits/merged_spectrum{x}_{y}.fits', format='fits', overwrite=True)
print(f"Successfully saved as fits/merged_spectrum{x}_{y}.fits")