from dust_extinction.parameter_averages import CCM89
import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
from skimage.draw import line
from helpers import get_uJy_error_cube, get_pixel

def find_peak_flux(fits_file, cube, error_cube, x, y, wavelength, width=0.003*u.um):
    '''
    Returns peak flux within the given width along with its error
    '''
    delta = width / 2
    wmin = wavelength - delta
    wmax = wavelength + delta
    subcube = cube.spectral_slab(wmin, wmax)
    slice = subcube.unmasked_data[:, y, x]
    error_subcube = error_cube.spectral_slab(wmin, wmax)
    error_slice = error_subcube.unmasked_data[:, y, x]
    
    # Handle edge case
    if np.all(np.isnan(slice)):
        
        return np.nan, np.nan
        
    
    idx_max = np.nanargmax(slice)
    return np.nanmax(slice), error_slice[idx_max]

def calculate_Av(fits_file, cube, error_cube, lambda1, lambda2, x, y):
    # Define extinction model with RV = 5.5 (McClure 2019)
    ext_model = CCM89(Rv=5.5)

    wavelengths = np.array([lambda1.value, lambda2.value]) * u.um
    x_values = 1/wavelengths

    # β = A(λ)/A(V)
    beta_vals = ext_model(x_values)
    beta_lambda1, beta_lambda2 = beta_vals[0], beta_vals[1]

    # Observed flux ratio
    Fobs1, error1 = find_peak_flux(fits_file, cube, error_cube, x, y, lambda1)
    Fobs2, error2 = find_peak_flux(fits_file, cube, error_cube, x, y, lambda2)
    R_obs = Fobs1 / Fobs2
   
    # Intrinsic flux ratio, check database (values from paper)
    R_int = 0
    if lambda1 == 1.257*u.um:
        R_int = 1.1
    elif lambda1 == 1.321*u.um:
        R_int = 0.32
    elif lambda1 == 1.533*u.um:
        R_int = 1.25
    elif lambda1 == 1.599*u.um:
        R_int = 3.6

    # Apply extinction formula
    Av = (-2.5 * np.log10(R_obs / R_int)) / (beta_lambda1 - beta_lambda2)
        
    # Uncertainty
    dF1 = np.sqrt(error1 ** 2 + (0.05 * Fobs1) ** 2)
    dF2 = np.sqrt(error2 ** 2 + (0.05 * Fobs2) ** 2)
    
    # SNR mask
    snr1 = Fobs1 / dF1
    snr2 = Fobs2 / dF2
    # print(f"SNR1: {snr1:.2f}, SNR2: {snr2:.2f}")
    if snr1 < 2 or snr2 < 2:
        return np.nan, np.nan
    
    # Error propagation in observed ratio 
    dR_obs = R_obs * np.sqrt((dF1/Fobs1)**2 + (dF2/Fobs2)**2)

    # Error propagation in AV
    dAv = (2.5 / np.log(10)) * dR_obs / (R_obs * (beta_lambda1 - beta_lambda2)) # IS THIS FORMULA RIGHT???
    # print(f"AV = {Av:.2f} ± {dAv:.2f} mag")
    
    return float(Av), dAv

centers_x = [25, 26, 27]
centers_y = [33, 38, 44]

# Avs = [calculate_Av('fits/4s3d.fits', 1.257*u.um, 1.644*u.um, centers_x[i], centers_y[i]) for i in range(3)]
# print([round(Av[0], 2) for Av in Avs])
# print([round(Av[1], 2) for Av in Avs])

# Av2 = calculate_Av('fits/4s3d.fits', 1.321*u.um, 1.644*u.um)
# Av3 = calculate_Av('fits/4s3d.fits', 1.533*u.um, 1.677*u.um)
# Av4 = calculate_Av('fits/4s3d.fits', 1.599*u.um, 1.712*u.um)

def plot_Avs(fits_file):
    cube = SpectralCube.read('fits/subtracted_cube_full4.fits')
    error_cube = get_uJy_error_cube(fits_file)
    ra = 69.896675   * u.degree
    dec = 25.69561666667 * u.degree
    x1, y1 = get_pixel(cube, ra, dec) # starting point
    x1 = 24
    y1 = 26 # THESE ARE NOT REAL
    # print(x1, y1)
    
    x2 = 29
    y2 = 51
    rr, cc = line(y1, x1, y2, x2) # y, then x
    
    lines = [(1.257,1.644), (1.321,1.644), (1.533,1.677), (1.599,1.712)] * u.um
    colors = ['black', 'red', 'green', 'purple']
    labels = [f'{lines[i][0]}/{lines[i][1]}' for i in range(len(lines))]
    distances = np.sqrt((cc - x1)**2 + (rr - y1)**2)

    plt.figure(figsize=(8, 5))

    # Iterate over all line combinations
    for i in range(len(lines)):
        Avs = [calculate_Av(fits_file, cube, error_cube, lines[i][0], lines[i][1], rr[k], cc[k]) for k in range(len(rr))] # (Av_value, Av_error) tuples in a list
        # Plot Avs at each distance
        for k in range(len(rr)):
            if k == 0: # add just one label for each color
                plt.errorbar(distances[k], Avs[k][0], yerr=Avs[k][1], fmt='o', color=colors[i], label=labels[i])
            else:
                plt.errorbar(distances[k], Avs[k][0], yerr=Avs[k][1], fmt='o', color=colors[i])
    
    plt.xlabel('Distance from star (pixels)', fontsize=14)
    plt.ylabel('Visual Extinction $A_V$ (mag)', fontsize=14)
    plt.title('Extinction $A_V$ Along Jet Axis from Different Line Ratios', fontsize=16)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_Avs('fits/4s3d.fits')



