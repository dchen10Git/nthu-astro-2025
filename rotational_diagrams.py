from pdrtpy.measurement import Measurement
from pdrtpy.tool.h2excitation import H2ExcitationFit
from pdrtpy.plot.excitationplot import ExcitationPlot
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dust_extinction.parameter_averages import G23
from astropy.units.quantity import Quantity
from helpers import pixar_sr, cdelt3

# # Replace with your actual flux and error values
# lines = []
# fluxes = np.array([1.2e-15, 3.5e-15, 2.1e-15, 2.8e-15, 1.5e-15, 1.1e-15, 0.8e-15, 0.5e-15])  # erg/s/cm^2
# errors = np.array([0.2e-15] * len(lines))  # uncertainties

# # Filter for v=0-0 transitions of H2
# filtered = [(line, f, e) for line, f, e in zip(lines, fluxes, errors) if re.search(r'H2.*v=0-0', line)]
# if not filtered:
#     raise ValueError("No v=0-0 transitions found in your line list!")

# lines, fluxes, errors = zip(*filtered)
fits_file = '../fits/6s3d.fits'
pixar = pixar_sr(fits_file).value
c = 1 / (pixar) / (10e9) # NOTE is this right???

cdelt = cdelt3(fits_file)

# Convert cdelt3 (wavelength delta) to Hertz using Doppler formula
# Well, do the integration (do moment 0)
# dv/c = d(nu)/nu = dλ/λ

intensity = dict()
intensity['H200S8'] = 76.16743412 * c
intensity['H200S9'] = 121.11178675 * c
intensity['H200S10'] = 31.6649388 * c
intensity['H200S11'] = 50.8594412 * c
# 12 is missing due to IFU gap
intensity['H200S13'] = 20.69110752 * c

a = []
for i in intensity:
    # For this example, set a largish uncertainty on the intensity.
    m = Measurement(data=intensity[i],uncertainty=StdDevUncertainty(intensity[i]),
                    identifier=i,unit="erg cm-2 s-1 sr-1")
    print(m)
    a.append(m)
    
h = H2ExcitationFit(a)# Use pdrtpy to compute column densities

print(h.column_densities(line=False, norm=False))

hplot = ExcitationPlot(h,"H_2")
h.run()

# make a plot showing the fit
hplot.ex_diagram(show_fit=True, xmin=5000, xmax=20000, ymin=20, ymax=60)

plt.show()

