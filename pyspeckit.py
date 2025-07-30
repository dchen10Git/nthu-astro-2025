import numpy as np
from astropy import units as u
import pyspeckit

xaxis = np.linspace(-50,150,100) * u.km/u.s
sigma = 10. * u.km/u.s
center = 50. * u.km/u.s
synth_data = np.exp(-(xaxis-center)**2/(sigma**2 * 2.))

# Add noise
stddev = 0.1
noise = np.random.randn(xaxis.size)*stddev
error = stddev*np.ones_like(synth_data)
data = noise+synth_data

# this will give a "blank header" warning, which is fine
sp = pyspeckit.Spectrum(data=data, error=error, xarr=xaxis,
                        unit=u.erg/u.s/u.cm**2/u.AA)

sp.plotter()
sp.plotter.savefig('basic_plot_example.png')

# Fit with automatic guesses
sp.specfit(fittype='gaussian')
# (this will produce a plot overlay showing the fit curve and values)
sp.plotter.savefig('basic_plot_example_withfit.png')

# Redo the overlay with no annotation

# remove both the legend and the model overlay
sp.specfit.clear()
# then re-plot the model without an annotation (legend)
sp.specfit.plot_fit(annotate=False)
sp.plotter.savefig('basic_plot_example_withfit_no_annotation.png')