
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.units.quantity import Quantity

import imp
hexitec_pile_up = imp.load_source("hexitec_pile_up", "../hexitec_pile_up.py")

FRAME_RATE = Quantity(1e4, unit=1/u.s)

def test_generate_random_photons_from_spectrum():
    """Tests generate_random_counts_from_spectrum()."""
    # Define bins for spectrum.
    n_bins = 1000
    lower_end = 1.
    upper_end = 100.
    bin_width = (upper_end-lower_end)/n_bins
    lower_bin_edges = Quantity(np.linspace(lower_end, upper_end-bin_width, n_bins), unit=u.keV)
    upper_bin_edges = Quantity(np.linspace(lower_end+bin_width, upper_end, n_bins), unit=u.keV)
    # Use thermal thin target bremsstrahlung model to create spectrum
    kB = Quantity(8.6173324e-8, unit=u.keV/u.K)  # Boltzmann constant
    T = Quantity(1.e7, unit=u.K) # Temperature
    EM = Quantity(1.e49, unit=u.cm**-3) # Emission measure
    a = Quantity(8.1e-39, unit=u.cm/u.s*u.K**0.5) # proportionality constant at 1AU
    expected_counts = a*EM*T**(-0.5)*np.exp(-upper_bin_edges/(kB*T))
    expected_spectrum = Table([lower_bin_edges, upper_bin_edges, expected_counts],
                              names=("lower_bin_edges", "upper_bin_edges", "counts"))
    # Use generate_random_counts_from_spectrum() to sample count list.
    n_counts = 1000
    hpu = hexitec_pile_up.HexitecPileUp(frame_rate=FRAME_RATE)
    test_photons = hpu.generate_random_photons_from_spectrum(expected_spectrum, Quantity(1e-3, unit=1/u.s), n_counts)
    # Turn list of counts into a spectrum.
    bins = list(lower_bin_edges.value)
    bins.append(upper_bin_edges.value[-1])
    test_counts = np.histogram(test_photons["energy"], bins=bins)    
    # Assert where test_spectrum has significant number of counts, that
    # they are approximately equal to true_spectrums when scaled to n_counts.
    w = np.where(test_counts > 10.)[0]
    np.testing.assert_allclose(expected_counts[w]/(expected_counts[0]/test_counts[0]),
                               test_counts[w], rtol=0.01)

def test_simulate_hexitec_on_photon_list_1pixel():
    """Test simulate_masking_photon_list_1pixel()."""
    # Define a HexitecPileUp object.
    hpu = hexitec_pile_up.HexitecPileUp(frame_rate=FRAME_RATE)
    # Define input photon list and waiting times.
    photon_energies = Quantity([1, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10], unit=u.keV)
    photon_times = Quantity(np.array(
        [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 4.5, 6., 7.5, 7.5, 10000.5])*hpu.frame_duration,
        unit=u.s)
    incident_photons = Table([photon_times, photon_energies], names=("time", "energy"))
    # Define expected output photon list.
    f = hpu._voltage_pulse_shape[
        np.where(hpu._voltage_pulse_shape == max(hpu._voltage_pulse_shape))[0][0]-1]
    expected_energies = Quantity([1, 3, 5, 6, 7*f, 7, 17, 10], unit=u.keV)
    expected_times = Quantity(np.array([0., 1., 2., 4., 5., 6., 7., 10000])*hpu.frame_duration, unit=u.s)
    expected_photons = Table([expected_times, expected_energies], names=("time", "energy"))
    # Calculate test measured photon list by calling
    # simulate_hexitec_photon_list_1pixel().
    hpu.simulate_hexitec_on_photon_list_1pixel(incident_photons)
    # Assert test photon list is the same as expected photon list.
    assert all(hpu.measured_photons["time"] == expected_photons["time"])
    assert all(hpu.measured_photons["energy"] == expected_photons["energy"])
