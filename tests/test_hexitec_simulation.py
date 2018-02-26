
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.units.quantity import Quantity

import imp
#hexitec_simulation = imp.load_source("hexitec_simulation", "../hexitec_simulation.py")
hexitec_simulation = imp.load_source("hexitec_simulation", "hexitec_simulation.py")

FRAME_RATE = Quantity(3000, unit=1/u.s)

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
    hpu = hexitec_simulation.HexitecSimulation(
        frame_rate=FRAME_RATE, detector_temperature=17*u.deg_C, bias_voltage=-500*u.V)
    test_photons = hpu.generate_random_photons_from_spectrum(
        expected_spectrum, Quantity(1e-3, unit=1/u.s), n_counts)
    # Turn list of counts into a spectrum.
    bins = list(lower_bin_edges.value)
    bins.append(upper_bin_edges.value[-1])
    test_counts = np.histogram(test_photons["energy"], bins=bins)[0]
    # Assert where test_spectrum has significant number of counts, that
    # they are approximately equal to true_spectrums when scaled to n_counts.
    w = np.where(test_counts > 10.)[0]
    np.testing.assert_allclose((expected_counts[w]/(expected_counts[0]/test_counts[0])).value,
                               test_counts[w], rtol=0.1)

def test_simulate_hexitec_on_photon_list_1pixel():
    """Test simulate_masking_photon_list_1pixel()."""
    # Define a HexitecPileUp object.
    hpu = hexitec_simulation.HexitecSimulation(
        frame_rate=FRAME_RATE, detector_temperature=17*u.deg_C, bias_voltage=-500*u.V)
    # Define input photon list and waiting times.
    photon_energies = Quantity([1, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10], unit=u.keV)
    photon_times = Quantity(np.array(
        [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 4.5, 6., 7.5, 7.5,
         (hexitec_simulation.DATAFRAME_MAX_POINTS*hpu._sample_step/hpu.frame_duration).si.value+0.5
        ])*hpu.frame_duration, unit=u.s)
    incident_photons = Table([photon_times, photon_energies], names=("time", "energy"))
    # Define expected output photon list.
    f = hpu._voltage_pulse_shape[
        np.where(hpu._voltage_pulse_shape == max(hpu._voltage_pulse_shape))[0][0]-1]
    expected_energies = Quantity([1, 3, 5, 6, 7*f, 7, 17, 10], unit=u.keV)
    expected_times = np.array(
        [0., 1., 2., 4., 5., 6., 7.,
         (hexitec_simulation.DATAFRAME_MAX_POINTS*hpu._sample_step/hpu.frame_duration).si.value]
        )*hpu.frame_duration
    expected_next_frame_energy = Quantity([0., 0., 0., 0., 7., 0., 0., 0.], unit=u.keV)
    expected_photons = Table([expected_times, expected_energies, expected_next_frame_energy],
                             names=("time", "energy", "next frame first energy"))
    # Calculate test measured photon list by calling
    # simulate_hexitec_photon_list_1pixel().
    hpu.measured_photons = hpu.simulate_hexitec_on_photon_list_1pixel(incident_photons)
    # Assert test photon list is the same as expected photon list.
    np.testing.assert_allclose(hpu.measured_photons["time"], expected_photons["time"],
                               atol=hpu.frame_duration.to(hpu.measured_photons["time"].unit).value)
    assert all(hpu.measured_photons["energy"] == expected_photons["energy"])
    assert all(hpu.measured_photons["next frame first energy"] == \
               expected_photons["next frame first energy"])

def test_account_for_charge_sharing_in_photon_list():
    """Test account_for_charge_sharing_in_photon_list()."""
    # Create object
    hpu = hexitec_simulation.HexitecSimulation()
    # Define test incident photon list.
    times = Quantity([1, 2, 2, 3, 4, 4], unit=u.s)
    pixel_coords = np.array([0, 3, 12, 3, 3, 4])
    incident_photons = Table([times, [1]*len(times), pixel_coords+0.5, pixel_coords+0.5],
                             names=("time", "energy", "x", "y"))
    # Define expected photon list.
    n_neighbours = 9
    expected_times = np.array([1]*4+[2]*n_neighbours+[2]*n_neighbours+[3]* \
                              n_neighbours+[4]*n_neighbours+[4]*n_neighbours, dtype=float)
    central_distribution = list(hpu._divide_charge_among_pixels(
        1.5, 1.5, hpu._charge_cloud_x_sigma, hpu._charge_cloud_y_sigma, hpu._n_1d_neighbours)[2])
    expected_energies = np.array(central_distribution[4:6] + central_distribution[7:9] + \
                                 central_distribution*5)
    expected_x = [0, 1, 0, 1] + [coord for sublist in [
        [pixel_coord-1, pixel_coord, pixel_coord+1]*3 for pixel_coord in pixel_coords[1:]]
        for coord in sublist]
    expected_y = [0, 0, 1, 1] + [coord for sublist in [
        [pixel_coord-1]*3 + [pixel_coord]*3 + [pixel_coord+1]*3
        for pixel_coord in pixel_coords[1:]] for coord in sublist]
    expected_photons = Table([expected_times, expected_energies, expected_x, expected_y],
                             names=("time", "energy", "x_pixel", "y_pixel"))
    # Run account_for_charge_sharing_in_photon_list().
    test_photons = hpu.account_for_charge_sharing_in_photon_list(
        incident_photons, hpu._charge_cloud_x_sigma, hpu._charge_cloud_y_sigma,
        hpu._n_1d_neighbours)
    # Assert expected photons equal test photons.
    assert expected_photons.colnames == test_photons.colnames
    assert all(expected_photons["time"] == test_photons["time"])
    assert all(expected_photons["energy"] == test_photons["energy"])
    assert all(expected_photons["x_pixel"] == test_photons["x_pixel"])
    assert all(expected_photons["y_pixel"] == test_photons["y_pixel"])

def test_divide_charge_among_pixels():
    """Test _divide_charge_among_pixels()."""
    x = y = 1.5
    x_sigma = y_sigma = 0.5
    n_1d_neighbours = 3
    hpu = hexitec_simulation.HexitecSimulation()
    # Define expected fractional energy.
    expected_x_pixels = array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    expected_y_pixels = array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    expected_fractional_energy = array([ 0.02474497, 0.10739071, 0.02474497,
                                         0.10739071, 0.46606494, 0.10739071,
                                         0.02474497, 0.10739071, 0.02474497])
    # Run divide_charge_among_pixels().
    test_x_pixels, test_y_pixels, test_fractional_energy = \
      hpu._divide_charge_among_pixels(x, y, x_sigma, y_sigma, n_1d_neighbours)
    # Assert the expected value equal returned values.
    assert all(expected_x_pixels == test_x_pixels)
    assert all(expected_y_pixels == test_y_pixels)
    assert all(expected_x_fractional_energy == test_fractional_energy)
