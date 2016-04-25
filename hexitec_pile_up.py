"""This module contains a class for simulating the effect of pile up in HEXITEC."""
import random
import warnings
import math

import numpy as np
from numpy import ma
import astropy.units as u
from astropy.table import Table
from astropy.units.quantity import Quantity

HEXITEC_FRAME_DURATION = Quantity(1e-4, unit=u.s)

class HexitecPileUp():
    """Determines effect of pile up on an input spectrum as observed by HEXITEC."""

    def __init__(self, incident_spectrum):
        """Instantiates a HexitecPileUp object."""
        self.hexitec_frame_duration = Quantity(1e-4, unit=u.s)
        self.incident_spectrum = incident_spectrum

def simulate_masking_on_spectrum_1pixel(incident_spectrum, n_photons, photon_rate):
    """
    Simulates "masking" effect in a single HEXITEC pixel on a binned incident photon spectrum.

    This simulation is a 1st order approximation of the effect of pile up.  It assumes than only
    the most energetic photon incident on the detector within the period of a single frame is
    recorded.

    Parameters
    ----------
    incident_spectrum : `astropy.table.Table`
      Incident photon spectrum.  Table has following columns:
        lower_bin_edges : `astropy.units.quantity.Quantity`
        upper_bin_edges : `astropy.units.quantity.Quantity`
        counts : array-like
    n_photons : `int`
      Number of counts to simulate hitting the detector.  Note that the first
      count will not be included in output spectrum as there is no information
      on the waiting time before it.
    photon_rate : `astropy.units.quantity.Quantity`
      The average photon rate.

    Returns
    -------
    measured_spectrum : `astropy.table.Table`
      Measured spectrum taking photon masking of HEXITEC pixel into account.

    """
    # Generate random photon energies from incident spectrum to enter detector.
    incident_photons = generate_random_photons_from_spectrum(
        incident_spectrum["lower_bin_edges"], incident_spectrum["upper_bin_edges"],
        incident_spectrum["counts"], n_photons)
    # Generate random waiting times between incident photons.
    photon_waiting_times = generate_poisson_waiting_times(n_photons-1, photon_rate)
    # Mark photons which were recorded and unrecorded using a masked array.
    measured_photons = simulate_masking_photon_list_1pixel(incident_photons, photon_waiting_times)
    # Convert measured photon list into counts in same bins as the
    # incident spectrum.
    measured_counts = np.histogram(measured_photons.compressed(), bins=list(
        incident_spectrum["lower_bin_edges"]).append(
            incident_spectrum["upper_bin_edges"][-1]))[0]
    # Return an astropy table of the measured spectrum.
    measured_spectrum = Table(
        [incident_spectrum["lower_bin_edges"], incident_spectrum["upper_bin_edges"],
         measured_counts], names=("lower_bin_edges", "upper_bin_edges", "counts"),
        meta={"info" : "Counts measured by HEXITEC after accounting for masking."})
    return measured_spectrum


def simulate_masking_photon_list_1pixel(incident_photons, photon_waiting_times,
                                        first_photon_offset=Quantity(0., unit=u.s)):
    """
    Simulates "masking" effect in a single HEXITEC pixel on an incident photon list.

    This simulation is a 1st order approximation of the effect of pile up.  It
    assumes than only the most energetic photon incident on the detector within
    the period of a single frame is recorded.
    
    Parameters
    ----------
    incident_photons : `astropy.units.quantity.Quantity`
      Array of each sequential photon falling of the HEXITEC pixel.
    photon_waiting_times : `astropy.units.quantity.Quantity`
      The time between each photon hit.  Note must therefore have length
      1 less than incident_photons.
    first_photon_offset : `astropy.units.quantity.Quantity`
      Delay from start of first observing frame of HEXITEC detector until
      first photon hit.  Default=0s.

    Returns
    -------
    measured_photons : masked_Quantity
      Incident photon list with unrecorded photons masked.

    """
    # Determine time of each photon hit from start of observing time.
    photon_times = first_photon_offset+photon_waiting_times.cumsum()
    # Determine length of time from start of observation to time of
    # final photon hit.
    total_observing_time = photon_times[-1]
    # Determine number of frames in observing times by rounding up.
    n_frames = int(total_observing_time/HEXITEC_FRAME_DURATION+1)
    # Assign photons to HEXITEC frames.
    n_photons = len(incident_photons)
    photon_indices = np.arange(n_photons)
    photon_indices_in_frames = [photon_indices[np.logical_and(
        photon_times >= HEXITEC_FRAME_DURATION*i,
        photon_times < HEXITEC_FRAME_DURATION*(i+1))] for i in np.arange(n_frames)]
    # Create array of measured photons by masking photons from
    # incident photons.
    measured_photons = ma.masked_array(incident_photons, mask=[1]*n_photons)
    unmask_indices = [frame[np.argmax(incident_photons[frame])]
                      for frame in photon_indices_in_frames if len(frame) > 0]
    measured_photons.mask[unmask_indices] = False
    return measured_photons


def generate_random_photons_from_spectrum(lower_bin_edges, upper_bin_edges, counts, n_counts):
    """Converts an input photon spectrum to a probability distribution.

    Parameters
    ----------
    counts : array-like
      Counts in each bin of spectrum.
     : array-like
      Lower edge of each bin of spectrum.
    upper_bin_edges : array_like
      Upper edge of each bin of spectrum.
    n_counts : `float` or `int`
      Total number of random counts to be generated.
      
    """
    # Calculate cumulative density function of spectrum for lower and
    # upper edges of spectral bins.
    cdf_upper = np.cumsum(counts)
    cdf_lower = np.insert(cdf_upper, 0, 0.)
    cdf_lower = np.delete(cdf_lower, -1)
    # Generate random numbers representing CDF values.
    randoms = np.asarray([random.random() for i in range(n_counts)])*cdf_upper[-1]
    # Generate array of spectrum bin indices.
    bin_indices = np.arange(len(lower_bin_edges))
    # Generate random counts from randomly generated CDF values.
    random_photons = Quantity([lower_bin_edges.value[
        bin_indices[np.logical_and(r >= cdf_lower, r < cdf_upper)][0]]
        for r in randoms], unit=lower_bin_edges.unit)
    return random_photons

def generate_poisson_waiting_times(n_times, rate, xmin=Quantity(0, unit=u.s)):
    """Generates waiting times between events for a given number of events.

    It is assumed that the events are produced by a Poisson process, i.e. they
    are statistically indepedent.  Therefore the waiting times distribution is
    exponential.

    Parameters
    ----------
    n_events : `int`
      Number of waiting times to be generated.
    rate : `astropy.units.quantity.Quantity`
      The average rate at which events are known to occur.
    xmin : `astropy.units.quantity.Quantity`
      minimum desired waiting time for any given event.
      Default = 0s

    Returns
    -------
    waiting_times : `astropy.units.quantity.Quantity`
      Time between each consecutive event.

    """
    # If rate and xmin inuts aren't astropy Quantities, raise TypeError.
    if type(rate) is not astropy.units.quantity.Quantity:
        raise TypeError("rate must be an astropy.units.quantity.Quantity")
    if type(xmin) is not astropy.units.quantity.Quantity:
        raise TypeError("rate must be an astropy.units.quantity.Quantity")
    # Generate random numbers for selecting random waiting times.
    randoms = np.asarray([random.random() for i in range(n_times)], dtype=float)
    # Convert random numbers to waiting times by transforming through
    # the CDF of the exponential distribution.
    # Note that xmin unit is also changed to agree with rate unit.
    waiting_times = (xmin-(1./rate)*np.log(1-randoms)).to(rate.unit**-1)
    return waiting_times

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
    # Use generate_random_counts_from_spectrum() to sample count list
    n_counts = 1000000
    test_photons = generate_random_photons_from_spectrum(lower_bin_edges, upper_bin_edges,
                                                        expected_counts.value, n_counts)
    # Turn list of counts into a spectrum.
    bins = list(lower_bin_edges.value)
    bins.append(upper_bin_edges.value[-1])
    test_counts = np.histogram(test_photons, bins=bins)    
    # Assert where test_spectrum has significant number of counts, that
    # they are approximately equal to true_spectrums when scaled to n_counts.
    w = np.where(test_counts > 10.)[0]
    np.testing.assert_allclose(expected_counts[w]/(expected_counts[0]/test_counts[0]), test_counts[w], rtol=0.01)

def test_simulate_masking_photon_list_1pixel():
    """Test simulate_masking_photon_list_1pixel()."""
    # Define input photon list and waiting times.
    incident_photons = Quantity([1, 1, 2, 3, 5, 4, 6], unit=u.keV)
    first_photon_offset = Quantity([0.], unit=u.s)
    photon_waiting_times = first_photon_offset + Quantity(
        np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 2.5])*HEXITEC_FRAME_DURATION, unit=u.s)
    # Define expected output photon list.
    expected_photons = np.ma.masked_array(incident_photons, mask=[0,1,1,0,0,1,0])
    # Calculate test measured photon list by calling
    # simulate_masking_photon_list_1pixel().
    test_photons = simulate_masking_photon_list_1pixel(incident_photons, photon_waiting_times,
                                                       first_photon_offset=first_photon_offset)
    # Assert test photon list is the same as expected photon list.
    np.testing.assert_array_equal(expected_photons.data, test_photons.data)
    np.testing.assert_array_equal(expected_photons.mask, test_photons.mask)
    assert expected_photons.data.unit == test_photons.data.unit
