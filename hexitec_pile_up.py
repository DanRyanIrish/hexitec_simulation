"""This module contains a class for simulating the effect of pile up in HEXITEC."""
import random
import warnings
import math

import numpy as np
from numpy import ma
import astropy.units as u
from astropy.table import Table
from astropy.units.quantity import Quantity
import pandas

import timeit

HEXITEC_FRAME_DURATION = Quantity(1e-4, unit=u.s)

class HexitecPileUp():
    """Determines effect of pile up on an input spectrum as observed by HEXITEC."""

    def __init__(self):
        """Instantiates a HexitecPileUp object."""
        self.frame_duration = Quantity(1e-4, unit=u.s)

    def simulate_masking_on_spectrum_1pixel(self, incident_spectrum, photon_rate, n_photons):
        """
        Simulates "masking" in a single HEXITEC pixel on a binned incident photon spectrum.

        This simulation is a 1st order approximation of the effect of pile up.
        It assumes than only the most energetic photon incident on the detector
        within the period of a single frame is recorded.

        Parameters
        ----------
        incident_spectrum : `astropy.table.Table`
          Incident photon spectrum.  Table has following columns:
            lower_bin_edges : `astropy.units.quantity.Quantity`
            upper_bin_edges : `astropy.units.quantity.Quantity`
            counts : array-like
        photon_rate : `astropy.units.quantity.Quantity`
          The average photon rate.
        n_photons : `int`
          Number of counts to simulate hitting the detector.  Note that the first
          count will not be included in output spectrum as there is no information
          on the waiting time before it.

        Returns
        -------
        measured_spectrum : `astropy.table.Table`
          Measured spectrum taking photon masking of HEXITEC pixel into account.

        """
        self.incident_spectrum = incident_spectrum
        if type(photon_rate) is not Quantity:
            raise TypeError("photon_rate must be an astropy.units.quantity.Quantity")
        self.photon_rate = photon_rate
        # Generate random photon energies from incident spectrum to
        # enter detector.  Result recorded in self.incident_photons.
        self.generate_random_photons_from_spectrum(n_photons)
        # Generate random waiting times between incident photons.
        # Model as a Poisson process, i.e. draw from an exponential
        # distrubtion.
        self.photon_waiting_times = Quantity(np.random.exponential(1./photon_rate, n_photons),
                                             unit='s')
        # Mark photons which were recorded and unrecorded using a
        # masked array.  Result recorded in self.measured_photons.
        self.simulate_masking_on_photon_list_1pixel()
        # Convert measured photon list into counts in same bins as the
        # incident spectrum.
        print "Converting masked photon list into spectrum."
        time1 = timeit.default_timer()
        #measured_counts = np.histogram(self.measured_photons.compressed(), bins=list(
        #    self.incident_spectrum["lower_bin_edges"]).append(
        #        self.incident_spectrum["upper_bin_edges"][-1]))[0]
        bins = list(self.incident_spectrum["lower_bin_edges"])
        bins.append(self.incident_spectrum["upper_bin_edges"][-1])
        measured_counts = np.histogram(self.measured_photons.compressed(), bins=bins)[0]
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Return an astropy table of the measured spectrum.
        self.measured_spectrum = Table(
            [self.incident_spectrum["lower_bin_edges"],
             self.incident_spectrum["upper_bin_edges"], measured_counts],
            names=("lower_bin_edges", "upper_bin_edges", "counts"),
            meta={"info" : "Counts measured by HEXITEC after accounting for masking."})


    def simulate_masking_on_photon_list_1pixel(self):
        """
        Simulates "masking" effect in a single HEXITEC pixel on an incident photon list.

        This simulation is a 1st order approximation of the effect of pile up.
        It assumes than only the most energetic photon incident on the
        detector within the period of a single frame is recorded.
    
        Parameters
        ----------
        self.incident_photons : `astropy.units.quantity.Quantity`
          Array of each sequential photon falling of the HEXITEC pixel.
        self.photon_waiting_times : `astropy.units.quantity.Quantity`
          The time between each photon hit.  Note must therefore have length
          1 less than incident_photons.
        first_photon_offset : `astropy.units.quantity.Quantity`
          Delay from start of first observing frame of HEXITEC detector until
          first photon hit.  Default=0s.

        Returns
        -------
        self.measured_photons : masked_Quantity
          Incident photon list with unrecorded photons masked.

        """
        # Determine time of each photon hit from start of observing time.
        photon_times = self.photon_waiting_times.cumsum()
        # Determine length of time from start of observation to time of
        # final photon hit.
        total_observing_time = photon_times[-1]
        # Determine number of frames in observing times by rounding up.
        n_frames = int((total_observing_time/self.frame_duration).si+1)
        # Assign photons to HEXITEC frames.
        n_photons = len(self.incident_photons)
        photon_indices = np.arange(n_photons)
        print "Assigning photons to frames."
        time1 = timeit.default_timer()
        photon_indices_in_frames = (photon_indices[np.logical_and(
            photon_times >= self.frame_duration*i,
            photon_times < self.frame_duration*(i+1))] for i in range(n_frames))
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Create array of measured photons by masking photons from
        # incident photons.
        print "Masking photons."
        time1 = timeit.default_timer()
        unmask_indices = [frame[np.argmax(self.incident_photons[frame])]
                          for frame in photon_indices_in_frames if len(frame) > 0]
        self.measured_photons = ma.masked_array(self.incident_photons, mask=[1]*n_photons)
        self.measured_photons.mask[unmask_indices] = 0
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)


    def simulate_masking_on_photon_list_1pixel_lc(self):
        """
        Simulates "masking" effect in a single HEXITEC pixel on an incident photon list.

        This simulation is a 1st order approximation of the effect of pile up.
        It assumes than only the most energetic photon incident on the
        detector within the period of a single frame is recorded.

        """
        # Generate time series of HEXITEC voltage vs. time from
        # photon energies and waiting times.
        timeseries = self._convert_photons_to_timeseries()
        # Convert timeseries into measured photon list by resampling
        # at frame rate and taking max.
        # In resample command, '100U' signifies 100 microsecs.
        frame_peaks = timeseries.resample("100U", how=max)
        threshold = 0.
        w = np.where(frame_peaks["Voltage"] > threshold)[0]
        # Determine time unit of pandas timeseries and convert photon
        # times to Quantity.
        tdunit = timeseries.index.dtype_str.split(("["))[-1].split(']')[0]
        if tdunit in ["s", "ms", "us", "ns", "ps", "fs"]:
            photon_times_quantity = Quantity(frame_peaks.index[w].values, unit=tdunit).to("s")
        else:
            raise TypeError("Unrecognised timedelta64 unit.")
        # Combine photon times and energies into measured photon list.
        self.measured_photons = Table([
            photon_times, Quantity(frame_peaks["Voltage"][w], unit=self.incident_photons.unit)],
            names=("Time", "Energy"))


    def generate_random_photons_from_spectrum(self, n_counts):
        """Converts an input photon spectrum to a probability distribution.

        Parameters
        ----------
        self.incident_spectrum["counts"] : array-like
          Counts in each bin of spectrum.
        self.incident_spectrum["lower_bin_edges"] : array-like
          Lower edge of each bin of spectrum.
        self.incident_spectrum["upper_bin_edges"] : array_like
          Upper edge of each bin of spectrum.
        n_counts : `int`
          Total number of random counts to be generated.
      
        """
        n_counts = int(n_counts)
        # Calculate cumulative density function of spectrum for lower and
        # upper edges of spectral bins.
        cdf_upper = np.cumsum(self.incident_spectrum["counts"])
        cdf_lower = np.insert(cdf_upper, 0, 0.)
        cdf_lower = np.delete(cdf_lower, -1)
        # Generate random numbers representing CDF values.
        print "Generating random numbers for photon energy transformation."
        time1 = timeit.default_timer()
        randoms = np.asarray([random.random() for i in range(n_counts)])*cdf_upper[-1]
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Generate array of spectrum bin indices.
        print "Transforming random numbers into photon energies."
        time1 = timeit.default_timer()
        bin_indices = np.arange(len(self.incident_spectrum["lower_bin_edges"]))
        # Generate random counts from randomly generated CDF values.
        #random_photons = Quantity([lower_bin_edges.value[
        self.incident_photons = Quantity([self.incident_spectrum["lower_bin_edges"].data[
            bin_indices[np.logical_and(r >= cdf_lower, r < cdf_upper)][0]]
            for r in randoms], unit=self.incident_spectrum["lower_bin_edges"].unit)
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)


     def _convert_photons_to_timeseries(self, samples_per_frame=100):
        """Create a time series of photon energies with a given sampling frequency."""
        sample_unit = 'ns'
        photon_times = self.photon_waiting_times.to(sample_unit).value.cumsum()
        total_observing_time = photon_times[-1]
        n_frames = int(total_observing_time/self.frame_duration.to(sample_unit).value+1)
        n_samples = n_frames*samples_per_frame
        # Define timestamps for times series.
        timestamps = np.linspace(0, n_frames*self.frame_duration.to(sample_unit).value, n_samples+1)
        timestamps = timestamps[:-1]
        # Find indices in timeseries closest to photon times.
        photon_time_indices = [np.argmin(abs(timestamps-photon_time)) for photon_time in photon_times]
        # Insert photons into timeseries.
        data = np.zeros(n_samples)
        data[photon_time_indices] = self.incident_photons.value
        timeseries = pandas.DataFrame(data, index=pandas.to_timedelta(timestamps, sample_unit),
                                      columns=["Voltage"])
        return timeseries


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
    # Use generate_random_counts_from_spectrum() to sample count list
    n_counts = 1000000
    #test_photons = generate_random_photons_from_spectrum(lower_bin_edges, upper_bin_edges,
    #                                                     expected_counts.value, n_counts)
    #test_photons = generate_random_photons_from_spectrum(expected_spectrum, n_counts)
    hpu = HexitecPileUp()
    hpu.incident_spectrum = expected_spectrum
    hpu.generate_random_photons_from_spectrum(n_counts)
    # Turn list of counts into a spectrum.
    bins = list(lower_bin_edges.value)
    bins.append(upper_bin_edges.value[-1])
    test_counts = np.histogram(hpu.incident_photons, bins=bins)    
    # Assert where test_spectrum has significant number of counts, that
    # they are approximately equal to true_spectrums when scaled to n_counts.
    w = np.where(test_counts > 10.)[0]
    np.testing.assert_allclose(expected_counts[w]/(expected_counts[0]/test_counts[0]), test_counts[w], rtol=0.01)

def test_simulate_masking_photon_list_1pixel():
    """Test simulate_masking_photon_list_1pixel()."""
    # Define a HexitecPileUp object.
    hpu = HexitecPileUp()
    # Define input photon list and waiting times.
    incident_photons = Quantity([1, 1, 2, 3, 5, 4, 6], unit=u.keV)
    first_photon_offset = Quantity([0.], unit=u.s)
    photon_waiting_times = first_photon_offset + Quantity(
        np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 2.5])*hpu.frame_duration, unit=u.s)
    # Define expected output photon list.
    expected_photons = np.ma.masked_array(incident_photons, mask=[0,1,1,0,0,1,0])
    # Calculate test measured photon list by calling
    # simulate_masking_photon_list_1pixel().
    hpu.first_photon_offset = first_photon_offset
    hpu.incident_photons = incident_photons
    hpu.photon_waiting_times = photon_waiting_times
    hpu.simulate_masking_on_photon_list_1pixel()
    # Assert test photon list is the same as expected photon list.
    np.testing.assert_array_equal(expected_photons.data, hpu.measured_photons.data)
    np.testing.assert_array_equal(expected_photons.mask, hpu.measured_photons.mask)
    assert expected_photons.data.unit == hpu.measured_photons.data.unit


    def test_simulate_masking_photon_list_1pixel_lc():
    """Test simulate_masking_photon_list_1pixel()."""
    # Define a HexitecPileUp object.
    hpu = HexitecPileUp()
    # Define input photon list and waiting times.
    incident_photons = Quantity([1, 1, 2, 3, 5, 4, 6], unit=u.keV)
    photon_waiting_times = Quantity(
        np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 2.5])*hpu.frame_duration, unit=u.s)
    # Define expected output photon list.
    expected_indices = [0, 3, 4, 6]
    expected_photons = Table([photon_waiting_times.cumsum()[expected_indices],
                              incident_photons[expected_indices]], names=("Time", "Energy"))
    # Calculate test measured photon list by calling
    # simulate_masking_photon_list_1pixel().
    hpu.incident_photons = incident_photons
    hpu.photon_waiting_times = photon_waiting_times
    hpu.simulate_masking_on_photon_list_1pixel_lc()
    # Assert test photon list is the same as expected photon list.
    assert all(measured_photons["Energy"] == expected_photons["Energy"])
    assert 0.*u.s <= all(expected_photons["Time"]-measured_photons["Time"]) < hpu.frame_duration
