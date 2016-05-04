"""This module contains a class for simulating the effect of pile up in HEXITEC."""

import random
import warnings
import math
import timeit

import numpy as np
from numpy import ma
import astropy.units as u
from astropy.table import Table
from astropy.units.quantity import Quantity
import pandas

class HexitecPileUp():
    """Simulates how HEXITEC records incident photons."""

    def __init__(self):
        """Instantiates a HexitecPileUp object."""
        # Define some magic numbers. N.B. _sample_unit must be in string
        # format so it can be used for Quantities and numpy datetime64.
        self._sample_unit = 'ns'
        self._sample_step = Quantity(100., unit='ns')
        self.voltage_peaking_time = Quantity(2., unit='us').to(self._sample_unit)
        self.voltage_decay_time = Quantity(8., unit='us').to(self._sample_unit)
        self.frame_duration = Quantity(1e-4, unit=u.s)


    def simulate_hexitec_on_spectrum_1pixel(self, incident_spectrum, photon_rate, n_photons):
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
        self.measured_spectrum : `astropy.table.Table`
          Measured spectrum taking photon masking of HEXITEC pixel into account.
        self.measured_photons : `astropy.table.Table`
          See description of "self.measured_photons" in "Returns" section of
          docstring of generate_random_photons_from_spectrum().
        self.incident_photons : `astropy.table.Table`
          See description of "photons" in "Returns" section of docstring of
          generate_random_photons_from_spectrum().
        self.incident_spectrum : `astropy.table.Table`
          Same as input incident_spectrum.
        self.photon_rate : `astropy.units.quantity.Quantity`
          Same as input photon_rate.

        """
        # Generate random photon energies from incident spectrum to
        # enter detector.
        incident_photons = self.generate_random_photons_from_spectrum(
            incident_spectrum, photon_rate, n_photons)
        # Mark photons which were recorded and unrecorded using a
        # masked array.  Result recorded in self.measured_photons.
        self.simulate_hexitec_on_photon_list_1pixel(incident_photons)
        # Convert measured photon list into counts in same bins as the
        # incident spectrum.
        print "Converting masked photon list into spectrum."
        time1 = timeit.default_timer()
        bins = list(self.incident_spectrum["lower_bin_edges"])
        bins.append(self.incident_spectrum["upper_bin_edges"][-1])
        measured_counts = np.histogram(self.measured_photons["energy"], bins=bins)[0]
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Return an astropy table of the measured spectrum.
        self.measured_spectrum = Table(
            [self.incident_spectrum["lower_bin_edges"],
             self.incident_spectrum["upper_bin_edges"], measured_counts],
            names=("lower_bin_edges", "upper_bin_edges", "counts"))


    def simulate_hexitec_on_photon_list_1pixel(self, incident_photons):
        """
        Simulates how HEXITEC records incoming photons in a single pixel.

        Given a list of photons entering the pixel, this function simulates the
        voltage vs. time timeseries in the HEXITEC ASIC caused by the photon hits.
        From that, it then determines the measured photon list.  This simulation
        includes effects "masking" and "pulse pile up".  It assumes that the photon
        waiting times are distributed exponentially with an average rate of
        photon_rate.

        Parameters
        ----------
        incident_photons : `astropy.table.Table`
          Table of photons incident on the pixel.  Contains the following columns
          time : Amount of time passed from beginning of observing until photon hit.
          energy : Energy of each photon.

        Returns
        -------
        self.measured_photons : `astropy.table.Table`
          Photon list as measured by HEXITEC.  Table format same as incident_photons.
        self.incident_photons : `astropy.table.Table`
          Same as input incident_photons.

        """
        self.incident_photons = incident_photons
        # Generate time series of HEXITEC voltage vs. time from
        # photon energies and waiting times.
        timeseries = self._convert_photons_to_timeseries()
        # Convert timeseries into measured photon list by resampling
        # at frame rate and taking max.
        # In resample command, '100U' signifies 100 microsecs.
        print "Converting timeseries in measured photon list."
        time1 = timeit.default_timer()
        frame_peaks = timeseries.resample("100U", how=min)
        # Convert voltages back to photon energies
        threshold = 0.
        w = np.where(frame_peaks["voltage"] < threshold)[0]
        measured_photon_energies = \
          self._convert_voltages_to_photon_energy(frame_peaks["voltage"][w].values)
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Determine time unit of pandas timeseries and convert photon
        # times to Quantity.
        photon_times = Quantity(frame_peaks.index[w].values, unit=self._sample_unit).to("s")
        # Combine photon times and energies into measured photon list.
        self.measured_photons = Table([photon_times, measured_photon_energies],
                                      names=("time", "energy"))
        self.timeseries = timeseries


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
        photon_times = self.first_photon_offset+self.photon_waiting_times.cumsum()
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


    def generate_random_photons_from_spectrum(self, incident_spectrum, photon_rate, n_photons):
        """Converts an input photon spectrum to a probability distribution.

        Parameters
        ----------
        incident_spectrum : `astropy.table.Table`
          Incident photon spectrum.  Table has following columns:
            lower_bin_edges : `astropy.units.quantity.Quantity`
            upper_bin_edges : `astropy.units.quantity.Quantity`
            counts : array-like
        photon_rate : `astropy.units.quantity.Quantity`
          Average rate at which photons hit the pixel.
        n_photons : `int`
          Total number of random counts to be generated.

        Returns
        -------
        photons : `astropy.table.Table`
          Table of photons incident on the pixel.  Contains the following columns
          time : Amount of time passed from beginning of observing
            until photon hit.
          energy : Energy of each photon.
        self.incident_spectrum : `astropy.table.Table`
          Same as input incident_spectrum.
        self.photon_rate : `astropy.units.quantity.Quantity`
          Same as input photon_rate.
        
        """
        self.incident_spectrum = incident_spectrum
        if type(photon_rate) is not Quantity:
            raise TypeError("photon_rate must be an astropy.units.quantity.Quantity")
        self.photon_rate = photon_rate
        n_counts = int(n_photons)
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
        # Generate random energies from randomly generated CDF values.
        photon_energies = Quantity([self.incident_spectrum["lower_bin_edges"].data[
            bin_indices[np.logical_and(r >= cdf_lower, r < cdf_upper)][0]]
            for r in randoms], unit=self.incident_spectrum["lower_bin_edges"].unit)
        # Generate random waiting times before each photon.
        photon_waiting_times = Quantity(
            np.random.exponential(1./self.photon_rate, n_photons), unit='s')
        # Associate photon energies and time since start of
        # observation (time=0) in output table.
        photons = Table([photon_waiting_times.cumsum(), photon_energies],
                        names=("time", "energy"))
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        return photons


    def _convert_photons_to_timeseries(self):
        """
        Create a time series of photon energies with a given sampling frequency.

        Returns
        -------
        timeseries : `pandas.DataFrame`
          Time series of voltage vs. time inside HEXITEC ASIC due to photon hits.
        
        """
        # Ensure certains variables are in correct unit.
        sample_step = self._sample_step.to(self._sample_unit)
        frame_duration = self.frame_duration.to(self._sample_unit)
        # Calculate number of frames and sample points in timeseries.
        total_observing_time = (self.incident_photons["time"][-1]* \
                                self.incident_photons["time"].unit).to(self._sample_unit)
        n_frames = int(total_observing_time.value/frame_duration.value+2)
        # Define timestamps for timeseries from number of frames and
        # sample points.
        print "Generating timestamps."
        time1 = timeit.default_timer()
        timestamps = np.arange(0, n_frames*frame_duration.value, sample_step.value)
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Find indices in timeseries closest to photon times.
        print "Locating photon indices in timestamps."
        time1 = timeit.default_timer()
        photon_time_indices = np.rint(self.incident_photons["time"].data/sample_step.to(
            self.incident_photons["time"].unit).value).astype(int)
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        # Convert photons to HEXITEC voltage signals.
        voltage_pulses = self._convert_photon_energies_to_voltage_bigaussian(
            self.incident_photons["energy"])
        # Determine time indices of each value of voltage_hits.
        voltage_pulse_length = int(np.round(
            (self.voltage_peaking_time+self.voltage_decay_time)/sample_step))
        voltage_time_indices = np.array(
            [range(photon_time_index, photon_time_index+voltage_pulse_length)
             for photon_time_index in photon_time_indices])
        # Insert photons into timeseries.
        print "Generating time series."
        time1 = timeit.default_timer()
        data = np.zeros(len(timestamps))
        data[voltage_time_indices] = voltage_pulses.value
        timeseries = pandas.DataFrame(
            data, index=pandas.to_timedelta(timestamps, self._sample_unit), columns=["voltage"])
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        return timeseries


    def _convert_photon_energies_to_voltage_bigaussian(self, photon_energies):
        """Models pulse shape of HEXITEC voltage in response to a photon as a bi-gaussian.

        Parameters
        ----------
        photon_energies : array-like
          Energies of photons
        
        Returns
        -------
        voltage_pulses : `astropy.units.quantity.Quantity`
          Quantity of nested arrays.  Each nested array gives the voltage
          time series values corresponding to an individual photon hit.

        """
        sample_step = self._sample_step.to(self._sample_unit)
        # Convert input peaking and decay times to a standard unit.
        voltage_peaking_time = self.voltage_peaking_time.to(self._sample_unit)
        voltage_decay_time = self.voltage_decay_time.to(self._sample_unit)
        # Convert photon energy into peak voltage amplitude.
        a = self._convert_photon_energy_to_voltage(photon_energies)
        # Define other Gaussian parameters.
        mu = voltage_peaking_time
        zero_equivalent = Quantity(1e-3, unit="V")
        sigma2_peaks = -0.5*mu**2/np.log(zero_equivalent.value/abs(a.value))
        sigma2_decays = -0.5*(voltage_peaking_time+voltage_decay_time-mu)**2/ \
          np.log(zero_equivalent.value/abs(a.value))
        # Generate time data points for peak and decay phases.
        t_peaking = Quantity(np.arange(
            0, voltage_peaking_time.value, self._sample_step.value), unit=self._sample_unit)
        t_decay = Quantity(np.arange(voltage_peaking_time.value,
                                     voltage_peaking_time.value+voltage_decay_time.value,
                                     self._sample_step.value), unit=self._sample_unit)
        # Create Quantity holding voltage signal due to each photon hit.
        # Do this by appending peaking and decay signals in each case.
        voltage_pulses = Quantity([np.append(
            a[i].value*np.exp(-(t_peaking-mu).value**2./(2*sigma2_peaks[i].value)),
            a[i].value*np.exp(-(t_decay-mu).value**2./(2*sigma2_decays[i].value)))
            for i in range(len(a))], unit=a.unit)
        return voltage_pulses


    def _convert_photon_energy_to_voltage(self, photon_energy):
        """Determines HEXITEC peak voltage due photon of given energy."""
        return -photon_energy.data*u.V

    def _convert_voltages_to_photon_energy(self, voltages):
        """Determines photon energy from HEXITEC peak voltage."""
        return -voltages*u.keV



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
    np.testing.assert_allclose(expected_counts[w]/(expected_counts[0]/test_counts[0]),
                               test_counts[w], rtol=0.01)

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


def test_simulate_hexitec_on_photon_list_1pixel():
    """Test simulate_masking_photon_list_1pixel()."""
    # Define a HexitecPileUp object.
    hpu = HexitecPileUp()
    # Define input photon list and waiting times.
    photon_energies = Quantity([1, 1, 2, 3, 5, 4, 6, 7, 8], unit=u.keV)
    photon_waiting_times = Quantity(
        np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 1., 0.])*hpu.frame_duration, unit=u.s)
    incident_photons = Table([photon_waiting_times.cumsum(), photon_energies], names=("time", "energy"))
    # Define expected output photon list.
    expected_indices = [0, 3, 4, 6, 7]
    expected_photons = incident_photons[expected_indices]
    expected_photons["energy"][4] = (photon_energies[7]+photon_energies[8]).value
    # Calculate test measured photon list by calling
    # simulate_hexitec_photon_list_1pixel().
    hpu.simulate_hexitec_on_photon_list_1pixel(incident_photons)
    # Assert test photon list is the same as expected photon list.
    assert all(hpu.measured_photons["energy"] == expected_photons["energy"])
    assert 0.*u.s <= all(expected_photons["time"]-hpu.measured_photons["time"]) < hpu.frame_duration


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
