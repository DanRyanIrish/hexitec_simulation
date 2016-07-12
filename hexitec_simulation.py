"""This module contains a class for simulating the effect of pile up in HEXITEC."""

import random
import warnings
import math
import timeit
from datetime import datetime

import numpy as np
from numpy import ma
import astropy.units as u
from astropy.table import Table, vstack
from astropy.units.quantity import Quantity
from astropy import constants
import pandas
from scipy.special import erf

# Defining max number of data points in a pandas dataframe.
DATAFRAME_MAX_POINTS = 5e7

class HexitecSimulation():
    """Simulates how HEXITEC records incident photons."""

    def __init__(self, frame_rate=Quantity(3900., unit=1/u.s),
                 incident_xpixel_range=(0,80), incident_ypixel_range=(0,80),
                 readout_xpixel_range=(0,80), readout_ypixel_range=(0,80)):
        """Instantiates a HexitecPileUp object."""
        # Define some magic numbers. N.B. _sample_unit must be in string
        # format so it can be used for Quantities and numpy datetime64.
        self._sample_unit = 'ns'
        self._sample_step = Quantity(100., unit=self._sample_unit)
        self._voltage_peaking_time = Quantity(2., unit='us').to(self._sample_unit)
        self._voltage_decay_time = Quantity(8., unit='us').to(self._sample_unit)
        self._voltage_pulse_shape = self._define_voltage_pulse_shape()
        # Set frame duration from inverse of input frame_rate and round
        # to nearest multiple of self._sample_step.
        self.frame_duration = Quantity(
            round((1./frame_rate).to(self._sample_unit).value/self._sample_step.value
                  )*self._sample_step).to(1/frame_rate.unit)
        self.incident_xpixel_range = incident_xpixel_range
        self.incident_ypixel_range = incident_ypixel_range
        self.readout_xpixel_range = readout_xpixel_range
        self.readout_ypixel_range = readout_ypixel_range
        self._n_1d_neighbours = 3
        charge_cloud_3sigma = 86.875/250.  # charge cloud diameter/pixel length
        #charge_cloud_3sigma = 17.2/250.  # charge cloud diameter/pixel length
        self._charge_cloud_x_sigma = charge_cloud_3sigma/3.
        self._charge_cloud_y_sigma = charge_cloud_3sigma/3.


    def simulate_hexitec_on_spectrum_1pixel(self, incident_spectrum, photon_rate, n_photons):
        """
        Simulates how a single HEXITEC pixel records photons from a given spectrum.

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
        self.measured_photons = self.simulate_hexitec_on_photon_list_1pixel(incident_photons)
        # Convert measured photon list into counts into bins with same
        # bins as the incident spectrum.  N.B. Measured photons can
        # have energies outside incident spectrum energy range.  For
        # these bins, use mean bin width of incident spectrum.
        # N.B. The exact values of incident spectrum bins must be used
        # as rounding errors/approximations can cause erroneous
        # behaviour when binning counts.
        bin_width = np.mean(
            incident_spectrum["upper_bin_edges"]-incident_spectrum["lower_bin_edges"])
        lower_bins = np.arange(incident_spectrum["lower_bin_edges"][0],
                               measured_photons["energy"].min()-bin_width,
                               -bin_width).sort()
        upper_bins = np.arange(incident_spectrum["upper_bin_edges"][-1]+bin_width,
                               measured_photons["energy"].max()+bin_width, bin_width)
        bin_edges = np.concatenate(
            (lower_bins[:-1], hpu.incident_spectrum["lower_bin_edges"],
             np.array([hpu.incident_spectrum["upper_bin_edges"][-1]]), upper_bins))
        # Return an astropy table of the measured spectrum.
        measured_counts = np.histogram(self.measured_photons["energy"], bins=bin_edges)[0]
        self.measured_spectrum = Table(
            [self.incident_spectrum["lower_bin_edges"],
             self.incident_spectrum["upper_bin_edges"], measured_counts],
            names=("lower_bin_edges", "upper_bin_edges", "counts"))


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


    def simulate_hexitec_on_spectrum(self, incident_spectrum, photon_rate, n_photons):
        """Simulates how a grid of HEXITEC pixels records photons from a given spectrum."""
        # Generate random photons incident on detector.
        self.incident_photons = self.generate_random_photons(incident_spectrum,
                                                             photon_rate, n_photons)
        # Simulate how HEXITEC records incident photons.
        self.simulate_hexitec_on_photon_list(self.incident_photons)


    def simulate_hexitec_on_photon_list(self, incident_photons):
        """Simulates how HEXITEC pixels record photons from a given photon list."""
        # Produce photon list accounting for charge sharing.
        pixelated_photons = self.account_for_charge_sharing_in_photon_list(
            incident_photons, self._charge_cloud_x_sigma,
            self._charge_cloud_y_sigma, self._n_1d_neighbours)
        # Separate photons by pixel and simulate each pixel's
        # measurements.
        measured_photons = Table([Quantity([], unit=self.incident_photons["time"].unit),
                                  Quantity([], unit=self.incident_photons["energy"].unit),
                                  [], []], names=("time", "energy", "x", "y"))
        for j in range(self.readout_ypixel_range[0], self.readout_ypixel_range[1]):
            for i in range(self.readout_xpixel_range[0], self.readout_xpixel_range[1]):
                print "Processing photons hitting pixel ({0}, {1}) of {2} at {3}".format(
                        i, j, (self.readout_xpixel_range[1]-1, self.readout_ypixel_range[1]-1), datetime.now())
                time1 = timeit.default_timer()
                w = np.logical_and(pixelated_photons["x_pixel"] == i,
                                   pixelated_photons["y_pixel"] == j)
                if w.any():
                    pixel_measured_photons = \
                      self.simulate_hexitec_on_photon_list_1pixel(pixelated_photons[w])
                    # Add pixel info to pixel_measured_photons table.
                    pixel_measured_photons["x"] = [i]*len(pixel_measured_photons)
                    pixel_measured_photons["y"] = [j]*len(pixel_measured_photons)
                    measured_photons = vstack((measured_photons, pixel_measured_photons))
                time2 = timeit.default_timer()
                print "Finished processing pixel ({0}, {1}) of {2} in {3} s.".format(
                    i, j, (self.readout_xpixel_range[1]-1, self.readout_ypixel_range[1]-1), time2-time1)
                print " "
        # Sort photons by time and return to object.
        measured_photons.sort("time")
        self.measured_photons = measured_photons


    def generate_random_photons(self, incident_spectrum, photon_rate, n_photons):
        """Generates random photon times, energies and detector hit locations."""
        self.photon_rate = photon_rate
        # Generate photon times since start of observations (t=0).
        photon_times = self._generate_random_photon_times(n_photons)
        # Generate photon energies.
        photon_energies = self._generate_random_photon_energies_from_spectrum(
            incident_spectrum, photon_rate, n_photons)
        # Generate photon hit locations.
        x_locations, y_locations = self._generate_random_photon_locations(n_photons)
        # Associate photon times, energies and locations.
        return Table([photon_times, photon_energies, x_locations, y_locations],
                     names=("time", "energy", "x", "y"))


    def _generate_random_photon_energies_from_spectrum(self, incident_spectrum,
                                                       photon_rate, n_photons):
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
        photon_energies : `astropy.units.quantity.Quantity`
            Photon energies

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
        time2 = timeit.default_timer()
        print "Finished in {0} s.".format(time2-time1)
        return photon_energies


    def _generate_random_photon_times(self, n_photons):
        """Generates random photon times."""
        # Generate random waiting times before each photon.
        photon_waiting_times = Quantity(
            np.random.exponential(1./self.photon_rate, n_photons), unit='s')
        return photon_waiting_times.cumsum()


    def _generate_random_photon_locations(self, n_photons):
        """Generates random photon hit locations."""
        # Generate random x locations for each photon.
        x = np.random.uniform(
                self.incident_xpixel_range[0], self.incident_xpixel_range[1], n_photons)
        y = np.random.uniform(
                self.incident_ypixel_range[0], self.incident_ypixel_range[1], n_photons)
        return x, y

    def simulate_cdte_fluorescence(energy):
        """Simulates Cadmium and Tellurium fluorescence."""
        # Define occurence rates, mean free paths and energies of
        # fluorecent photons.
        ca_probability = 0.1
        te_probability = 0.1
        ca_mfp = 0.1
        te_mfp = 0.1
        ca_energy = 0.1
        te_energy = 0.1
        # Select a random number to determine whether in incident
        # photon will produce fluorescence.
        
        

    def account_for_charge_sharing_in_photon_list(self, incident_photons, charge_cloud_x_sigma,
                                                  charge_cloud_y_sigma, n_1d_neighbours):
        """
        Divides photon hits among neighbouring pixels by the charge sharing process.

        """
        # For each photon create extra pseudo-photons in nearest
        # neighbours due to charge sharing.
        n_neighbours = n_1d_neighbours**2
        n_photons_shared = len(incident_photons)*n_neighbours
        times = np.full(n_photons_shared, np.nan)
        x_pixels = np.full(n_photons_shared, np.nan)
        y_pixels = np.full(n_photons_shared, np.nan)
        energy = np.full(n_photons_shared, np.nan)
        neighbor_positions = np.array([""]*n_photons_shared, dtype="S10")
        for i, photon in enumerate(incident_photons):
            # Find fraction of energy in central & neighbouring pixels.
            x_shared_pixels, y_shared_pixels, fractional_energy_in_pixels, \
            pixel_neighbor_positions = self._divide_charge_among_pixels(
                    photon["x"], photon["y"], charge_cloud_x_sigma,
                    charge_cloud_y_sigma, n_1d_neighbours)
            # Insert new shared photon parameters into relevant list.
            times[i*n_neighbours:(i+1)*n_neighbours] = photon["time"]
            x_pixels[i*n_neighbours:(i+1)*n_neighbours] = x_shared_pixels
            y_pixels[i*n_neighbours:(i+1)*n_neighbours] = y_shared_pixels
            energy[i*n_neighbours:(i+1)*n_neighbours] = \
              photon["energy"]*fractional_energy_in_pixels
            neighbor_positions[i*n_neighbours:(i+1)*n_neighbours] = pixel_neighbor_positions
        # Discard any charge lost at edges of detector and events with
        # 0 energy.
        w = np.logical_and(
                np.logical_and(x_pixels >= self.readout_xpixel_range[0],
                               x_pixels < self.readout_xpixel_range[1]),
                np.logical_and(y_pixels >= self.readout_ypixel_range[0],
                               y_pixels < self.readout_ypixel_range[1]),
                energy > 0.)
        # Combine shared photons into new table.
        pixelated_photons = Table([Quantity(times[w], incident_photons["time"].unit),
                                   Quantity(energy[w], incident_photons["energy"].unit),
                                   x_pixels[w], y_pixels[w], neighbor_positions],
                                  names=("time", "energy", "x_pixel", "y_pixel",
                                         "neighbor_positions"))
        return pixelated_photons


    def _divide_charge_among_pixels(self, x, y, x_sigma, y_sigma, n_1d_neighbours):
        """Divides charge-shared photon hits into separate photon hits."""
        # Generate pixel numbers of central pixel & nearest neighbours.
        x_hit_pixel = int(x)
        y_hit_pixel = int(y)
        half_nearest = (n_1d_neighbours-1)/2
        neighbours_range = range(-half_nearest, half_nearest+1)
        x_shared_pixels = np.array([x_hit_pixel+i for i in neighbours_range]*n_1d_neighbours)
        y_shared_pixels = np.array(
            [[y_hit_pixel+i]*n_1d_neighbours for i in neighbours_range]).flatten()
        neighbor_positions = ["down left", "down", "down right", "left", "central",
                              "right", "up left", "up", "up right"]
        # Find fraction of charge in each pixel.
        return x_shared_pixels, y_shared_pixels, self._integrate_gaussian2d(
            (x_shared_pixels, x_shared_pixels+1), (y_shared_pixels, y_shared_pixels+1),
            x, y, x_sigma, y_sigma), neighbor_positions


    def _charge_cloud_radius(d, T, V):
        """
        Returns the radius of the charge cloud when is reaches the anode.

        Parameters
        ----------
        d : `astropy.units.quantity.Quantity`
            Drift length of charge cloud from site of photon interactiont to anode.
            Given by CdTe thickness - mean free path of photon in CdTe.
        T : `astropy.units.quantity.Quantity`
            Operating temperature of the detector.
        V : `astropy.units.quantity.Quantity`
            Operating bias voltage of detector.

        Returns
        -------
        r : `astropy.units.quantity.Quantity`
            Radius of charge cloud at anode.

        References
        ----------
        [1] : Veale et al. (2014), Measurements of Charege Sharing in Small Pixelated Detectors
        [2] : Iniewski et al. (2007)

        """
        # Determine initial radius of charge cloud, r0, from empirical
        # relation derived by Veale et al. (2014).
        r0 = Quantity(0.1477*T.to(u.Celsius).value+14.66, unit="um")
        # Return Quantity of radius at anode.  Is the unit system correct?  I don't think so.  Check!
        return r0 + Quantity(1.15*d.si.value* \
                             (2*constants.k_B.si.value*T.to(u.Celsius).value/ \
                              (constants.e.si.value*abs(V.to("V").value)))**0.5,
                              unit="um")


    def _integrate_gaussian(self, limits, mu, sigma):
        return (1/(sigma*np.sqrt(2*np.pi))) * np.sqrt(np.pi/2)*sigma* \
          (erf((limits[1]-mu)/(np.sqrt(2)*sigma))-erf((limits[0]-mu)/(np.sqrt(2)*sigma)))


    def _integrate_gaussian2d(self, x_limits, y_limits, x_mu, y_mu, x_sigma, y_sigma):
        return self._integrate_gaussian(x_limits, x_mu, x_sigma)* \
          self._integrate_gaussian(y_limits, y_mu, y_sigma)


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
        #self.incident_photons = incident_photons
        sample_step = self._sample_step.to(incident_photons["time"].unit)
        frame_duration = self.frame_duration.to(incident_photons["time"].unit)
        # Break incident photons into sub time series manageable for a
        # pandas time series.
        # Convert max points into max duration such that duration
        # includes an integer number of frames.
        subseries_n_frames = int(DATAFRAME_MAX_POINTS*sample_step.value/frame_duration.value)
        timeseries_n_frames = subseries_n_frames+2
        subseries_max_duration = Quantity(subseries_n_frames*frame_duration)
        # Determine lower time edges of each sub time series.
        final_frame_upper_edge = int(
                incident_photons["time"][-1]/frame_duration.value+1)*frame_duration.value
        subseries_edges = np.arange(0, final_frame_upper_edge, subseries_max_duration.value)
        if subseries_edges[-1] < final_frame_upper_edge:
            subseries_edges = np.append(subseries_edges, final_frame_upper_edge)
        subseries_edges = subseries_edges*incident_photons["time"].unit
        # Define arrays to hold measured photons
        measured_photon_times = np.array([], dtype=float)
        measured_photon_energies = np.array([], dtype=float)
        # Use for loop to analyse each sub timeseries.
        print "Photons will be analysed in {0} sub-timeseries.".format(len(subseries_edges)-1)
        for i in range(len(subseries_edges)-1):
            print "Processing subseries {0} of {1} at {2}".format(
                i+1, len(subseries_edges)-1, datetime.now())
            time1 = timeit.default_timer()
            # Determine which photons are in current subseries.  Include
            # photons in the frames either side of the subseries edges
            # as their voltage pulses may influence the first and last
            # frames of the subseries.  Any detections in these outer
            # frames should be removed later.
            timeseries_start = subseries_edges[i]-frame_duration
            timeseries_end = subseries_edges[i+1]+frame_duration
            timeseries_incident_photons = incident_photons[
                np.logical_and(incident_photons["time"] >= timeseries_start,
                               incident_photons["time"] < timeseries_end)]
            # If there are photons in this subseries, continue.
            # Else move to next frame.
            if len(timeseries_incident_photons) > 0:
                # Generate sub time series of voltage pulses due to photons.
                timeseries, timseries_voltage_unit = \
                  self._convert_photons_to_voltage_timeseries(
                      timeseries_incident_photons, timeseries_start, timeseries_n_frames)
                subseries_measured_photon_times, subseries_measured_photon_energies = \
                  self._convert_voltage_timeseries_to_measured_photons(
                      timeseries, voltage_unit=timseries_voltage_unit)
                subseries_measured_photon_times = subseries_measured_photon_times.to(
                    incident_photons["time"].unit)
                subseries_measured_photon_energies = subseries_measured_photon_energies.to(
                    incident_photons["energy"].unit)
                # Add subseries measured photon times and energies to all
                # photon times and energies Quantities excluding any
                # photons from first or last frame.
                w = np.logical_and(subseries_measured_photon_times >= subseries_edges[i],
                                   subseries_measured_photon_times < subseries_edges[i+1])
                measured_photon_times = np.append(
                    measured_photon_times, subseries_measured_photon_times.value[w])
                measured_photon_energies = np.append(
                    measured_photon_energies, subseries_measured_photon_energies.value[w])
            time2 = timeit.default_timer()
            print "Finished {0}th subseries in {1} s".format(i+1, time2-time1)
            print " "
        # Convert results into table and return.
        return Table(
            [Quantity(measured_photon_times, unit=incident_photons["time"].unit),
             Quantity(measured_photon_energies, unit=incident_photons["energy"].unit)],
            names=("time", "energy"))


    def _convert_voltage_timeseries_to_measured_photons(self, timeseries, voltage_unit):
        """Converts a time series of HEXITEC voltage to measured photons."""
        # Convert timeseries into measured photon list by resampling
        # at frame rate and taking min.
        # In resample command, 'xN' signifies x nanoseconds.
        frame_peaks = timeseries.resample(
            "{0}N".format(int(self.frame_duration.to(u.ns).value)), how=min)
        # Convert voltages back to photon energies
        threshold = 0.
        w = np.where(frame_peaks["voltage"] < threshold)[0]
        measured_photon_energies = self._convert_voltages_to_photon_energy(
            frame_peaks["voltage"][w].values, voltage_unit)
        # Determine time unit of pandas timeseries and convert photon
        # times to Quantity.
        frame_duration_secs = self.frame_duration.to("s").value
        rounded_photon_times = np.round(
            frame_peaks.index[w].total_seconds()/frame_duration_secs)*frame_duration_secs
        measured_photon_times = Quantity(rounded_photon_times, unit="s")
        # Combine photon times and energies into measured photon list.
        return measured_photon_times, measured_photon_energies


    def _convert_photons_to_voltage_timeseries(self, incident_photons, series_start, n_frames):
        """
        Create a time series of photon energies with a given sampling frequency.

        Parameters
        ----------
        incident_photons :
        series_start_end : `tuple`
          Start and end times of time series and number of frames.
        
        Returns
        -------
        timeseries : `pandas.DataFrame`
          Time series of voltage vs. time inside HEXITEC ASIC due to photon hits.
        
        """
        # Ensure certain variables are in correct unit.
        sample_step = self._sample_step.to(self._sample_unit)
        frame_duration = self.frame_duration.to(self._sample_unit)
        # If there are simulataneous photon hits (i.e. photons assigned
        # to same time index) combine them as though they were one
        # photon with an energy equal to the sum of the simultaneous
        # photons.
        non_simul_photon_times, non_simul_photon_time_indices, n_incident_photons_per_index = \
          np.unique(incident_photons["time"], return_index=True, return_counts=True)
        non_simul_photon_time_indices = np.sort(non_simul_photon_time_indices)
        if max(n_incident_photons_per_index) > 1:
            w = np.where(n_incident_photons_per_index > 1)[0]
            combined_photons = incident_photons[non_simul_photon_time_indices]
            combined_photons["energy"][w] = \
              [sum(incident_photons["energy"][i:i+n_incident_photons_per_index[i]])
               for i in w]
        else:
            combined_photons = incident_photons
        # Convert photon energies to voltage delta functions.
        voltage_deltas = \
          self._convert_photon_energy_to_voltage(combined_photons["energy"])
        # Create time series of voltage delta functions.
        n_samples = int(np.rint(n_frames*frame_duration.value/sample_step.value))
        voltage_delta_timeseries = np.zeros(n_samples)
        # Calculate indices of photon hits in voltage time series.
        photon_time_indices = np.rint(
            (combined_photons["time"].data-series_start.to(combined_photons["time"].unit).value) \
            /sample_step.to(combined_photons["time"].unit).value).astype(int)
        # Enter voltage delta functions into timeseries.
        voltage_delta_timeseries[photon_time_indices] = voltage_deltas.value
        # Convolve voltage delta function time series with voltage
        # pulse shape.
        voltage = np.convolve(voltage_delta_timeseries, self._voltage_pulse_shape)
        # Trim edges of convolved time series so that peaks of voltage
        # pulses align with photon times.
        start_index = int(np.rint(self._voltage_peaking_time/self._sample_step))
        end_index = int(np.rint(self._voltage_decay_time/self._sample_step))*(-1)+1
        voltage = voltage[start_index:end_index]
        # Define timestamps for timeseries.
        timestamps = np.arange(0, n_samples*sample_step.value, sample_step.value) + \
          series_start.to(self._sample_unit).value
        # Generate time series from voltage and timestamps.
        timeseries = pandas.DataFrame(
            voltage, index=pandas.to_timedelta(timestamps, self._sample_unit), columns=["voltage"])
        timeseries_voltage_unit = voltage_deltas.unit
        return timeseries, timeseries_voltage_unit


    def _define_voltage_pulse_shape(self):
        """Defines the normalised shape of voltage pulse with given discrete sampling frequency."""
        sample_step = self._sample_step.to(self._sample_unit).value
        # Convert input peaking and decay times to a standard unit.
        voltage_peaking_time = self._voltage_peaking_time.to(self._sample_unit).value
        voltage_decay_time = self._voltage_decay_time.to(self._sample_unit).value
        # Define other Gaussian parameters.
        mu = voltage_peaking_time
        zero_equivalent = 1e-3
        sigma2_peak = -0.5*mu**2/np.log(zero_equivalent)
        sigma2_decay = \
          -0.5*(voltage_peaking_time+voltage_decay_time-mu)**2/np.log(zero_equivalent)
        # Generate time data points for peak and decay phases.
        t_peaking = np.arange(0, voltage_peaking_time, sample_step)
        t_decay = np.arange(voltage_peaking_time,
                            voltage_peaking_time+voltage_decay_time,
                            sample_step)
        voltage_pulse_shape = np.append(np.exp(-(t_peaking-mu)**2./(2*sigma2_peak)),
                                        np.exp(-(t_decay-mu)**2./(2*sigma2_decay)))
        return voltage_pulse_shape


    def _convert_photon_energy_to_voltage(self, photon_energy):
        """Determines HEXITEC peak voltage due photon of given energy."""
        return -photon_energy.to(u.keV).value*u.V


    def _convert_voltages_to_photon_energy(self, voltages, voltage_unit):
        """Determines photon energy from HEXITEC peak voltage."""
        return -(voltages*voltage_unit).to(u.V).value*u.keV
