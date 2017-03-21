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
DATAFRAME_MAX_POINTS = 1e8

class HexitecSimulation():
    """Simulates how HEXITEC records incident photons."""

    def __init__(self, frame_rate=Quantity(3900., unit=1/u.s),
                 incident_xpixel_range=(0,80), incident_ypixel_range=(0,80),
                 readout_xpixel_range=(0,80), readout_ypixel_range=(0,80),
                 charge_cloud_sigma=None, charge_drift_length=1*u.mm,
                 detector_temperature=None, bias_voltage=None, threshold=None):
        """
        Instantiates a HexitecPileUp object.

        Parameters
        ----------
        frame_rate : `astropy.units.quantity.Quantity`
            Operating frame rate of ASIC.
        incident_xpixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            x-direction upon which incident photons fall.
        incident_ypixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            y-direction upon which incident photons fall.
        readout_xpixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            x-direction to be read out.
        readout_ypixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            x-direction to be read out.
        charge_cloud_sigma : `astropy.units.quantity.Quantity`
            Standard deviation of charge cloud assuming it to be a
            2D symmetric gaussian. Default=None.
            If not set, the charge clous standard deviation is calculated
            with self._charge_cloud_sigma() using charge_drift_length,
            detector_temperature, and bias_voltage inputs (below).
        charge_drift_length : `astropy.units.quantity.Quantity`
            Drift length of charge cloud from site of photon interaction to
            anode. Given by CdTe thickness - mean free path of photon in CdTe.
            Default=1mm
        detector_temperature : `astropy.units.quantity.Quantity`
            Operating temperature of the detector.
            Default=None
        bias_voltage : `astropy.units.quantity.Quantity`
            Operating bias voltage of detector.
            Default=None
        threshold: `astropy.units.quantity.Quantity`
            Threshold below which photons are not recorded.  Must be in units
            of energy or voltage.  If unit is energy, threshold refers to
            photon energy.  If unit is voltage, threshold refers to voltage
            induced in pixel due to a photon hit which is a function of the
            photon's energy.
            Default=None implies a threshold of 0

        """
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
        pixel_pitch = 250*u.um
        # Charge cloud standard deviation in units of pixel length.
        if charge_cloud_sigma:
            self._charge_cloud_x_sigma = charge_cloud_sigma
            self._charge_cloud_y_sigma = charge_cloud_sigma
        else:
            self._charge_cloud_x_sigma = self._charge_cloud_y_sigma = \
              self._charge_cloud_sigma(charge_drift_length, detector_temperature,
                                       bias_voltage).to(u.um).value/pixel_pitch.to(u.um).value
        # Define threshold
        self.threshold = threshold


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
        print("Generating random numbers for photon energy transformation.")
        time1 = timeit.default_timer()
        randoms = np.asarray([random.random() for i in range(n_counts)])*cdf_upper[-1]
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
        # Generate array of spectrum bin indices.
        print("Transforming random numbers into photon energies.")
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
        print("Finished in {0} s.".format(time2-time1))
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
                print("Processing photons hitting pixel ({0}, {1}) of {2} at {3}".format(
                        i, j, (self.readout_xpixel_range[1]-1,
                               self.readout_ypixel_range[1]-1), datetime.now()))
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
                print("Finished processing pixel ({0}, {1}) of {2} in {3} s.".format(
                    i, j, (self.readout_xpixel_range[1]-1, self.readout_ypixel_range[1]-1),
                    time2-time1))
                print(" ")
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
        print("Generating random numbers for photon energy transformation.")
        time1 = timeit.default_timer()
        randoms = np.asarray([random.random() for i in range(n_counts)])*cdf_upper[-1]
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
        # Generate array of spectrum bin indices.
        print("Transforming random numbers into photon energies.")
        time1 = timeit.default_timer()
        bin_indices = np.arange(len(self.incident_spectrum["lower_bin_edges"]))
        # Generate random energies from randomly generated CDF values.
        photon_energies = Quantity([self.incident_spectrum["lower_bin_edges"].data[
            bin_indices[np.logical_and(r >= cdf_lower, r < cdf_upper)][0]]
            for r in randoms], unit=self.incident_spectrum["lower_bin_edges"].unit)
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
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
                                   x_pixels[w], y_pixels[w], neighbor_positions[w]],
                                  names=("time", "energy", "x_pixel", "y_pixel",
                                         "neighbor_positions"))
        return pixelated_photons


    def _divide_charge_among_pixels(self, x, y, x_sigma, y_sigma, n_1d_neighbours):
        """Divides charge-shared photon hits into separate photon hits."""
        # Generate pixel numbers of central pixel & nearest neighbours.
        x_hit_pixel = int(x)
        y_hit_pixel = int(y)
        half_nearest = (n_1d_neighbours-1)/2
        neighbours_range = np.arange(-half_nearest, half_nearest+1)
        x_shared_pixels = np.array([x_hit_pixel+i for i in neighbours_range]*n_1d_neighbours)
        y_shared_pixels = np.array(
            [[y_hit_pixel+i]*n_1d_neighbours for i in neighbours_range]).flatten()
        neighbor_positions = ["down left", "down", "down right", "left", "central",
                              "right", "up left", "up", "up right"]
        # Find fraction of charge in each pixel.
        return x_shared_pixels, y_shared_pixels, self._integrate_gaussian2d(
            (x_shared_pixels, x_shared_pixels + 1), (y_shared_pixels, y_shared_pixels + 1),
            x, y, x_sigma, y_sigma), neighbor_positions


    def _charge_cloud_sigma(self, charge_drift_length, detector_temperature, bias_voltage):
        """
        Returns the standard deviation of the charge cloud when is reaches the anode.

        Parameters
        ----------
        charge_drift_length : `astropy.units.quantity.Quantity`
            Drift length of charge cloud from site of photon interactiont to anode.
            Given by CdTe thickness - mean free path of photon in CdTe.
        detector_temperature : `astropy.units.quantity.Quantity`
            Operating temperature of the detector.
        bias_voltage : `astropy.units.quantity.Quantity`
            Operating bias voltage of detector.

        Returns
        -------
        sigma : `astropy.units.quantity.Quantity`
            1D standard deviation of charge cloud at anode.

        References
        ----------
        [1] : Veale et al. (2014), Measurements of Charege Sharing in Small Pixelated Detectors
        [2] : Iniewski et al. (2007)

        """
        # Determine initial radius of charge cloud (FWHM), r0, from
        # empirical relation derived by Veale et al. (2014) (Fig. 8).
        r0 = Quantity(
            0.1477*detector_temperature.to(u.Celsius, equivalencies=u.temperature()).value+14.66,
            unit="um")
        # Determine radius (FWHM) at anode.
        r = r0 + 1.15*charge_drift_length*np.sqrt(
            2*constants.k_B.si.value*detector_temperature.to(
                u.K, equivalencies=u.temperature()).value/ \
                (constants.e.si.value*abs(bias_voltage.si.value)))
        # Convert FWHM to sigma.
        return r.to(u.um)/1.15


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
        samples_per_frame = int(round((frame_duration/sample_step.to(frame_duration.unit)).value))
        frame_duration_in_sample_unit = int(round(frame_duration.to(self._sample_unit).value))
        # Determine how many frames will be needed to model all photons
        # hitting pixel.
        # Get frame number from time 0 of all photons in pixel.
        photon_frame_numbers = np.array(incident_photons["time"]/frame_duration.value).astype(int)
        # Extend frame numbers to include adjacent frames to allow for
        # frame carryover while ensuring no duplicate frames.
        extended_frame_numbers = np.unique(np.array(
            [[x-1, x, x+1] for x in photon_frame_numbers]).ravel())
        extended_frame_numbers = extended_frame_numbers[np.where(extended_frame_numbers >= 0)[0]]
        # Determine total frames required for all photons hitting pixel.
        total_n_frames = len(extended_frame_numbers)
        # Break incident photons into sub time series manageable for a
        # pandas time series.
        # Determine max frames per subseries.
        subseries_max_frames = int(DATAFRAME_MAX_POINTS*sample_step.value/frame_duration.value)
        subseries_frame_edges = extended_frame_numbers[np.arange(0, total_n_frames,
                                                                 subseries_max_frames)]
        if subseries_frame_edges[-1] < extended_frame_numbers[-1]:
            subseries_frame_edges = np.append(subseries_frame_edges, extended_frame_numbers[-1])
        # Determine number of subseries
        n_subseries = len(subseries_frame_edges)-1
        # Define arrays to hold measured photons
        measured_photon_times = np.array([], dtype=float)
        measured_photon_energies = np.array([], dtype=float)
        # Use for loop to analyse each sub timeseries.
        print("Photons will be analysed in {0} sub-timeseries.".format(n_subseries))
        for i in range(n_subseries):
            print("Processing subseries {0} of {1} at {2}".format(i+1, n_subseries,
                                                                  datetime.now()))
            time1 = timeit.default_timer()

            # Determine which photons are in current subseries.  Include
            # photons in the frames either side of the subseries edges
            # as their voltage pulses may influence the first and last
            # frames of the subseries.  Any detections in these outer
            # frames should be removed later.
            subseries_first_frame = subseries_frame_edges[i]-1
            subseries_last_frame = subseries_frame_edges[i+1]+1
            subseries_incident_photons = incident_photons[np.logical_and(
                photon_frame_numbers >= subseries_first_frame,
                photon_frame_numbers < subseries_last_frame)]

            # Model voltage signal inside HEXITEC ASIC in response to
            # photon hits.
            # Define some basic parameters of subseries.
            subseries_frame_numbers = extended_frame_numbers[np.logical_and(
                extended_frame_numbers >= subseries_first_frame,
                extended_frame_numbers < subseries_last_frame)]
            n_subseries_frames = len(subseries_frame_numbers)
            subseries_photon_frame_numbers = photon_frame_numbers[np.logical_and(
                photon_frame_numbers >= subseries_first_frame,
                photon_frame_numbers < subseries_last_frame)]
            # Generate time stamps for sparse timeseries & determine
            # how many photons there are in each frame.
            n_samples = n_subseries_frames*samples_per_frame
            timestamps = np.zeros(n_samples)
            n_photons_per_frame = np.zeros(n_subseries_frames, dtype=int)
            for j, fn in enumerate(subseries_frame_numbers):
                timestamps[j*samples_per_frame:(j+1)*samples_per_frame] = np.arange(
                    fn*frame_duration_in_sample_unit, (fn+1)*frame_duration_in_sample_unit,
                    sample_step.to(self._sample_unit).value)
                n_photons_per_frame[j] = len(np.where(subseries_photon_frame_numbers == fn)[0])
            n_photons_per_frame = n_photons_per_frame[np.where(n_photons_per_frame > 0)[0]]
            # Get indices of subseries_frame_numbers array corresponding
            # to frames in subseries_photons_frame_numbers array.
            ind = np.arange(n_subseries_frames)[np.in1d(subseries_frame_numbers,
                                                        subseries_photon_frame_numbers)]
            inds_nested = [[ind[j]]*n_photons_per_frame[j]
                           for j in range(len(n_photons_per_frame))]
            inds = [item for sublist in inds_nested for item in sublist]
            # Get frames skipped as a function of subseries frame number.
            m = np.insert(subseries_frame_numbers, 0, 0)
            skipped_frames = (m[1:]-m[:-1]-1).cumsum()+1
            # Get number of frames skipped as a function of photon
            # frame number.
            skipped_frames_by_photon = skipped_frames[inds]
            # Get indices of photon times in subseries.
            photon_time_indices = np.rint(
                subseries_incident_photons["time"].data/sample_step.to(
                    subseries_incident_photons["time"].unit).value).astype(
                        int)-skipped_frames_by_photon*samples_per_frame
            # If there are photons assigned to same time index combine
            # them as though they were one photon with an energy equal
            # to the sum of photon energies.
            non_simul_photon_time_indices, non_simul_photon_list_indices, \
              n_photons_per_time_index = np.unique(photon_time_indices, return_index=True,
                                                   return_counts=True)
            if max(n_photons_per_time_index) > 1:
                w = np.where(n_photons_per_time_index > 1)[0]
                subseries_incident_photon_energies = \
                  subseries_incident_photons["energy"][non_simul_photon_list_indices]
                subseries_incident_photon_energies[w] = \
                  [sum(subseries_incident_photons["energy"][j:j+n_photons_per_time_index[j]])
                   for j in w]
            else:
                subseries_incident_photon_energies = subseries_incident_photons["energy"]
            # Convert photon energies to voltage timeseries
            voltage_deltas = self._convert_photon_energy_to_voltage(
                subseries_incident_photon_energies)
            voltage_delta_timeseries = np.zeros(n_samples)
            voltage_delta_timeseries[non_simul_photon_time_indices] = voltage_deltas.value
            # Convolve voltage delta function time series with voltage
            # pulse shape.
            voltage = np.convolve(voltage_delta_timeseries, self._voltage_pulse_shape)
            # Trim edges of convolved time series so that peaks of voltage
            # pulses align with photon times.
            start_index = int(np.rint(self._voltage_peaking_time/self._sample_step))
            end_index = int(np.rint(self._voltage_decay_time/self._sample_step))*(-1)+1
            voltage = voltage[start_index:end_index]
            # Generate time series from voltage and timestamps.
            subseries = pandas.DataFrame(
                voltage, index=pandas.to_timedelta(timestamps, self._sample_unit),
                columns=["voltage"])
            subseries_voltage_unit = voltage_deltas.unit

            # Calculate how HEXITEC peak-hold would measure photon
            # times and energies.
            subseries_measured_photon_times, subseries_measured_photon_energies = \
              self._convert_voltage_timeseries_to_measured_photons(
                  subseries, voltage_unit=subseries_voltage_unit, threshold=self.threshold)
            subseries_measured_photon_times = subseries_measured_photon_times.to(
                incident_photons["time"].unit)
            subseries_measured_photon_energies = subseries_measured_photon_energies.to(
                incident_photons["energy"].unit)

            # Add subseries measured photon times and energies to all
            # photon times and energies Quantities excluding any
            # photons from first or last frame.
            w = np.logical_and(
                subseries_measured_photon_times >= subseries_frame_edges[i]*frame_duration,
                subseries_measured_photon_times < subseries_frame_edges[i+1]*frame_duration)
            measured_photon_times = np.append(
                measured_photon_times, subseries_measured_photon_times.value[w])
            measured_photon_energies = np.append(
                measured_photon_energies, subseries_measured_photon_energies.value[w])
            time2 = timeit.default_timer()
            print("Finished {0}th subseries in {1} s".format(i+1, time2-time1))
            print(" ")
        # Convert results into table and return.
        return Table(
            [Quantity(measured_photon_times, unit=incident_photons["time"].unit),
             Quantity(measured_photon_energies, unit=incident_photons["energy"].unit)],
            names=("time", "energy"))


    def _convert_voltage_timeseries_to_measured_photons(self, timeseries, voltage_unit, threshold=None):
        """Converts a time series of HEXITEC voltage to measured photons."""
        # Convert timeseries into measured photon list by resampling
        # at frame rate and taking min.
        # In resample command, 'xN' signifies x nanoseconds.
        frame_peaks = timeseries.resample(
            "{0}N".format(int(self.frame_duration.to(u.ns).value)), how=min).fillna(0)
        # Convert voltages back to photon energies
        if threshold is None:
            threshold = 0.
        else:
            try:
                threshold = self._convert_photon_energy_to_voltage(threshold).value
            except u.UnitConversionError:
                threshold = threshold.to(u.V).value
            except u.UnitConversionError as err:
                err.args = (
                    "'{0}' not convertible to either 'keV' (energy) or 'V' (voltage)".format(
                        threshold.unit),)
                raise err
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
