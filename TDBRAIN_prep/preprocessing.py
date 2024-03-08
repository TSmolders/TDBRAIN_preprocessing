"""
Created on Thu Mar 7 16:23 2024

author: T.Smolders

description: Object for automated preprocessing of the TDBRAIN dataset

name: preprocessing.py

version: 1.0

"""

# Importing necessary packages
import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from pyprep.prep_pipeline import PrepPipeline

mne.set_log_level('WARNING')

class Preproccesing:

    
    def __init__(
            self,
            filename, # path to the .csv file containing the EEG data
            epochs_length = 0, # length of epochs in seconds, 0 = no epoching
            line_noise = [], # frequencies for line noise removal, empty list = no line noise removal
            sfreq = 500 # sampling frequency in Hz
    ):
        ## Set montage based on channel names and locations provided in Van Dijk et al., (2022) (Copied from Anne van Duijvenbode)
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                'eog', 'eog', 'eog', 'eog', 'ecg', 'eog', 'emg']

        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', \
                    'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', \
                    'OrbOcc', 'Mass']

        dict_ch_pos_array = {'Fp1' : np.array([-0.02681, 0.08406, -0.01056]),
                    'Fp2' : np.array([0.02941, 0.08374, -0.01004]),
                    'F7'  : np.array([-0.06699, 0.04169, -0.01596]),
                    'F3'  : np.array([-0.04805, 0.05187, 0.03987]),
                    'Fz'  : np.array([0.00090, 0.05701, 0.06636]),
                    'F4'  : np.array([0.05038, 0.05184, 0.04133]),
                    'F8'  : np.array([0.06871, 0.04116, -0.01531]),
                    'FC3' : np.array([-0.05883, 0.02102, 0.05482]),
                    'FCz' : np.array([0.00057, 0.02463, 0.08763]),
                    'FC4' : np.array([0.06029, 0.02116, 0.05558]), 
                    'T7'  : np.array([-0.08336, -0.01652, -0.01265]), 
                    'C3'  : np.array([-0.06557, -0.01325, 0.06498]),
                    'Cz'  : np.array([0.000023, -0.01128, 0.09981]),
                    'C4'  : np.array([0.06650, -0.01280, 0.06511]),
                    'T8'  : np.array([0.08444, -0.01665, -0.01179]), 
                    'CP3' : np.array([-0.06551, -0.04848, 0.06857]),
                    'CPz' : np.array([-0.0042, -0.04877, 0.09837]), 
                    'CP4' : np.array([0.06503, -0.04835, 0.06857]), 
                    'P7'  : np.array([-0.07146, -0.07517, -0.00370]), 
                    'P3'  : np.array([-0.05507, -0.08011, 0.05944]), 
                    'Pz'  : np.array([-0.00087, -0.08223, 0.08243]),
                    'P4'  : np.array([0.05351, -0.08013, 0.05940]), 
                    'P8'  : np.array([0.07110, -0.07517, -0.00369]), 
                    'O1'  : np.array([-0.02898, -0.11452, 0.00967]),  
                    'Oz'  : np.array([-0.00141, -0.11779, 0.01584]),
                    'O2'  : np.array([0.02689, -0.11468, 0.00945])
                    }
        montage = mne.channels.make_dig_montage(ch_pos = dict_ch_pos_array, coord_frame = 'head')

        # create info object for MNE
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        info.set_montage(montage=montage, on_missing= 'raise')
        
        # read raw EEG data
        eeg_data = pd.read_csv(filename, sep=',')
        eeg_data = eeg_data.transpose().to_numpy() # transpose data because MNE expects one channel per row instead of per column
        raw = mne.io.RawArray(eeg_data, info) # load data as MNE object, with the previously created 'info'
        print('\n', 'RAW DATA LOADED', '\n')

        # define PREP parameters
        prep_params = {
            "ref_chs": "eeg", # channels to be used for rereferencing
            "reref_chs": "eeg", # channels from which reference signal will be subtracted
            "line_freqs": line_noise, # frequencies for line noise removal
        }
        prep = PrepPipeline(raw, prep_params, montage) # documentation: https://pyprep.readthedocs.io/en/latest/_modules/pyprep/prep_pipeline.html#PrepPipeline
        print('\n', "'PREP' OBJECT CREATED", '\n')

        # run/fit PREP, applying the following steps to the raw data:
        #   1. 1Hz high pass filtering
        #   2. removing line noise
        #   3. rereferencing
        #   4. detect and interpolate bad channels
        prep.fit()
        print('\n', "'PREP' PIPELINE APPLIED", '\n')

        # store preprocessed data & bad channels as attributes
        self.prep_data = prep.raw
        self.bad_channels_original = prep.noisy_channels_original # (dict) Detailed bad channels in each criteria before robust reference.
        self.bad_channels_before_interpolation = prep.noisy_channels_before_interpolation # (dict) Detailed bad channels in each criteria just before interpolation.
        self.bad_channels_after_interpolation = prep.noisy_channels_after_interpolation # (dict) Detailed bad channels in each criteria just after interpolation.
        self.still_bad_channels = prep.still_noisy_channels # (list) Names of the noisy channels after interpolation.

        raw = prep.raw

        ## Repairing EOG, ECG, and EMG artifacts with ICA
        # create a copy of the raw data and apply a low pass filter to remove 
        # slow drifts for better ICA fitting. The fitted ICA can be applied to 
        # the unfiltered signal, because filtering is a linear operation.
        filt_raw = raw.copy().filter(l_freq=1, h_freq=None)

        # creating & fitting ICA object
        ica = ICA(n_components=15, method='picard', max_iter="auto") # n PCA components
        ica.fit(filt_raw)
        print('\n', "ICA FITTED", '\n')

        # automatically detect ICs that best capture EOG signal
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        print('\n', "EOG ARTIFACTS DETECTED", '\n')

        # automatically detect ICs that best capture ECG signal
        ica.exclude = []
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
        print('\n', "ECG ARTIFACTS DETECTED", '\n')

        # automatically detect ICs that best capture muscle artifacts
        ica.exclude = []
        emg_indices, emg_scores = ica.find_bads_muscle(raw)
        print('\n', "EMG ARTIFACTS DETECTED", '\n')

        # repair all artifacts with ICA
        ica.apply(raw, exclude = eog_indices + ecg_indices + emg_indices)
        print('\n', "ICA APPLIED", '\n')

        ## Applying high-pass & low-pass filter
        raw.filter(l_freq=1, h_freq=100)

        self.preprocessed_raw = raw # Raw object with PREP preprocessing and ICA applied

        ## Epoching the data
        if epochs_length > 0:
            self.preprocessed_epochs = mne.make_fixed_length_epochs(raw, duration = epochs_length, overlap = 0) # Epochs object with PREP preprocessing and ICA applied
        else:
            self.preprocessed_epochs = 'No epoching applied'
        print('\n', "EPOCHING APPLIED", '\n')