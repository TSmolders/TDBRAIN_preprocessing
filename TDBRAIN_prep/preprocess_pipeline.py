"""
Created on Thu Mar 7 2024

author: T.Smolders

description: automated pipeline for preprocessing of the TDBRAIN dataset

name: preprocess_pipeline.py

version: 1.1

"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from preprocessing import Preproccesing
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close

# initialize counter
count = 1

def preprocess_pipeline(params):
    """
    Pipeline to preprocess EEG data from the TDBRAIN dataset.

    Requires params dictionary with the following keys (initialized in __main__):
    - derivatives_dir: path to the directory containing the \derivatives\ folder
    - preprocessed_dir: path to the directory where the preprocessed data will be saved
    - condition: list of conditions to be preprocessed
    - sessions: list of sessions to be preprocessed
    - epochs_length: length of epochs in seconds, 0 = no epoching
    - sfreq: sampling frequency in Hz
    - line_noise: list of frequencies for line noise removal, empty list = no line noise removal

    Output:
    - preprocessed data object saved as .npy file
    - plots for each preprocessing step saved in a .pdf file
    """
    # if parameters are not defined, set them to values which will not affect the preprocessing
    if not 'epochs_length' in params:
        params['epochs_length'] = 0
    if not 'line_noise' in params:
        params['line_noise'] = []

    derivates_dir = params['derivatives_dir']
    sessions = params['sessions']
    conditions = params['condition']
    preprocessed_dir = params['preprocessed_dir']
    epochs_length = params['epochs_length']
    line_noise = params['line_noise']
    sfreq = params['sfreq']

    print(line_noise, epochs_length)

    # calculate total number of files if count = 1 to show progress
    if count == 1:
        total_files = 0
        for _, _, filenames in os.walk(derivates_dir):
            total_files += len(filenames)

    for subdir, params, files in os.walk(derivates_dir): # iterate through all files
        for file in files:
            if '.csv' in file:
                if any(session in file for session in sessions) & any(condition in file for condition in conditions):
                    filepath = os.path.join(subdir, file)

                    # split file name to obtain ID, session number, and condition
                    ID = str(file.split('_')[0])
                    sessID = str(file.split('_')[1])
                    cond = str(file.split('_')[2])

                    print(f'\n [INFO]: processing subject: {ID}, session: {sessID}, condition: {cond}')
                    print(f'[INFO]: file {count} of {total_files} \n')

                    
                    # create Preprocessing object
                    preprocessed_data = Preproccesing(
                        filepath, 
                        epochs_length, 
                        line_noise, 
                        sfreq
                        )
                    
                    # define directory and subdirectories for preprocessed data
                    save_dir = f'{preprocessed_dir}/{ID}/{sessID}/eeg'
                    print(f'{save_dir = }')
                    save_path_data = f'{save_dir}/{ID}_{sessID}_{cond}_preprocessed.npy'
                    print(f'{save_path_data = }')
                    save_path_plots = f'{save_dir}/{ID}_{sessID}_{cond}_preprocessing_plots.pdf'
                    print(f'{save_path_plots = }')

                    # create directory if it does not exist
                    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)

                    # save plots in pdf file
                    figs = preprocessed_data.figs
                    with PdfPages(save_path_plots) as pdf:
                        for fig in figs:
                            pdf.savefig(fig)
                    close('all') # closes all figures for memory management

                    # store preprocessed data object as .npy file
                    delattr(preprocessed_data, 'figs') # delete figs attribute, because cannot be pickled
                    with open(save_path_data, 'wb') as output:
                        pickle.dump(preprocessed_data, output, pickle.HIGHEST_PROTOCOL)
                    print(f'\n [INFO]: preprocessed data object saved to: {save_path_data} \n')


if __name__ == '__main__':
    print(f'Root directory: {sys.argv[1]}')
    params = {}

    params['derivatives_dir'] = sys.argv[1] + '/derivatives/'
    print(f'Reading data from: {params["derivatives_dir"]}')

    params['preprocessed_dir'] = sys.argv[1] + '/preprocessed_tdbrain_prep'
    print(f'Writing preprocessed data to: {params["preprocessed_dir"]}')

    # the following parameters can be changed by the user
    params['condition'] = ['EO', 'EC'] # conditions to be preprocessed
    params['sessions'] = ['ses-1', 'ses-2'] # sessions to be preprocessed
    params['epochs_length'] = 9.95 # length of epochs in seconds, comment out for no epoching
    params['sfreq'] = 500 # sampling frequency
    params['line_noise'] = np.arange(50, params['sfreq'] / 2, 50) # 50 Hz line noise removal, comment out for no line noise removal

    preprocess_pipeline(params)