import os
import sys
import numpy as np
import pandas as pd
import pickle
from preprocessing import Preproccesing


def preprocess_pipeline(params):

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

    for subdir, params, files in os.walk(derivates_dir): # iterate through all files
        for file in files:
            if '.csv' in file:
                if any(session in file for session in sessions) & any(condition in file for condition in conditions):
                    filepath = os.path.join(subdir, file)

                    # split file name to obtain ID, session number, and condition
                    ID = str(file.split('_')[0])
                    sessID = str(file.split('_')[1])
                    cond = str(file.split('_')[2])

                    print(f'\n [INFO]: processing subject: {ID}, session: {sessID}, condition: {cond} \n')

                    
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
                    save_path = f'{save_dir}/{ID}_{sessID}_{cond}_preprocessed.npy'
                    print(f'{save_path = }')

                    # create directory if it does not exist
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # store preprocessed data object as .npy file
                    with open(save_path, 'wb') as output:
                        pickle.dump(preprocessed_data, output, pickle.HIGHEST_PROTOCOL)
                    print(f'\n [INFO]: preprocessed data object saved to: {save_path} \n')

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