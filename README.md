# Automatic preprocessing pipeline for the TDBRAIN dataset, using pyprep and MNE packages:
### With pyprep package according to Bidgely-Shamlo et al. 2015 (https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full):
1. Remove line-noise without committing to a filtering strategy (optional)
2. Robustly reference the signal relative to an estimate of the 'true' average reference
3. Detect and interpolate bad channels relative to this reference
	- Detection by:
		a. extreme amplitudes (deviation criterion)
		b. lack of correlation with any other channel (correlation criterion)
		c. lack of predictability by other channels (predictability/RANSAC criterion)
		d. unusual high frequency noise (noisiness criterion)
### With MNE:
4. Repair EOG, ECG, and EMG artifacts with ICA (fitting ICA to high-pass filtered copy of eeg data, but applying to unfiltered eeg data)
5. BP filter (1, 100)
6. epoch data (optional)
(7. baseline correction?) not applied yet

### preprocess_pipeline.py:
Main file for running the pipeline. Takes the sourcepath of the directory containing the 'derivatives' folder of the TDBRAIN dataset.
e.g:
```
python preprocess_pipeline.py 'D:\Documents\TD-BRAIN\TDBRAIN-dataset-derivatives\'
```


Can adjust preprocessing parameters below in the file.

### preprocessing.py:
Contains the class which actually performs the preprocessing steps

