# Automatic preprocessing pipeline for the [TDBRAIN](https://doi.org/10.1038/s41597-022-01409-z) dataset, using pyprep and MNE packages:
### With [pyprep](https://github.com/sappelhoff/pyprep) package according to PREP [Bidgely-Shamlo et al. 2015](https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full):
1. Remove line-noise without committing to a filtering strategy (optional)
2. Robustly reference the signal relative to an estimate of the 'true' average reference
3. Detect and interpolate bad channels relative to this reference
	- Detection by:
		1. extreme amplitudes (deviation criterion)
		2. lack of correlation with any other channel (correlation criterion)
		3. lack of predictability by other channels (predictability/RANSAC criterion)
		4. unusual high frequency noise (noisiness criterion)
### With [MNE](https://mne.tools/stable/index.html):
4. Repair EOG, ECG, and EMG artifacts with ICA (fitting ICA to high-pass filtered copy of eeg data, but applying to unfiltered eeg data)
5. Bandpass filter (1, 100)
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

