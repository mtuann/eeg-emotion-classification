This github repository contains the code for Emotion Classification from EEG data

## Requirements
* Python 3.6

## Data processing
- Original data is in the form of .dat files (32 dat files for each subject). Each file contains 40 trials of 40 seconds each. Each trial is a 40x8064 matrix. Each row of the matrix is a channel and each column is a time point. The sampling rate is 128Hz.
```python
data[:, 0:32, 384:8064]
```
- Feature extraction is done using the following code: [EEG-Emotion-Recognition](https://github.com/shyammarjit/EEG-Emotion-Recognition/). After feature extraction, the data is in the form of a 40x16 matrix.
- Label extraction includes valence and arousal values for each trial. 
  - If the valence value is greater than 5, the label is 1, else 0
  - If the arousal value is greater than 5, the label is 1, else 0.
  - For 4 labels (1, 1), (1, 0), (0, 1), (0, 0), the data is split into 4 classes (0 - HVHA, 1 - HVLA, 2 - LVHA, 3 - LVLA) respectively.
- The data is split into train, validation and test sets. The train set contains 32 subjects, the validation set contains 8 subjects.

## Dataset
- DEAP dataset: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
