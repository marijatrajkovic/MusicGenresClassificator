import os

ROOT_FOLDER = os.path.join('D:\\', 'MusicGenreClassificator')
DATA_PATH = os.path.join(ROOT_FOLDER, 'dataset2')
TMP_PATH = os.path.join(ROOT_FOLDER, 'tmp')
WAV_PATH = os.path.join(ROOT_FOLDER, 'data_wav')
WAV_DATA_PATTERN = os.path.join(WAV_PATH, '*/*.wav')

WINDOW_SIZE = 10000  # number of raw audio samples
LENGTH_SIZE_2D = 50176  # number of data points to form the Mel spectrogram
FEATURE_SIZE = 85210  # size of the feature vector
DATA_SIZE = (224, 224, 3)  # required data size for transfer learning
BATCH_SIZE = 1

if not os.path.exists(TMP_PATH):
    os.mkdir(TMP_PATH)
