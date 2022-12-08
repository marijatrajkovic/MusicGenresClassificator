import matplotlib.pyplot as plt

from settings import *
from os import path
import os
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
import time


def load_data(audio_dir):
    categories = []
    train_data = []
    test_data = []

    music_folders = os.listdir(audio_dir)
    for g in music_folders:  # g = genres
        path = os.path.join(audio_dir, g)
        if os.path.isdir(path):
            categories.append(g)

            cnt = 0
            files = os.listdir(path)
            for f in files:
                audio_file = os.path.join(path, f)
                if os.path.isfile(audio_file):
                    if cnt % 5 == 2:
                        test_data.append((audio_file, categories.index(g)))
                    else:
                        train_data.append((audio_file, categories.index(g)))
                    cnt += 1

    x_train = np.zeros((len(train_data), 128, 2584), dtype='uint8')  # detect spect_size
    y_train = np.zeros((len(train_data), 1), dtype='uint8')
    x_test = np.zeros((len(test_data), 128, 2584), dtype='uint8')
    y_test = np.zeros((len(test_data), 1), dtype='uint8')

    for i in range(len(train_data)):
        file, g = train_data[i]
        y_train[i] = g
        spec = get_melspectrogram_db(file)
        x_train[i] = spec

    for i in range(len(test_data)):
        file, g = test_data[i]
        y_test[i] = g
        spec = get_melspectrogram_db(file)
        x_test[i] = spec

    rand_train_idx = np.random.RandomState(seed=0).permutation(len(train_data))
    x_train = x_train[rand_train_idx]
    y_train = y_train[rand_train_idx]

    rand_test_idx = np.random.RandomState(seed=0).permutation(len(test_data))
    x_test = x_test[rand_test_idx]
    y_test = y_test[rand_test_idx]

    return categories, train_data, test_data, x_train, y_train, x_test, y_test

def get_melspectrogram_db(file_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    try:
        # dst = os.path.join(file_path[:-3] + 'wav')

        # Convert mp3 to wav
        # sound = AudioSegment.from_mp3(file_path)
        # sound.export(dst, format="wav")

        # if (os.path.exists(dst) == False):
            # print("Path does not exists: " + dst)

        # wav,sr = librosa.load(dst, sr=sr)
        wav, sr = librosa.load(file_path, sr=sr)

        if wav.shape[0]<30*sr:
            wav=np.pad(wav,int(np.ceil((30*sr-wav.shape[0])/2)),mode='reflect')
        else:
            wav=wav[:30*sr]

        spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec_db)
        plt.show()
        print(spec_db.shape)

        #Remove wav file
        # os.remove(dst)
        return spec_db

    except Exception as ex:
            print (ex)
            return -1
            pass
    #print(spec_db.shape) #224x224x3 or 224x224x1 print numpy array dimension


def convert_files_to_spectrogram(file_path):
    hl = 512
    hi = 224
    wi = 224
    # Load wav file
    y, sr = librosa.load(file_path, sr=44100)

    window = y[0:wi * hl]
    # Get spectrogram
    s = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=hi, hop_length=hl)
    # Convert it to log scale (convert to decibels)
    log_s = librosa.amplitude_to_db(s)
    # Get mfcc features
    mfcc = librosa.feature.mfcc(S=log_s)

    return mfcc


# define a function to get the label associated with a file path
def get_label(file_path):
    dir_name = os.path.basename(os.path.dirname(file_path))
    return dir_name

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled