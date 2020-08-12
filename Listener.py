import pyaudio  # import necessary libraries for audio recording and storage
import wave
from array import array
import numpy as np
import librosa
import soundfile
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, cross_val_score
import os
from pynput.mouse import Button,Controller


def record(deployed):
    """This function records incoming audio by first running for a set
    time; if the user speaks above the set volume threshold then append
    the audio data to an array called frames. If the user has already
    spoke and volume is below the threshold then end the function call. Returns
    the sample width and the array of audio data for processing
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 15

    audio = pyaudio.PyAudio()

    stream = audio.open(format = FORMAT,channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = CHUNK)

    frames = []
    check_state = False

    for _ in range(0,int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK,exception_on_overflow = False)
        data_chunk = array('h',data)
        vol = max(data_chunk)
        if vol >= 3500:
            check_state = True
            frames.append(data)
            if not deployed:
                print("something said")
        elif check_state == True and vol < 3000:
            break
        else:
            check_state = False
            if not deployed:
                print("nothing")
        if not deployed:
            print("\n")

    sample_width = audio.get_sample_size(FORMAT)
    return sample_width,frames,CHANNELS,RATE


def file_writer(FILE_NAME,deployed):
    """retrieves the sample width, array of audio data, and record settings
    as they are returned from record(). The function uses that information to write
    a wav file with that data"""
    sample_width,frames,CHANNELS,RATE = record(deployed)
    wf = wave.open(FILE_NAME,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))  # append frames recorded to file
    wf.close()


def extract_features(file_name,mfcc,chroma,mel):
    """Extracts the features for ML classification from a given audio file.
    It takes the presence of each feature as arguments such that the model
    can be tuned for feature relevance later"""
    with soundfile.SoundFile(file_name) as audio_data:
        x = audio_data.read(dtype = "float32")
        sample_rate = audio_data.samplerate
        stft = np.absolute(librosa.stft(x))
        features = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y = x,sr = sample_rate,n_mfcc = 40).T,axis = 0)
            features = np.hstack((features,mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S = stft,sr = sample_rate).T,axis = 0)
            features = np.hstack((features,chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(x,sr = sample_rate).T,axis = 0)
            features = np.hstack((features,mel))
    return features


def build_dataset(testing):
    """creates two numPy arrays, containing the features and label of an audio
    recording, respectively."""
    FILE_NAME = "recording.wav"
    file_writer(FILE_NAME,testing)
    X = extract_features(FILE_NAME,mfcc = True,chroma = False,mel = False)
    os.remove(FILE_NAME)
    return X


def get_training_data(samples,state):
    """This function returns the complete arrays of observations and
    labels, respectively."""
    observations = np.empty((0,40))
    for i in range(samples):
        X = build_dataset(state)
        examples = np.array([X])
        observations = np.append(observations,examples,axis = 0)
    observations.reshape(1,-1)
    return observations


def get_sample():
    X_test = np.empty((0,40))
    filename = "test.wav"
    file_writer(filename,deployed = True)
    new_sample = extract_features(filename,mfcc = True,chroma = False,mel = False)
    inference = np.array([new_sample])
    X_test = np.append(X_test,inference,axis = 0)
    os.remove(filename)
    return X_test


def generate_data(test_size):
    synthetic_data = np.empty((0,40))
    X_train = get_training_data(test_size,False)
    for _ in range(1000):
        # create dataset with a degree of variation
        synthetic_data = np.append(synthetic_data, X_train, axis = 0)
    return synthetic_data


def deploy_model(test_size):
    mouse = Controller()
    clf = OneClassSVM(kernel = 'rbf')
    X = generate_data(test_size)
    clf.fit(X)
    while True:
        X_predict = get_sample()
        if clf.predict(X_predict) == 1:
            print('Click!')
            mouse.click(Button.left,1)
        else:
            print("failed")
        print("\n")
