import pyaudio  # import necessary libraries for audio recording and storage
import wave
from array import array
import numpy as np
import librosa
import soundfile
from sklearn.svm import OneClassSVM
import os
from pynput.mouse import Button,Controller
from random import uniform


def record(deployed):
    """This function records incoming audio by first running for a set
    time; if the user speaks above the set volume threshold then append
    the audio data to an array called frames. If the user has already
    spoke and volume is below the threshold then end the function call. Returns
    the sample width and the array of audio data for processing.

    Accepts boolean 'deployed' that toggles displaying recording prompts
    """
    FORMAT = pyaudio.paInt16  # format the audio input
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
    for _ in range(0,int(RATE / CHUNK * RECORD_SECONDS)):  # begin recording process
        data = stream.read(CHUNK,exception_on_overflow = False)
        data_chunk = array('h',data)
        vol = max(data_chunk)
        if vol >= 3200:
            check_state = True
            frames.append(data)
            if not deployed:
                print("something said")
        elif check_state == True and vol < 2500:
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
    a wav file with that data

    Note: deployed functions the same as in record()
    """
    sample_width,frames,CHANNELS,RATE = record(deployed)
    wf = wave.open(FILE_NAME,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))  # append frames recorded to file
    wf.close()


def extract_features(file_name,mfcc):  # ,mel):
    """Extracts the features for ML classification from a given audio file.
    It takes the presence of each feature as arguments such that the model
    can be tuned for feature relevance later"""
    with soundfile.SoundFile(file_name) as audio_data:
        x = audio_data.read(dtype = "float64")  # read the file
        sample_rate = audio_data.samplerate
        x_normalized = librosa.util.normalize(x)  # normalize the clip's volume
        features = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y = x_normalized,  # extract mel frequency cepstral coefficients
                                                 sr = sample_rate,n_mfcc = 4).T,axis = 0)
            mfccs_norm = librosa.util.normalize(mfccs)
            features = np.hstack((features,mfccs_norm))  # append to NumPy array of features
        # if mel:
        # mel = np.mean(librosa.feature.melspectrogram(y = x_normalized,  # extract mel spectrogram
        # sr = sample_rate).T,axis = 0)
        # mel_norm = librosa.util.normalize(mel)
        # features = np.hstack((features,mel_norm))  # append to NumPy array of features
    return features


def build_dataset(testing):
    """creates a numPy array, containing the features of an audio
    recording"""
    FILE_NAME = "recording.wav"
    file_writer(FILE_NAME,testing)  # record a sample
    X = extract_features(FILE_NAME,mfcc = True)  # ,mel = False)  # extract the sample's features
    os.remove(FILE_NAME)
    return X


def get_training_data(samples,state):
    """returns complete arrays of n samples"""
    observations = np.empty((0,4))
    for i in range(samples):
        X_samples = build_dataset(state)  # record and extract features
        examples = np.array([X_samples])
        observations = np.append(observations,examples,axis = 0)  # append to larger 2D array of features
    observations.reshape(1,-1)
    return observations


def get_sample():
    """This function is very similar, however instead of collecting a set pool of
    samples based on the argument provided, it collects a single audio signal"""
    X_test = np.empty((0,4))
    filename = "test.wav"
    file_writer(filename,deployed = True)
    new_sample = extract_features(filename,mfcc = True)  # ,mel = True)
    inference = np.array([new_sample])
    X_test = np.append(X_test,inference,axis = 0)
    os.remove(filename)
    return X_test


def generate_data(test_size):
    """Creates a synthetic dataset with about a 20%
    difference from sample to sample generated randomly"""
    synthetic_data = np.empty((0,4))
    X_train = get_training_data(test_size,state = False)
    synthetic_data = np.append(synthetic_data,X_train,axis = 0)
    for i in range(800):
        X_random = X_train * uniform(0.8,1.2)  # randomly alter each sample ~20% from the initial sample
        synthetic_data = np.append(synthetic_data,X_random,axis = 0)
    return synthetic_data


def deploy_model(test_size):
    """Initializes a OneClassSVM for binary classification, trained on data
    collected from the generate_data() function. After the model is trained, the
    function calls get_sample() and classifies that sound either as a click or not.
    If a click is successful, generate a left click event"""
    mouse = Controller()
    clf = OneClassSVM(nu = .0001,kernel = 'rbf',gamma = 'auto',max_iter = test_size ** 2)
    X = generate_data(test_size)
    clf.fit(X)
    while True:
        X_predict = get_sample()
        if clf.predict(X_predict) == 1:
            print('Click!')
            mouse.click(Button.left,1)
            # new_trainingX = np.append(X, X_predict,axis = 0)
            # clf.fit(new_trainingX)
        else:
            print("failed")
        print("\n")


if __name__ == "__main__":
    deploy_model(5)

