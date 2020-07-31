import pyaudio  # import necessary libraries for audio recording and storage
import wave
from array import array
import numpy as np
import librosa
import soundfile
from sklearn.neural_network import MLPClassifier
import os

sound = {'00': 'click_failed','01': 'click'}


def record():
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
        if vol >= 4000:
            check_state = True
            frames.append(data)
            print("something said")
        elif check_state == True and vol < 4000:
            break
        else:
            check_state = False
            print("nothing")
        print("\n")

    sample_width = audio.get_sample_size(FORMAT)
    return sample_width,frames,CHANNELS,RATE


def file_writer(FILE_NAME):
    """retrieves the sample width, array of audio data, and record settings
    as they are returned from record(). The function uses that information to write
    a wav file with that data"""
    sample_width,frames,CHANNELS,RATE = record()
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
        if chroma:
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


def build_dataset(i):
    """creates two numPy arrays, containing the features and label of an audio
    recording, respectively."""
    y = np.array([])
    FILE_NAME = "recording%d.wav" % i
    file_writer(FILE_NAME)
    X = extract_features(FILE_NAME,mfcc = True,chroma = True,mel = True)
    y = np.append(y,sound['01'])
    os.remove(FILE_NAME)
    return X,y


def get_training_data(samples):
    """This function returns the complete arrays of observations and
    labels, respectively."""
    observations = np.empty((0,180))
    labels = np.empty((0,1))
    for i in range(samples):
        X,y = build_dataset(i)
        examples,target = np.array([X]),np.array([y])
        observations = np.append(observations,examples,axis = 0)
        labels = np.append(labels,target,axis = 0)
    observations.reshape(1,-1),labels.reshape(1,-1)
    return observations,labels


def get_sample(activation):
    X_test = np.empty((0,180))
    filename = "test%d.wav" % activation
    file_writer(filename)
    new_sample = extract_features(filename,mfcc = True,chroma = True,mel = True)
    inference = np.array([new_sample])
    X_test = np.append(X_test, inference, axis = 0)
    os.remove(filename)
    return X_test


def deploy_model(test_size):
    clf = MLPClassifier(random_state = 1, batch_size = 100)
    X_train,y_train = get_training_data(test_size)
    clf.fit(X_train,y_train.ravel())
    activations = 0
    while True:
        X_predict = get_sample(activations)
        if clf.predict(X_predict) == sound['01']:
            print('Click!')
            # generate_click
        else:
            print('not click:(')
        print("\n")
        activations += 1

    stream.stop_stream()
    stream.close()
    audio.terminate()

