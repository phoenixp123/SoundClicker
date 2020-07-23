import pyaudio  # import necessary libraries for audio recording and storage
import wave
from array import array
import numpy as np
import librosa
import soundfile
from pathlib import Path

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 15
sound = {
    '00': 'nothing',
    '01': 'click'
}

audio = pyaudio.PyAudio()

stream = audio.open(format = FORMAT,channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK)


def record():
    """This function records incoming audio by first running for a set
    time; if the user speaks above the set volume threshold then append
    the audio data to an array called frames. If the user has already
    spoke and volume is below the threshold then end the function call. Returns
    the sample width and the array of audio data for processing
    """
    frames = []
    check_state = False

    for _ in range(0,int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data_chunk = array('h',data)
        vol = max(data_chunk)
        if vol >= 2200:
            check_state = True
            print("something said")
            frames.append(data)
        elif check_state == True and vol < 2000:
            break
        else:
            check_state = False
            print("nothing")
        print("\n")

    sample_width = audio.get_sample_size(FORMAT)
    return sample_width,frames


def file_writer(FILE_NAME):
    sample_width,frames = record()  # placeholder
    wf = wave.open(FILE_NAME,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))  # append frames recorded to file
    wf.close()


def extract_features(file_name,mfcc,chroma,mel):
    with soundfile.SoundFile(file_name) as audio_data:
        x = audio_data.read(dtype = "float32")
        sample_rate = audio_data.samplerate
        if chroma:
            stft = np.abs(librosa.stft(x))
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


def get_training_data(true_samples, false_samples):
    X,y = [],[]
    for i in range(true_samples):
        FILE_NAME = "Trecording%d.wav" % i
        file_writer(FILE_NAME)
        features = extract_features(FILE_NAME,mfcc = True,chroma = True,mel = True)
        X.append(features)
        y.append(sound['01'])
    for i in range(false_samples):
        FILE_NAME = "Frecording%d.wav" % i
        file_writer(FILE_NAME)
        features = extract_features(FILE_NAME,mfcc = True,chroma = True,mel = True)
        X.append(features)
        y.append(sound['00'])
    stream.stop_stream()
    stream.close()
    audio.terminate()
    observations,labels = np.array(X),np.array(y)
    return observations,labels

