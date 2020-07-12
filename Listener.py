import pyaudio
import wave
from array import array
import shutil
import librosa
import numpy as np
import soundfile
import glob


FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 15

audio = pyaudio.PyAudio()

stream = audio.open(format = FORMAT,channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK)


def record():
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
        elif check_state == True and vol < 2500:
            break
        else:
            check_state = False
            print("nothing")
        print("\n")

    sample_width = audio.get_sample_size(FORMAT)
    return sample_width,frames


def file_writer(FILE_NAME):
    sample_width,frames = record()
    wf = wave.open(FILE_NAME,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))  # append frames recorded to file
    wf.close()
    shutil.move(FILE_NAME,"sound_files")


def get_sample(n):
    for i in range(n):
        FILE_NAME = "recording%d.wav" % i
        file_writer(FILE_NAME)
    stream.stop_stream()
    stream.close()
    audio.terminate()


def extract_features(file_name,mfcc,chroma,mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype = "float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y = X,sr = sample_rate,n_mfcc = 40).T,axis = 0)
            result = np.hstack((result,mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S = stft,sr = sample_rate).T,axis = 0)
            result = np.hstack((result,chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X,sr = sample_rate).T,axis = 0)
            result = np.hstack((result,mel))
    return result


def load_data():
    X = []
    for file in glob.glob('/Users/phoenix/Documents/SummerResearch2020/ListeningScript/sound_files'):
        feature = extract_features(file, mfcc=True, chroma=True, mel=True)
        X.append(feature)
        feature_matrix = np.array(X)
    return feature_matrix
















