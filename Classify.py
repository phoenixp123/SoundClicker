import Listen
import pyaudio  # import necessary libraries for audio recording and storage
import numpy as np
from sklearn.neural_network import MLPClassifier

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 15

sound = {
    '00': 'nothing',
    '01': 'click'
}

model = MLPClassifier(alpha = 0.01,batch_size = 256,epsilon = 1e-08,hidden_layer_sizes = (300,),
                      learning_rate = 'adaptive',max_iter = 500)


def deploy_model(test_size,state):
    observations,label = Listen.get_training_data(test_size)
    model.fit(observations,label)
    audio = pyaudio.PyAudio()
    stream = audio.open(format = FORMAT,channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = CHUNK)
    X = []
    iteration = 0
    while state:
        Listen.file_writer("new_sample%d" % iteration)
        new_input_features = Listen.extract_features("new_sample")
        predictive_sample = X.append(new_input_features)
        click_state = model.predict(np.array(predictive_sample))
        if click_state == sound['01']:
            print("click")
            # generate click
        elif click_state == sound['00']:
            print("not a click")
            continue
        iteration += 1
    stream.stop_stream()
    stream.close()
    audio.terminate()


def main():





























