import pyaudio
import numpy as np
from deepspeech import Model as ASR

ASR_MODEL_PATH = 'data/deepspeech-0.8.1-models.pbmm'
FORMAT = pyaudio.paInt16    # 16-bit sound
CHANNELS = 1                # no stereo
RATE = 16000                # Sample rate of our ASR
CHUNK = 1024                # How many samples in a frame

audio = pyaudio.PyAudio()
asr = ASR(ASR_MODEL_PATH)

def record_audio(t):
    # TODO: this should be done on frontend.
    # t - number of seconds to record
    frames = []
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    for i in range(50):
        data = stream.read(CHUNK)  # record the sound
        frame = np.frombuffer(data, dtype=np.int16)
        frames.append(frame)  # save the sound

    stream.stop_stream()
    stream.close()
    wav = np.concatenate(frames)  # combine recording in one track
    return RATE, wav  # return sample rate for convenience

def recognize(t):
    sr, wav = record_audio(t)
    return asr.stt(wav)