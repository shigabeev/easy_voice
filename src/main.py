import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile

from asr import recognize # voice recognition API
from tts import pronounce # text-to-speech API
from chatbot import SmallTalkAgent # Smalltalk API

from utils import write_result

# from IPython.display import display, Audio  # Uncomment To test in Jupyter Notebook

tink = AudioSegment.from_wav("data/Tink.wav")   # Sound that means we started audio recording
morse = AudioSegment.from_wav("data/Morse.wav") # Sound that means recording is over

# TODO: The sounds should be recorded and played on a frontend

def play_audio(sr, wav):
    # I bet you've seen this duct-tape before
    wav = np.multiply(wav, (2**15)).astype(np.int16)
    wavfile.write("output.wav", rate=sr, data=wav)
    sound = AudioSegment.from_wav('output.wav')
    play(sound)


if __name__ == "__main__":
    bot = SmallTalkAgent()

    for _ in range(10):
        play(tink)
        user_input = recognize(7) # 7 - is a duration of recorded sound
        play(morse)

        print(f"you: {user_input}")
        write_result(user_input, 'you')

        response = bot.talk(user_input)
        write_result(user_input, 'bot')

        print(f"bot: {response}")

        sr, wav = pronounce(response)
        play_audio(sr, wav)

