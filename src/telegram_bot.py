import io
import time

import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import librosa
from scipy.io import wavfile
import soundfile as sf
from urllib.request import urlopen
from telebot import TeleBot, logger


from asr import asr # voice recognition API
from tts import pronounce # text-to-speech API
from chatbot import SmallTalkAgent, DumbAgent # Smalltalk API

from utils import write_result

TELEGRAM_TOKEN = open("telegram_key.txt", 'r').read()
MAX_MESSAGE_SIZE = 1000 * 50  # in bytes
MAX_MESSAGE_DURATION = 15  # in seconds
TELEGRAM_RATE = 24000

bot = TeleBot(TELEGRAM_TOKEN)
smalltalk = DumbAgent()

@bot.message_handler(commands=['start'])
def start_prompt(message):
    """Print prompt to input voice message.
    """
    reply = ' '.join((
      "Press and hold screen button with microphone picture.",
      "Say your phrase and release the button.",
    ))
    return bot.reply_to(message, reply)

@bot.message_handler(content_types=['voice'])
def echo_voice(message):
    """Voice message handler.
    """
    wav, orig_sr = load_voice(message, bot)
    ratio = len
    # recognize
    wav = librosa.resample(wav, orig_sr, asr.sampleRate())
    wav16 = (wav*(2 ** 15)).astype('int16')
    user_input = asr.stt(wav16)

    response = smalltalk.talk(user_input)

    tts_sr, wav = pronounce(response)
    play_audio(tts_sr, wav)
    wav = librosa.resample(wav, tts_sr, TELEGRAM_RATE)
    return send_voice(wav, TELEGRAM_RATE, message, bot)

def load_voice(message, bot):
    data = message.voice
    if (data.file_size > MAX_MESSAGE_SIZE) or (data.duration > MAX_MESSAGE_DURATION):
        reply = ' '.join((
          "The voice message is too big.",
          "Maximum duration: {} sec.".format(MAX_MESSAGE_DURATION),
          "Try to speak in short.",
        ))
        return bot.reply_to(message, reply)

    file_url = "https://api.telegram.org/file/bot{}/{}".format(
      bot.token,
      bot.get_file(data.file_id).file_path
    )
    try:
        wav, samplerate = sf.read(io.BytesIO(urlopen(file_url).read()))
    except RuntimeError:
        # just try again
        time.sleep(1)
        wav, samplerate = sf.read(io.BytesIO(urlopen(file_url).read()))
    return wav, samplerate


def send_voice(wav, samplerate, message, bot):
    fname = 'voice.ogg'
    sf.write(fname, wav, samplerate)
    with open(fname, 'rb') as f:
        ogg = b''.join(f.readlines())

    return bot.send_voice(message.chat.id, ogg)


def play_audio(sr, wav):
    # I bet you've seen this duct-tape before
    wav = np.multiply(wav, (2**15)).astype(np.int16)
    wavfile.write("output.wav", rate=sr, data=wav)
    sound = AudioSegment.from_wav('output.wav')
    play(sound)

if __name__ == "__main__":
    bot.polling()

