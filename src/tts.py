import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model

fs, lang = 22050, "English"
tag = "kan-bayashi/ljspeech_tacotron2"
vocoder_tag = "ljspeech_multi_band_melgan.v2"

d = ModelDownloader()
text2speech = Text2Speech(
    **d.download_and_unpack(tag),
    device="cpu",
    # Only for Tacotron 2
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2
    speed_control_alpha=1.0,
)
text2speech.spc2wav = None  # Disable griffin-lim
# NOTE: Sometimes download is failed due to "Permission denied". That is
#   the limitation of google drive. Please retry after serveral hours.
vocoder = load_model(download_pretrained_model(vocoder_tag)).to("cpu").eval()
vocoder.remove_weight_norm()


def pronounce(x):
    with torch.no_grad():
        wav, c, *_ = text2speech(x)
        wav = vocoder.inference(c)
    return fs, wav.view(-1).cpu().numpy()