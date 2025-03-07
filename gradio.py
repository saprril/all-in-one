import torch
from IPython.display import Audio
#from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from src.allin1.models import load_pretrained_model
from src.allin1.postprocessing import (postprocess_metrical_structure,postprocess_functional_structure, estimate_tempofrom_beats, )

import gradio as gr
import io

from torchaudio.transforms import Fade

def waveform_to_buffer(waveform, sample_rate):
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)  # Reset buffer position
    return buffer

def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    _, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()

def segmentate(filepath):
  # output = pipeline(filepath)
  # print(pipeline)
  # return outpu

  bundle = HDEMUCS_HIGH_MUSDB_PLUS
  model = bundle.get_model()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # return device
  model.to(device)
  sample_rate = bundle.sample_rate
  waveform, sr = torchaudio.load(filepath)

  # Load audio to GPU memory
  waveform = waveform.to(device)
  mixture = waveform

  # parameters
  segment: int = 10
  overlap = 0.1

  # Waveform Normalization
  ref = waveform.mean(0)
  waveform = (waveform - ref.mean()) / ref.std()  # normalization

  # Inference function in action
  sources = separate_sources(
    model,
    waveform[None],
    device=device,
    segment=segment,
    overlap=overlap,
  )[0]
  sources = sources * ref.std() + ref.mean()

  sources_list = model.sources
  sources = list(sources)

  # The result is a dictionary with format: {"drums":tensor(waveform.shape), "vocals":.., "bass":.., "other":...}
  audios = dict(zip(sources_list, sources))
  audios_cpu = {}

  for key in audios.keys():
    audios_cpu[key] = audios[key].to('cpu')

  buffers = {name: waveform_to_buffer(waveform, sr) for name, waveform in audios_cpu.items()}

  frames = FramedSignalProcessor(frame_size=2048, fps=int(44100 / 441))
  stft = ShortTimeFourierTransformProcessor()  # caching FFT window
  filt = FilteredSpectrogramProcessor(num_bands=12, fmin=30, fmax=17000, norm_filters=True)
  spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
  processor = SequentialProcessor([frames, stft, filt, spec])

  sig_bass = Signal(buffers['bass'], num_channels=1, sample_rate=sr)
  sig_drums = Signal(buffers["drums"], num_channels=1, sample_rate=sr)
  sig_other = Signal(buffers["other"], num_channels=1, sample_rate=sr)
  sig_vocals = Signal(buffers["vocals"], num_channels=1, sample_rate=sr)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec_all = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])
  model = load_pretrained_model(model_name="harmonix-all", device=device)
  spec_all = torchfrom_numpy(spec_all).unsqueeze(0).to(device)

  with torch.no_grad():
    logits = model(spec_all)
    functional_structure = postprocess_functional_structure(logits, model.cfg)
  
  segment_dictionary = {seg.label:(seg.start, seg.end) for seg in result.segments}

  return str(segment_dictionary)


demo = gr.Interface(
    segmentate,
    gr.Audio(sources="upload"),
    "text",
)

demo.launch()