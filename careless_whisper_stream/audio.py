import os
import time
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import wave
import torch
import pyaudio
import numpy as np
import soundfile as sf
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


class MyStream:
    def __init__(self, 
                 ms_gran: int = 200, 
                 sample_rate: int = 16000, 
                 channels: int = 2, 
                 filename: str = None, 
                 inp_dtype: any = pyaudio.paInt16, 
                 simulate_stream: bool = False,
                 wav_file: str = None,
                 relay: bool = False,
                 use_latency: bool = False,
                 pad_trim: bool = True,
                 use_remote_machine: bool = False):

        assert ms_gran % 20 == 0, "ms_gran must be a multiple of 20"

        self.ms_gran = ms_gran
        self.sample_rate = sample_rate
        self.channels = channels
        self.inp_dtype = inp_dtype
        self.relay = relay
        self.use_latency = use_latency
        self.use_remote_machine = use_remote_machine

        rate_fraction = ms_gran / 1000
        self.chunk_size = int(rate_fraction * sample_rate)
        self.filename = filename
        self.streamed_wav_file = wav_file

        self.simulate_stream = simulate_stream
        if self.simulate_stream:
            assert wav_file is not None, "when simulating stream a wav file must be provided."
            if pad_trim:
                self.wav_array = pad_or_trim(load_audio(wav_file, sample_rate), length=N_SAMPLES+180) # wav array
            else:
                audio = load_audio(wav_file, sample_rate)
                self.wav_array = pad_or_trim(audio, length=audio.shape[-1]+180)
                print(f"{self.wav_array.shape=}")
    
    def _simulate_stream_using_wav(self):
        print("Streaming simulation of a wav started...")

        for i in range(self.wav_array.shape[-1] // self.chunk_size):
            if i == 0:
                yield self.wav_array[..., :(((i + 1) * self.chunk_size) + 40 + 320)] # 320 is extra 20 msec buffer we need!  
            else:
                yield self.wav_array[..., ((i * self.chunk_size) + 40 + 320):(((i + 1) * self.chunk_size) + 40 + 320)]
            
            if self.use_latency: time.sleep(self.ms_gran / 1000) # simulating the latency between audio chunks

    def open_stream(self):
        if self.simulate_stream or self.relay or self.use_remote_machine: return

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(input=True, format=self.inp_dtype, channels=self.channels, rate=self.sample_rate, frames_per_buffer=self.chunk_size)
    
    def _read_from_stream(self):
        print("Streaming instance recording started...")
        
        while True:
            yield self.stream.read(self.chunk_size)

    def _follow_growing_wav(self):
        while not os.path.exists(self.streamed_wav_file):
            time.sleep(0.1)

        with sf.SoundFile(self.streamed_wav_file, mode='r') as f:
            while True:
                block = f.read(self.chunk_size)
                if len(block) == 0:
                    time.sleep(self.ms_gran / 1000)  # Wait for more data
                    continue
                yield block

    def _read_raw_pcm(self):
        samples_per_chunk = int(self.sample_rate * (self.ms_gran / 1000))
        bytes_per_sample = 2  # s16le = 16 bits = 2 bytes
        chunk_size = samples_per_chunk * bytes_per_sample

        while not os.path.exists(self.streamed_wav_file):
            time.sleep(0.1)

        with open(self.streamed_wav_file, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    time.sleep((self.ms_gran / 1000))
                    continue
                yield np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

    def read(self):
        if self.simulate_stream:
            return self._simulate_stream_using_wav()
        
        if self.use_remote_machine:
            return self._read_raw_pcm()

        return self._read_from_stream()

    def _save_recording_file(self, frames: list):
        print(f"Saving recorded audio file on path {self.filename}")
        
        waveFile = wave.open(self.filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.inp_dtype))
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    def close_stream(self, frames: list):
        if self.simulate_stream: return
        
        # Stop Recording
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        print("Finished recording, stream and audio terminated.")

        if self.filename: self._save_recording_file(frames)


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
        
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class SpectrogramStream:
    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = 80, window: Optional[str] = "hann", pad_mode: str = "reflect"):

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_mels = n_mels

        self.window = torch.hann_window(n_fft)
        self.window_type = window

        self.ctx_samples = self.n_fft - self.hop_length

        self.reset()

    def reset(self):
        self.is_first = True
        self.audio_ctx = torch.tensor([])
        self.log_spec_max = -torch.inf

    def calc_mel_with_new_frame(self, audio_frame: torch.Tensor, is_last: bool = False):
        
        self.window = self.window.to(audio_frame.device)
        
        if len(audio_frame.shape) == 1:
            audio_frame = audio_frame.unsqueeze(0)

        n_batch = audio_frame.shape[0]
        
        if isinstance(self.log_spec_max, float):
            self.log_spec_max = torch.ones((n_batch)).to(audio_frame.device) * -torch.inf

        # check if we are on first frame, if so, pad using reflection
        if self.is_first:
            pad = int(self.n_fft // 2) + 1
            audio_input = F.pad(audio_frame, [pad, 0], self.pad_mode)
            self.is_first = False
        else: # pad with previous context
            audio_input = torch.cat([self.audio_ctx[..., -self.ctx_samples:], audio_frame], dim=-1)
        
        if is_last: # pad reflect last frame
            pad = int(self.n_fft // 4) + 1
            audio_input = F.pad(audio_input, [pad, 0], self.pad_mode)

        self.audio_ctx = audio_frame # now audio ctx is the last frame

        stft = torch.stft(audio_input, self.n_fft, self.hop_length, window=self.window, return_complex=True, center=False)
        magnitudes = stft.abs() ** 2
        filters = mel_filters(audio_frame.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10() # from shape (b, n_mels, audio_frames)
        self.log_spec_max = torch.maximum(log_spec.view(n_batch, -1).max(dim=-1).values, self.log_spec_max).to(log_spec.device)
        
        log_spec = torch.maximum(log_spec.view(n_batch, -1).permute(1, 0), self.log_spec_max - 8.0).permute(1, 0).view(n_batch, self.n_mels, -1)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _simulate_streaming_log_spec(self, audio: torch.Tensor, ms_gran: int = 300, total_frames: int = 3000, get_gt: bool = False):
        self.reset()

        samples_gran = HOP_LENGTH * (ms_gran // 10)
        sub_mel_frames = int(total_frames / ms_gran) * 10
        # print(samples_gran, sub_mel_frames)
        pred_mel = torch.cat([self.calc_mel_with_new_frame(audio[..., (i * samples_gran) + (40 * int(i != 0)): ((i + 1) * samples_gran) + 40], is_last=(i == sub_mel_frames - 1)) for i in range(sub_mel_frames)], dim=-1)
        
        if get_gt: 
            gt_mel = log_mel_spectrogram(audio)
            return pred_mel, gt_mel
        
        return pred_mel
