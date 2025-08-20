import torch
import pickle
import pandas as pd
import careless_whisper_stream
import careless_whisper_stream.tokenizer
from praatio import textgrid
from dataclasses import dataclass
from careless_whisper_stream.tokenizer import Tokenizer
from careless_whisper_stream.audio import SpectrogramStream

class WAVsDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path: str, 
                 sep="\t", 
                 tokenizer: careless_whisper_stream.tokenizer = None, 
                 no_labels: bool = False, 
                 custom_len: int = 0,
                 get_streamed_mel: bool = False) -> None:
        super().__init__()

        if not no_labels:
            self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        self.ds_df = pd.read_csv(ds_path, sep=sep)
        self.sr = 16_000
        self.no_labels = no_labels
        self.custom_len = custom_len
        self.get_streamed_mel = get_streamed_mel

    def __len__(self):
        return int(self.custom_len) if 0 < self.custom_len < len(self.ds_df) else int(len(self.ds_df))

    def _calc_mel(self, audio):
        if self.get_streamed_mel:
            spec_streamer = SpectrogramStream()
            return spec_streamer._simulate_streaming_log_spec(torch.tensor(audio)).squeeze(0)
            
        return careless_whisper_stream.log_mel_spectrogram(audio)

    def __getitem__(self, idx):
        item = self.ds_df.iloc[idx]

        audio = careless_whisper_stream.load_audio(item["wav_path"], sr=self.sr)
        audio = careless_whisper_stream.pad_or_trim(audio.flatten())
        mel = self._calc_mel(audio)
        
        if self.no_labels: return dict(input_ids=mel)
        
        text = item["raw_text"]
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return dict(input_ids=mel, labels=labels, dec_input_ids=text)


@dataclass
class Interval:
    label: str = None
    start: float = 0.0
    end: float = 0.0

class AlignedTextGridDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ds_path: str, 
                 tokenizer: Tokenizer = None, 
                 sample_rate: int = 16_000, 
                 custom_len: int = 0,
                 get_streamed_mel: bool = False,
                 gran: int = 15,
                 extra_gran_blocks: int = 0,
                 n_mels: int = 80,
                 multilingual: bool = False): # most of the times we train just on english librispeech
        super().__init__()

        self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        self.ds_df = pd.read_csv(ds_path)
        self.sr = sample_rate
        self.custom_len = custom_len
        self.get_streamed_mel = get_streamed_mel
        self.gran = gran
        self.extra_gran_blocks = extra_gran_blocks
        self.n_mels = n_mels
        self.multilingual = multilingual

    def __len__(self):
        return int(self.custom_len) if 0 < self.custom_len < len(self.ds_df) else len(self.ds_df)
    
    def _calc_mel(self, audio):
        if self.get_streamed_mel:
            spec_streamer = SpectrogramStream(n_mels=self.n_mels)
            return spec_streamer._simulate_streaming_log_spec(torch.tensor(audio))
            
        return careless_whisper_stream.log_mel_spectrogram(audio)

    def _get_intervals_from_wrd_file(self, path: str):
        with open(path, "r") as file:
            lines = file.readlines()
        
        intervals = []
        for line in lines:
            start, end, label = line.strip().split()
            intervals.append(Interval(label, int(start) / self.sr, int(end) / self.sr))
        
        return intervals

    def __getitem__(self, index):
        item = self.ds_df.iloc[index]

        audio = careless_whisper_stream.pad_or_trim(careless_whisper_stream.load_audio(item["wav_path"], sr=self.sr))
        mel = self._calc_mel(audio)
        
        if ".wrd" in item["tg_path"]:
            text_intervals = self._get_intervals_from_wrd_file(item["tg_path"])            
        else:
            tg = textgrid.openTextgrid(item["tg_path"], includeEmptyIntervals=False)
            text_intervals = tg.getTier("words")

        tokenizer = self.tokenizer if not self.multilingual else careless_whisper_stream.tokenizer.get_tokenizer(True, language=item["lang"], task="transcribe")

        endpoints = [0, 0, 0]
        tokens = []
        for i, interval in enumerate(text_intervals):
            curr_tokens = self.tokenizer.encode(interval.label if i == 0 else " " + interval.label)
            n_diff = (interval.end - interval.start) / len(curr_tokens)
            endpoints.extend([interval.start + (i + 1) * n_diff for i in range(len(curr_tokens))])
            tokens.extend(curr_tokens)
        
        text = [*tokenizer.sot_sequence_including_notimestamps] + tokens
        labels = text[1:] + [self.tokenizer.eot]
        endpoints.append(endpoints[-1] + 0.5)
        
        assert len(endpoints) == len(labels) == len(text)

        return dict(input_ids=mel,
                    dec_input_ids=torch.tensor(text),
                    labels=torch.tensor(labels),
                    endpoints=torch.tensor(endpoints))


class TIMIT(torch.utils.data.Dataset):
    def __init__(self, ds_path: str, tokenizer: Tokenizer = None, n_state: int = 384) -> None:
                
        self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")

        with open(ds_path, 'rb') as file:
            self.dataset = pickle.load(file)

        self.n_state = n_state
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        audio, sr, text, _, _ = self.dataset[index]
        audio_len = audio.shape[-1]
        assert sr == 16000
        audio = careless_whisper_stream.pad_or_trim(torch.Tensor(audio).flatten())
        mel = careless_whisper_stream.log_mel_spectrogram(audio)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        num_frames = ((audio_len // 16000) * 50) + 2
        mask = torch.ones(1, 1500, self.n_state)
        mask[0, num_frames:, :] = 0

        return dict(
            input_ids=mel,
            labels=labels,
            dec_input_ids=text,
            mask=mask,
        )
