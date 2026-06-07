import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF

from config import AUDIO

def load_audio(path: str) -> torch.Tensor:

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != AUDIO.sample_rate:
        waveform = AF.resample(waveform, sr, AUDIO.sample_rate)
    return waveform.squeeze(0)


def amplitude_normalize(waveform: torch.Tensor) -> torch.Tensor:

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def preemphasis(waveform: torch.Tensor,
                coef: float = AUDIO.preemphasis) -> torch.Tensor:

    return torch.cat([waveform[:1], waveform[1:] - coef * waveform[:-1]])


def fix_length(waveform: torch.Tensor,
               max_samples: int = AUDIO.max_samples) -> torch.Tensor:

    if waveform.numel() >= max_samples:
        return waveform[:max_samples]
    pad = max_samples - waveform.numel()
    return torch.nn.functional.pad(waveform, (0, pad))

class LogMelExtractor(nn.Module):


    def __init__(self, cfg=AUDIO):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            window_fn=torch.hamming_window,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)
        log_mel = self.to_db(mel)
        return log_mel


class WaveAugment:

    def __init__(self, sample_rate=AUDIO.sample_rate, p=0.5):
        self.sr = sample_rate
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.p:
            snr_db = np.random.uniform(15, 30)
            sig_power = waveform.pow(2).mean()
            noise_power = sig_power / (10 ** (snr_db / 10))
            waveform = waveform + torch.randn_like(waveform) * noise_power.sqrt()

        if np.random.rand() < self.p:
            n_steps = float(np.random.uniform(-2, 2))
            try:
                waveform = AF.pitch_shift(
                    waveform.unsqueeze(0), self.sr, n_steps
                ).squeeze(0)
            except Exception:

                pass
        return waveform


class SpecAugment(nn.Module):


    def __init__(self, freq_mask=24, time_mask=40, n_freq=2, n_time=2):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask) for _ in range(n_freq)])
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask) for _ in range(n_time)])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        for m in self.freq_masks:
            spec = m(spec)
        for m in self.time_masks:
            spec = m(spec)
        return spec


def waveform_to_logmel(path: str,
                       augment_wave: WaveAugment = None) -> torch.Tensor:
    wav = load_audio(path)
    wav = amplitude_normalize(wav)
    wav = preemphasis(wav)
    if augment_wave is not None:
        wav = augment_wave(wav)
    wav = fix_length(wav)
    extractor = LogMelExtractor()
    log_mel = extractor(wav)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.unsqueeze(0)
