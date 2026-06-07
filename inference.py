"""
inference.py — прототип системы SER (раздел 3.4).
Поддерживает оффлайн-распознавание одного файла и потоковую обработку
длинной записи скользящим окном.

Запуск:
    python inference.py --weights artifacts/checkpoints/cnn_bilstm_best.pt --audio sample.wav
    python inference.py --weights ... --audio long.wav --stream
"""
import argparse

import torch
import torch.nn.functional as F

from config import AUDIO, EMOTIONS, EMOTION_RU
from models import build_model
from preprocessing import (load_audio, amplitude_normalize, preemphasis,
                           fix_length, LogMelExtractor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTOR = LogMelExtractor()


def load_model(model_name, weights):
    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model


def predict_segment(model, wav: torch.Tensor):
    wav = fix_length(preemphasis(amplitude_normalize(wav)))
    log_mel = EXTRACTOR(wav)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    x = log_mel.unsqueeze(0).unsqueeze(0).to(DEVICE)   # [1, 1, n_mels, T]
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0).cpu()
    return {EMOTION_RU[e]: float(probs[i]) for i, e in enumerate(EMOTIONS)}


def predict_file(model, path):
    return predict_segment(model, load_audio(path))


def predict_stream(model, path, window=4.0, step=2.0):
    """Потоковая обработка: окно длительностью window c шагом step секунд."""
    wav = load_audio(path)
    win = int(window * AUDIO.sample_rate)
    hop = int(step * AUDIO.sample_rate)
    results = []
    for start in range(0, max(1, wav.numel() - win + 1), hop):
        seg = wav[start:start + win]
        t = start / AUDIO.sample_rate
        pred = predict_segment(model, seg)
        top = max(pred, key=pred.get)
        results.append((t, top, pred[top]))
        print(f"[{t:6.1f} с] {top:<12} ({pred[top]:.2f})")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn_bilstm",
                   choices=["cnn2d", "cnn_bilstm"])
    p.add_argument("--weights", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--stream", action="store_true")
    a = p.parse_args()

    model = load_model(a.model, a.weights)
    if a.stream:
        predict_stream(model, a.audio)
    else:
        pred = predict_file(model, a.audio)
        top = max(pred, key=pred.get)
        print(f"Распознанная эмоция: {top} ({pred[top]:.2f})")
        for emo, p_ in sorted(pred.items(), key=lambda kv: -kv[1]):
            print(f"  {emo:<12} {p_:.3f}")
