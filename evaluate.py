import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ARTIFACTS, EMOTIONS, EMOTION_RU, TRAIN, AUDIO, LABEL2ID
from dataset import (build_dusha_dataframes, read_resd, SERDataset)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_predictions_cnn(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            logits = model(x.to(DEVICE))
            y_true.extend(y.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
    return np.array(y_true), np.array(y_pred)


def get_predictions_hubert(model_dir, df):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    from preprocessing import load_audio, fix_length
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModelForAudioClassification.from_pretrained(model_dir).to(DEVICE).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            wav = fix_length(load_audio(row["audio_path"])).numpy()
            inp = fe(wav, sampling_rate=AUDIO.sample_rate,
                     max_length=AUDIO.max_samples, truncation=True,
                     padding="max_length", return_tensors="pt")
            logits = model(inp["input_values"].to(DEVICE)).logits
            y_true.append(LABEL2ID[row["label"]])
            y_pred.append(int(logits.argmax(1).cpu()))
    return np.array(y_true), np.array(y_pred)


def report(y_true, y_pred, tag):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"\n=== {tag} ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro-F1    : {macro:.4f}")
    print(f"Weighted-F1 : {weighted:.4f}")
    print(classification_report(
        y_true, y_pred, target_names=[EMOTION_RU[e] for e in EMOTIONS],
        digits=3, zero_division=0))
    from confusion_plots import plot_all
    plot_all(y_true, y_pred, [EMOTION_RU[e] for e in EMOTIONS], tag, top_n=10)
    return {"tag": tag, "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def main(model_name, weights, testset):
    _, _, test_df = build_dusha_dataframes()
    df = read_resd() if testset == "resd" else test_df

    if model_name == "hubert":
        y_true, y_pred = get_predictions_hubert(weights, df)
    else:
        from models import build_model
        model = build_model(model_name).to(DEVICE)
        model.load_state_dict(torch.load(weights, map_location=DEVICE))
        loader = DataLoader(SERDataset(df, mode="logmel"),
                            batch_size=TRAIN.batch_size)
        y_true, y_pred = get_predictions_cnn(model, loader)

    res = report(y_true, y_pred, f"{model_name}_{testset}")
    pd.DataFrame([res]).to_csv(
        ARTIFACTS / f"metrics_{model_name}_{testset}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["cnn2d", "cnn_bilstm", "hubert"])
    p.add_argument("--weights", required=True)
    p.add_argument("--testset", default="dusha", choices=["dusha", "resd"])
    a = p.parse_args()
    main(a.model, a.weights, a.testset)
