import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import TRAIN, ARTIFACTS
from dataset import (make_loaders, compute_class_weights,
                     build_dusha_dataframes, warmup_feature_cache)
from models import build_model, count_parameters



DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)


torch.backends.cudnn.benchmark = True


def set_seed(seed=TRAIN.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def _make_scaler():
    if not (TRAIN.use_amp and DEVICE_TYPE == "cuda"):
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _autocast_ctx():
    enabled = TRAIN.use_amp and DEVICE_TYPE == "cuda"
    try:
        return torch.autocast(device_type=DEVICE_TYPE, enabled=enabled)
    except TypeError:
        return torch.autocast(DEVICE_TYPE, enabled=enabled)


def _zero_grad(optimizer):
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()




def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
              desc: str = ""):
    train = optimizer is not None
    model.train() if train else model.eval()

    losses, y_true, y_pred = [], [], []
    n_samples = 0
    data_time_sum = 0.0
    compute_time_sum = 0.0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    t_epoch_start = time.time()
    t_last = time.time()

    with grad_ctx:
        for x, y in tqdm(loader, leave=False, desc=desc or None):
            t_after_fetch = time.time()
            data_time_sum += t_after_fetch - t_last

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if train:
                _zero_grad(optimizer)

            with _autocast_ctx():
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            losses.append(loss.item())
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(logits.detach().argmax(1).cpu().numpy())

            n_samples += x.size(0)
            t_now = time.time()
            compute_time_sum += t_now - t_after_fetch
            t_last = t_now

    elapsed = time.time() - t_epoch_start
    if not y_true:
        return 0.0, 0.0, 0.0, {"elapsed": elapsed, "samples_per_sec": 0.0,
                               "data_time": 0.0, "compute_time": 0.0}

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    stats = {
        "elapsed": elapsed,
        "samples_per_sec": n_samples / max(elapsed, 1e-9),
        "data_time": data_time_sum,
        "compute_time": compute_time_sum,
    }
    return float(np.mean(losses)), macro_f1, acc, stats




def profile_loader(loader, n_batches: int = 30):
    if loader is None:
        print("[profile] loader is None")
        return
    print(f"[profile] прогон {n_batches} батчей для замера времени загрузки...")
    it = iter(loader)
    try:
        next(it)
    except StopIteration:
        print("[profile] loader пуст")
        return

    t0 = time.time()
    cnt = 0
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        cnt += 1
    dt = time.time() - t0
    avg = dt / max(cnt, 1)
    print(f"[profile] средн. время батча: {avg*1000:.1f} ms "
          f"(всего {cnt} батчей за {dt:.2f} с)")



def plot_curves(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    if len(history["train_loss"]) == 0:
        print("Нет данных для построения кривых обучения.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"],   label="val")
    ax[0].set_title("Функция потерь")
    ax[0].set_xlabel("Эпоха")
    ax[0].legend()

    ax[1].plot(epochs, history["train_f1"], label="train")
    ax[1].plot(epochs, history["val_f1"],   label="val")
    ax[1].set_title("Macro-F1")
    ax[1].set_xlabel("Эпоха")
    ax[1].legend()

    fig.suptitle(f"Кривые обучения: {model_name}")
    fig.tight_layout()

    out_dir = ARTIFACTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"curves_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Кривые обучения сохранены: {out}")



def main(model_name: str, do_warmup: bool, do_profile: bool):
    set_seed()
    print(f"Устройство: {DEVICE}")

    if do_warmup:
        print("[warmup] прогрев кэша logmel-признаков...")
        tr_df, va_df, te_df = build_dusha_dataframes()
        all_df = pd.concat([tr_df, va_df, te_df], ignore_index=True)
        warmup_feature_cache(all_df, desc="features warmup")
        print("[warmup] готово.\n")

    loaders, (train_df, val_df, test_df) = make_loaders(mode="logmel")

    if loaders["train"] is None:
        raise RuntimeError("Train loader пуст — нечего обучать.")
    if loaders["val"] is None:
        raise RuntimeError("Val loader пуст — нечего валидировать. "
                           "Уменьшите test_size или проверьте данные.")

    if do_profile:
        profile_loader(loaders["train"], n_batches=30)

    model = build_model(model_name).to(DEVICE)
    print(f"Модель {model_name}: {count_parameters(model):,} параметров")

    weights = (compute_class_weights(train_df).to(DEVICE)
               if TRAIN.use_class_weights else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=TRAIN.lr,
                      weight_decay=TRAIN.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN.num_epochs)
    scaler = _make_scaler()

    ckpt_dir = ARTIFACTS / "checkpoints"
    fig_dir  = ARTIFACTS / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{model_name}_best.pt"

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1, patience = 0.0, 0

    for epoch in range(1, TRAIN.num_epochs + 1):
        tr_loss, tr_f1, tr_acc, tr_st = run_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            desc=f"train e{epoch}")
        va_loss, va_f1, va_acc, va_st = run_epoch(
            model, loaders["val"], criterion,
            desc=f"val e{epoch}")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        print(
            f"Эпоха {epoch:02d} | "
            f"train loss {tr_loss:.3f} F1 {tr_f1:.3f} acc {tr_acc:.3f} "
            f"({tr_st['samples_per_sec']:.0f} smp/s, "
            f"data {tr_st['data_time']:.1f}s / compute {tr_st['compute_time']:.1f}s) | "
            f"val loss {va_loss:.3f} F1 {va_f1:.3f} acc {va_acc:.3f} "
            f"({va_st['samples_per_sec']:.0f} smp/s)"
        )

        if va_f1 > best_f1:
            best_f1, patience = va_f1, 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ↑ Новый лучший val Macro-F1: {best_f1:.4f} — сохранено в {ckpt}")
        else:
            patience += 1
            if patience >= TRAIN.early_stopping_patience:
                print(f"Ранняя остановка на эпохе {epoch} "
                      f"(patience={TRAIN.early_stopping_patience})")
                break

    history_path = ARTIFACTS / f"history_{model_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"История обучения сохранена: {history_path}")

    plot_curves(history, model_name)
    print(f"Лучший val Macro-F1: {best_f1:.4f}. Веса: {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn2d",
                        choices=["cnn2d", "cnn_bilstm"])
    parser.add_argument("--warmup-cache", action="store_true",
                        help="Прогреть дисковый кэш logmel перед обучением.")
    parser.add_argument("--profile-loader", action="store_true",
                        help="Замерить время загрузки батча DataLoader-ом.")
    args = parser.parse_args()
    main(args.model, do_warmup=args.warmup_cache,
         do_profile=args.profile_loader)
