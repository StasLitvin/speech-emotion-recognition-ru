import os
from functools import partial
from typing import Dict, List

import numpy as np
import torch
import soundfile as sf
from sklearn.metrics import f1_score, accuracy_score
from transformers import (AutoFeatureExtractor,
                          AutoModelForAudioClassification,
                          TrainingArguments, Trainer,
                          DataCollatorWithPadding)

from config import (TRAIN, ARTIFACTS, PRETRAINED_MODEL,
                    NUM_CLASSES, LABEL2ID, ID2LABEL, AUDIO)
from dataset import build_dusha_dataframes, compute_class_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Грузим один раз на уровне модуля — так его смогут увидеть воркеры DataLoader
feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL)


# ----------------------------- IO helpers -----------------------------

def _load_wav_resampled(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.squeeze(0).numpy()
    return wav.astype(np.float32, copy=False)


def _quick_check(path: str) -> bool:
    try:
        info = sf.info(path)
        return info.frames > 0
    except Exception:
        return False


# ----------------------------- Dataset --------------------------------

def _prepare_batch(batch: Dict[str, List],
                   sr: int,
                   max_samples: int,
                   tag: str) -> Dict[str, List]:
    """Трансформ на лету для HF Dataset. ВАЖНО: это функция уровня модуля,
    чтобы её можно было запиклить для воркеров DataLoader на Windows."""
    wavs = []
    for p in batch["audio_path"]:
        try:
            wavs.append(_load_wav_resampled(p, sr))
        except Exception as e:
            print(f"[hubert/{tag}] не удалось прочитать {p}: "
                  f"{type(e).__name__}: {e}")
            wavs.append(np.zeros(sr, dtype=np.float32))

    inputs = feature_extractor(
        wavs,
        sampling_rate=sr,
        max_length=max_samples,
        truncation=True,
    )
    inputs["labels"] = [LABEL2ID[l] for l in batch["label"]]
    return inputs


def df_to_hf(df, tag: str = ""):
    from datasets import Dataset

    df = df[df["audio_path"].map(lambda p: isinstance(p, str) and os.path.exists(p))]
    mask = df["audio_path"].map(_quick_check)
    dropped = (~mask).sum()
    if dropped:
        print(f"[hubert/{tag}] отброшено нечитаемых: {int(dropped)}")
    df = df[mask].reset_index(drop=True)
    print(f"[hubert/{tag}] валидных файлов: {len(df)}")

    ds = Dataset.from_pandas(df[["audio_path", "label"]], preserve_index=False)

    transform = partial(
        _prepare_batch,
        sr=AUDIO.sample_rate,
        max_samples=AUDIO.max_samples,
        tag=tag,
    )
    ds = ds.with_transform(transform)
    return ds


# ----------------------------- Trainer --------------------------------

class WeightedTrainer(Trainer):

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = (class_weights.to(DEVICE)
                              if class_weights is not None else None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1":    f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "accuracy":    accuracy_score(labels, preds),
    }


def _build_training_args():
    common = dict(
        output_dir=str(ARTIFACTS / "checkpoints" / "hubert"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=TRAIN.lr_hubert,
        weight_decay=TRAIN.weight_decay,
        num_train_epochs=8,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="tensorboard",
        remove_unused_columns=False,
        # На Windows важна именно функция уровня модуля (см. _prepare_batch).
        # Если воркеры всё равно капризничают — поставьте 0.
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )
    try:
        return TrainingArguments(evaluation_strategy="epoch", **common)
    except TypeError:
        # Для новых версий transformers, где параметр переименован.
        common.pop("dataloader_persistent_workers", None)
        return TrainingArguments(eval_strategy="epoch", **common)


# ----------------------------- Main -----------------------------------

def main():
    train_df, val_df, test_df = build_dusha_dataframes()

    if len(train_df) == 0:
        raise RuntimeError("Train пуст — нечего обучать.")
    if len(val_df) == 0:
        raise RuntimeError("Val пуст — нечего валидировать.")

    train_ds = df_to_hf(train_df, tag="train")
    val_ds   = df_to_hf(val_df,   tag="val")

    if len(train_ds) == 0:
        raise RuntimeError("После фильтрации битых файлов train пуст.")
    if len(val_ds) == 0:
        raise RuntimeError("После фильтрации битых файлов val пуст.")

    weights = compute_class_weights(train_df) if TRAIN.use_class_weights else None

    model = AutoModelForAudioClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    (ARTIFACTS / "checkpoints" / "hubert").mkdir(parents=True, exist_ok=True)

    args = _build_training_args()

    data_collator = DataCollatorWithPadding(
        feature_extractor,
        padding=True,
        return_tensors="pt",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=weights,
        data_collator=data_collator,
    )

    trainer.train()

    save_dir = ARTIFACTS / "checkpoints" / "hubert_best"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    feature_extractor.save_pretrained(str(save_dir))

    metrics = trainer.evaluate()
    print("Итоговые метрики на val:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # На Windows крайне желательно явно задать spawn (Trainer и так его использует,
    # но если будете запускать вручную — это страховка).
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    main()
