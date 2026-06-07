
import os
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from config import (AUDIO, TRAIN, DUSHA_ROOT, RESD_ROOT,
                    LABEL2ID, NUM_CLASSES, ARTIFACTS)
from preprocessing import (load_audio, amplitude_normalize, preemphasis,
                           fix_length, LogMelExtractor, WaveAugment, SpecAugment)


DUSHA_GOLDEN_MAP = {
    1: "positive",
    2: "neutral",
    3: "sad",
    4: "angry",
    5: "other",
}

COLUMN_MAP = {
    "audio": ["audio_path", "wav_path", "path", "file", "filepath", "filename"],
    "label": ["label", "emotion", "golden_emo", "annotator_emo"],
}


FEATURE_CACHE_DIR = Path(getattr(TRAIN, "feature_cache_dir",
                                 ARTIFACTS / "feature_cache"))
USE_FEATURE_CACHE = bool(getattr(TRAIN, "use_feature_cache", True))

CACHE_FOR_TRAIN = bool(getattr(TRAIN, "cache_for_train", True))



def _pick_column(df: pd.DataFrame, kind: str) -> str:
    for name in COLUMN_MAP[kind]:
        if name in df.columns:
            return name
    raise KeyError(
        f"Не найден столбец для '{kind}' среди {list(df.columns)}. "
        f"Допустимые имена: {COLUMN_MAP[kind]}"
    )


def _find_dusha_tsv(part: str, split: str):
    candidates = [
        (DUSHA_ROOT / f"{part}_{split}" / f"raw_{part}_{split}.tsv",
         DUSHA_ROOT / f"{part}_{split}"),
        (DUSHA_ROOT / part / f"{split}.tsv",
         DUSHA_ROOT / part),
        (DUSHA_ROOT / part / f"raw_{part}_{split}.tsv",
         DUSHA_ROOT / part),
        (DUSHA_ROOT / f"{part}_{split}" / f"{split}.tsv",
         DUSHA_ROOT / f"{part}_{split}"),
    ]
    for tsv, base in candidates:
        if tsv.exists():
            return tsv, base, candidates
    return None, None, candidates


def _find_wav_root(base: Path) -> Path:
    for cand in (base / "wavs", base, base / "audio", base / "data" / "wavs"):
        if cand.exists() and cand.is_dir():
            try:
                next(cand.rglob("*.wav"))
                return cand
            except StopIteration:
                continue
    return base / "wavs"


def _majority_vote(series: pd.Series) -> str:
    vals = [str(v).strip().lower() for v in series.dropna()
            if str(v).strip().lower() not in ("", "nan", "none")]
    if not vals:
        return ""
    counts = Counter(vals).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return ""
    return counts[0][0]


def _filter_existing_paths(df: pd.DataFrame, tag: str,
                           path_col: str = "audio_path") -> pd.DataFrame:
    if len(df) == 0:
        return df
    mask = df[path_col].map(lambda p: isinstance(p, str) and os.path.exists(p))
    n_missing = int((~mask).sum())
    if n_missing:
        examples = df.loc[~mask, path_col].head(3).tolist()
        print(f"[{tag}] пропущено отсутствующих wav: {n_missing} из {len(df)}")
        for ex in examples:
            print(f"    нет файла: {ex}")
    return df[mask].reset_index(drop=True)



def read_dusha_split(part: str, split: str) -> pd.DataFrame:
    tsv, base, candidates = _find_dusha_tsv(part, split)
    if tsv is None:
        listing = "\n  ".join(
            str(p.relative_to(DUSHA_ROOT))
            for p in sorted(DUSHA_ROOT.rglob("*.tsv"))
        ) or "(*.tsv не найдены)"
        tried = "\n  ".join(str(t) for t, _ in candidates)
        raise FileNotFoundError(
            f"Не найден TSV для part={part}, split={split}.\n"
            f"Искал:\n  {tried}\n\n"
            f"TSV в DUSHA_ROOT={DUSHA_ROOT}:\n  {listing}"
        )

    df = pd.read_csv(tsv, sep="\t")
    print(f"[{part}_{split}] {tsv.name}: строк={len(df)}, "
          f"колонки={df.columns.tolist()}")

    if "audio_path" not in df.columns:
        raise KeyError(f"[{part}_{split}] нет столбца 'audio_path'.")
    if "annotator_emo" not in df.columns and "golden_emo" not in df.columns:
        raise KeyError(f"[{part}_{split}] нет ни 'annotator_emo', ни 'golden_emo'.")

    def golden_to_label(v):
        if pd.isna(v):
            return np.nan
        try:
            iv = int(float(v))
        except (TypeError, ValueError):
            return np.nan
        return DUSHA_GOLDEN_MAP.get(iv, np.nan)

    if "golden_emo" in df.columns:
        df["label_golden"] = df["golden_emo"].map(golden_to_label)
    else:
        df["label_golden"] = np.nan

    if "annotator_emo" in df.columns:
        df["annotator_emo"] = df["annotator_emo"].astype(str).str.strip().str.lower()
    else:
        df["annotator_emo"] = np.nan

    agg = df.groupby("audio_path", sort=False).agg(
        label_golden=("label_golden",
                      lambda s: next((x for x in s if isinstance(x, str)), np.nan)),
        label_vote=("annotator_emo", _majority_vote),
    ).reset_index()

    agg["label"] = agg["label_golden"].where(
        agg["label_golden"].notna() & (agg["label_golden"] != ""),
        agg["label_vote"]
    )
    agg["label"] = agg["label"].replace({"": np.nan})

    n_total = len(agg)
    agg = agg.dropna(subset=["label"])
    n_after_agg = len(agg)

    uniq_before = sorted(agg["label"].unique().tolist())
    agg = agg[agg["label"].isin(LABEL2ID.keys())].reset_index(drop=True)
    print(f"[{part}_{split}] уникальных wav: {n_total}, "
          f"с меткой: {n_after_agg}, в нужных классах: {len(agg)}; "
          f"уникальные метки до фильтра: {uniq_before}")

    if len(agg) == 0:
        raise RuntimeError(
            f"[{part}_{split}] 0 строк после агрегации меток.\n"
            f"  Уникальные метки: {uniq_before}\n"
            f"  Ожидаются: {list(LABEL2ID.keys())}"
        )

    wav_root = _find_wav_root(base)
    sample = str(agg["audio_path"].iloc[0])

    if (DUSHA_ROOT / sample).exists():
        resolve = lambda p: str(DUSHA_ROOT / str(p))
    elif (base / sample).exists():
        resolve = lambda p: str(base / str(p))
    elif (wav_root / Path(sample).name).exists():
        resolve = lambda p: str(wav_root / Path(str(p)).name)
    else:
        resolve = lambda p: str(base / str(p))

    agg["audio_path"] = agg["audio_path"].apply(resolve)

    first_wav = Path(agg["audio_path"].iloc[0])
    if not first_wav.exists():
        print(f"[{part}_{split}] ВНИМАНИЕ: первый wav не найден: {first_wav}")
        print(f"  Проверьте папку с wav. Ожидалось рядом с TSV: {base}")

    agg = _filter_existing_paths(agg, tag=f"{part}_{split}")

    print(f"[{part}_{split}] распределение классов (после фильтра существующих):")
    if len(agg) > 0:
        print(agg["label"].value_counts().to_string())
    else:
        print("  (пусто — ни одного существующего wav)")

    return agg[["audio_path", "label"]]


def build_dusha_dataframes():
    from sklearn.model_selection import train_test_split

    train_parts = [read_dusha_split(p, "train") for p in ("crowd", "podcast")]
    test_parts  = [read_dusha_split(p, "test")  for p in ("crowd", "podcast")]
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)

    print(f"Итого: train={len(train_df)}, test={len(test_df)}")
    if len(train_df) == 0:
        raise RuntimeError(
            "Train пуст. Проверьте DUSHA_ROOT, структуру папок и метки."
        )
    if len(test_df) == 0:
        print("ВНИМАНИЕ: test пуст — ни одного существующего wav.")

    if TRAIN.subset_fraction < 1.0:
        parts = []
        for lbl, grp in train_df.groupby("label", sort=False):
            n = max(1, int(round(len(grp) * TRAIN.subset_fraction)))
            parts.append(grp.sample(n=min(n, len(grp)), random_state=TRAIN.seed))
        train_df = pd.concat(parts, ignore_index=True)
        print(f"После subset_fraction={TRAIN.subset_fraction}: "
              f"train={len(train_df)}, столбцы={train_df.columns.tolist()}")
        if len(train_df) < 10:
            raise RuntimeError(
                "Слишком маленький subset_fraction — train почти пуст."
            )

    label_counts = train_df["label"].value_counts()
    rare = label_counts[label_counts < 2]
    if len(rare) > 0:
        print(f"ВНИМАНИЕ: классы с <2 примерами отброшены "
              f"для стратификации: {rare.to_dict()}")
        train_df = train_df[~train_df["label"].isin(rare.index)].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_df, test_size=0.10, stratify=train_df["label"],
        random_state=TRAIN.seed)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))



def read_resd() -> pd.DataFrame:
    df = pd.read_csv(RESD_ROOT / "test.csv")
    audio_col, label_col = _pick_column(df, "audio"), _pick_column(df, "label")
    df = df.rename(columns={audio_col: "audio_path", label_col: "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["audio_path"] = df["audio_path"].apply(
        lambda p: str((RESD_ROOT / p).resolve()))
    resd_to_dusha = {
        "anger": "angry", "sadness": "sad", "neutral": "neutral",
        "happiness": "positive", "enthusiasm": "positive",
        "disgust": "other", "fear": "other",
    }
    df["label"] = df["label"].map(resd_to_dusha)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df = _filter_existing_paths(df, tag="resd")
    return df[["audio_path", "label"]]



def _feature_cache_key() -> str:
    keys = (
        getattr(AUDIO, "sample_rate", 16000),
        getattr(AUDIO, "n_fft", 1024),
        getattr(AUDIO, "hop_length", 256),
        getattr(AUDIO, "n_mels", 64),
        getattr(AUDIO, "fmin", 0),
        getattr(AUDIO, "fmax", 8000),
        getattr(AUDIO, "duration", 4.0),
        "logmel-v1",
    )
    h = hashlib.md5(str(keys).encode("utf-8")).hexdigest()[:10]
    return h


def _cache_path_for(audio_path: str, cache_key: str) -> Path:
    h = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    sub = h[:2]
    return FEATURE_CACHE_DIR / cache_key / sub / f"{h}.pt"



class SERDataset(Dataset):


    def __init__(self, df: pd.DataFrame, mode: str = "logmel",
                 train: bool = False, use_cache: bool = True):
        assert mode in ("logmel", "raw")
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.train = train
        self.use_cache = bool(
            use_cache and USE_FEATURE_CACHE and mode == "logmel"
            and (CACHE_FOR_TRAIN or not train)
        )
        self.cache_key = _feature_cache_key() if self.use_cache else None
        self.extractor = LogMelExtractor()

        wave_aug_ok = train and not self.use_cache
        self.wave_aug = WaveAugment(p=0.5) if wave_aug_ok else None
        self.spec_aug = SpecAugment() if train else None

        if self.use_cache:
            FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _load_one(self, idx):
        row = self.df.iloc[idx]
        path = row["audio_path"]
        wav = load_audio(path)
        label = LABEL2ID[row["label"]]
        return wav, label, path

    def _compute_logmel(self, wav):
        wav = amplitude_normalize(wav)
        wav = preemphasis(wav)
        if self.wave_aug is not None:
            wav = self.wave_aug(wav)
        wav = fix_length(wav)
        log_mel = self.extractor(wav)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = log_mel.unsqueeze(0)
        return log_mel

    def _get_logmel_cached(self, path: str):
        cp = _cache_path_for(path, self.cache_key)
        if cp.exists():
            try:
                return torch.load(cp, map_location="cpu")
            except Exception as e:
                print(f"[cache] битый файл {cp} ({type(e).__name__}) — пересчёт")
        # пересчёт
        wav = load_audio(path)
        log_mel = self._compute_logmel(wav)
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(".pt.tmp")
            torch.save(log_mel, tmp)
            os.replace(tmp, cp)
        except Exception as e:
            print(f"[cache] не удалось сохранить {cp}: {type(e).__name__}")
        return log_mel

    def __getitem__(self, idx):
        n = len(self.df)
        if n == 0:
            raise RuntimeError("SERDataset пуст.")

        tries, cur = 0, idx
        last_err_type = None
        while tries < 10:
            try:
                row = self.df.iloc[cur]
                path = row["audio_path"]
                label = LABEL2ID[row["label"]]

                if self.mode == "raw":
                    wav = load_audio(path)
                    wav = amplitude_normalize(wav)
                    wav = fix_length(preemphasis(wav))
                    return wav, label

                # logmel
                if self.use_cache:
                    log_mel = self._get_logmel_cached(path)
                else:
                    wav = load_audio(path)
                    log_mel = self._compute_logmel(wav)

                if self.spec_aug is not None:
                    log_mel = self.spec_aug(log_mel)

                return log_mel, label

            except Exception as e:
                last_err_type = type(e).__name__
                bad_path = self.df.iloc[cur]["audio_path"]
                print(f"[SERDataset] пропуск idx={cur} ({last_err_type}): {bad_path}")
                cur = (cur + 1) % n
                tries += 1

        raise RuntimeError(
            f"Не удалось прочитать 10 файлов подряд, начиная с idx={idx}. "
            f"Последняя ошибка: {last_err_type}"
        )




def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["label"].map(LABEL2ID).value_counts().sort_index()
    counts = counts.reindex(range(NUM_CLASSES), fill_value=0).values
    total = counts.sum()
    weights = total / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def make_loaders(mode: str = "logmel"):
    train_df, val_df, test_df = build_dusha_dataframes()

    num_workers = int(getattr(TRAIN, "num_workers", 4))
    pin = bool(getattr(TRAIN, "pin_memory", True)) and torch.cuda.is_available()
    prefetch = int(getattr(TRAIN, "prefetch_factor", 4))
    persistent = bool(getattr(TRAIN, "persistent_workers", True)) and num_workers > 0

    common_kwargs = dict(
        batch_size=TRAIN.batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=_seed_worker,
    )
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = prefetch
        common_kwargs["persistent_workers"] = persistent

    def _loader(df, shuffle, drop_last, is_train):
        if len(df) == 0:
            return None
        ds = SERDataset(df, mode=mode, train=is_train)
        return DataLoader(
            ds,
            shuffle=shuffle,
            drop_last=drop_last,
            **common_kwargs,
        )

    loaders = {
        "train": _loader(train_df, shuffle=True,  drop_last=True,  is_train=True),
        "val":   _loader(val_df,   shuffle=False, drop_last=False, is_train=False),
        "test":  _loader(test_df,  shuffle=False, drop_last=False, is_train=False),
    }
    return loaders, (train_df, val_df, test_df)



def warmup_feature_cache(df: pd.DataFrame, num_workers: int = None,
                         desc: str = "warmup"):
    if not USE_FEATURE_CACHE:
        print("[warmup] USE_FEATURE_CACHE=False — пропуск.")
        return
    from tqdm import tqdm

    nw = num_workers if num_workers is not None else int(getattr(TRAIN, "num_workers", 4))
    ds = SERDataset(df, mode="logmel", train=False, use_cache=True)
    loader = DataLoader(
        ds, batch_size=TRAIN.batch_size, shuffle=False,
        num_workers=nw, pin_memory=False,
        persistent_workers=False,
    )
    for _ in tqdm(loader, desc=desc):
        pass
    print(f"[warmup] кэш готов: {FEATURE_CACHE_DIR / _feature_cache_key()}")
