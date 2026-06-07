"""
diagnose.py — разовая диагностика данных Dusha:
  1) формат меток (golden_emo числовой/строковый) и сверка с annotator_emo;
  2) где физически лежат wav-файлы (особенно podcast — плоско или в подпапках).

Запуск:
    python diagnose.py
"""
import pandas as pd
from pathlib import Path

from config import DUSHA_ROOT


def check_labels(part="crowd", split="train"):
    tsv = DUSHA_ROOT / f"{part}_{split}" / f"raw_{part}_{split}.tsv"
    if not tsv.exists():
        print(f"[нет файла] {tsv}")
        return
    df = pd.read_csv(tsv, sep="\t", nrows=5000)
    print(f"\n=== Метки {part}_{split} ===")
    if "golden_emo" in df.columns:
        g = df["golden_emo"].dropna()
        is_num = pd.api.types.is_numeric_dtype(g)
        print(f"golden_emo: dtype={g.dtype} (числовой={is_num}) | "
              f"примеры={g.unique()[:8].tolist()}")
        if is_num and "annotator_emo" in df.columns:
            sub = df.dropna(subset=["golden_emo"]).copy()
            sub["annotator_emo"] = sub["annotator_emo"].astype(str).str.lower()
            ct = pd.crosstab(sub["golden_emo"].astype(int), sub["annotator_emo"])
            print("Сверка golden(число) × annotator(строка) — ищем доминанту в строке:")
            print(ct)
    if "annotator_emo" in df.columns:
        print("annotator_emo примеры:",
              df["annotator_emo"].dropna().astype(str).str.lower().unique()[:8].tolist())
    print("audio_path пример:", df["audio_path"].iloc[0])


def check_wavs(part, split):
    base = DUSHA_ROOT / f"{part}_{split}"
    print(f"\n=== wav {part}_{split} ===")
    if not base.exists():
        print(f"[нет папки] {base}")
        return
    print("в папке сплита:", [p.name for p in base.iterdir()][:10])
    wavs = base / "wavs"
    if wavs.exists():
        print("первые элементы wavs/:", [p.name for p in list(wavs.iterdir())[:8]])
        n = sum(1 for _ in wavs.rglob("*.wav"))
        print(f"всего .wav рекурсивно в wavs/: {n}")
        first = next(wavs.rglob("*.wav"), None)
        if first:
            print("пример пути к реальному .wav:", first.relative_to(base))
    else:
        n = sum(1 for _ in base.rglob("*.wav"))
        print(f"папки wavs/ нет; всего .wav рекурсивно в сплите: {n}")


if __name__ == "__main__":
    check_labels("crowd", "train")
    check_wavs("crowd", "train")
    check_wavs("podcast", "train")
