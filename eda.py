
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import ARTIFACTS, EMOTIONS, EMOTION_RU, AUDIO
from dataset import build_dusha_dataframes
from preprocessing import load_audio, amplitude_normalize, LogMelExtractor


def _ensure_dir(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_counts(df, emotions):
    if df is None or len(df) == 0 or "label" not in df.columns:
        return [0] * len(emotions)
    return [int((df["label"] == e).sum()) for e in emotions]


def plot_class_distribution(train_df, val_df, test_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(len(EMOTIONS))

    splits = [("train", train_df), ("val", val_df), ("test", test_df)]
    for i, (name, df) in enumerate(splits):
        counts = _safe_counts(df, EMOTIONS)
        ax.bar(x + (i - 1) * width, counts, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([EMOTION_RU[e] for e in EMOTIONS])
    ax.set_ylabel("Число записей")
    ax.set_title("Распределение классов в датасете Dusha")
    ax.legend()
    fig.tight_layout()

    out = ARTIFACTS / "figures" / "class_distribution.png"
    _ensure_dir(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out}")

    train_counts = _safe_counts(train_df, EMOTIONS)
    val_counts   = _safe_counts(val_df,   EMOTIONS)
    test_counts  = _safe_counts(test_df,  EMOTIONS)

    print("\nКласс           train     val    test")
    for e, tr, va, te in zip(EMOTIONS, train_counts, val_counts, test_counts):
        print(f"{EMOTION_RU[e]:<14}{tr:>8}{va:>8}{te:>8}")
    print(f"{'ИТОГО':<14}"
          f"{sum(train_counts):>8}"
          f"{sum(val_counts):>8}"
          f"{sum(test_counts):>8}")


def plot_example_spectrograms(train_df):
    if train_df is None or len(train_df) == 0:
        print("plot_example_spectrograms: пустой train_df, пропуск.")
        return

    extractor = LogMelExtractor()
    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(16, 3.2))

    # На случай len(EMOTIONS) == 1
    if len(EMOTIONS) == 1:
        axes = [axes]

    plotted_any = False
    for ax, emo in zip(axes, EMOTIONS):
        sample = train_df[train_df["label"] == emo]
        if len(sample) == 0:
            ax.set_title(f"{EMOTION_RU[emo]} (нет данных)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        path = sample.iloc[0]["audio_path"]
        try:
            wav = amplitude_normalize(load_audio(path))
            log_mel = extractor(wav).numpy()
        except Exception as exc:
            print(f"  не удалось обработать {path}: {exc}")
            ax.set_title(f"{EMOTION_RU[emo]} (ошибка)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(EMOTION_RU[emo])
        ax.set_xlabel("Время")
        ax.set_yticks([])
        plotted_any = True

    axes[0].set_ylabel("Мел-канал")
    fig.suptitle("Примеры логарифмических мел-спектрограмм по эмоциям")
    fig.tight_layout()

    out = ARTIFACTS / "figures" / "example_spectrograms.png"
    _ensure_dir(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    if plotted_any:
        print(f"Сохранено: {out}")
    else:
        print(f"Сохранено (без валидных примеров): {out}")


if __name__ == "__main__":
    train_df, val_df, test_df = build_dusha_dataframes()
    print(f"Размеры выборок: train={len(train_df)}, "
          f"val={len(val_df)}, test={len(test_df)}")

    plot_class_distribution(train_df, val_df, test_df)
    plot_example_spectrograms(train_df)