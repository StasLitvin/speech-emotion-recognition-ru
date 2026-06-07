"""
_diag_imports.py — поиск импорта, который роняет процесс (0xC0000005).

Импортирует ровно то же, что train_hubert.py, по одному модулю, печатая
breadcrumb ПЕРЕД каждым импортом (flush=True). Последняя напечатанная строка =
шаг, на котором произошёл нативный краш. faulthandler дополнительно попытается
напечатать кадр краша.

Запуск:
    python -u _diag_imports.py 2>&1
"""
import sys
import faulthandler
faulthandler.enable()


def step(msg):
    print(f"[diag] {msg} ...", file=sys.stderr, flush=True)


step("stdlib (argparse, hashlib, os, time)")
import argparse, hashlib, os, time  # noqa

step("numpy")
import numpy as np  # noqa

step("torch")
import torch  # noqa
step(f"  torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")

step("soundfile")
import soundfile as sf  # noqa

step("sklearn.metrics")
from sklearn.metrics import f1_score, accuracy_score  # noqa

step("scipy.signal.resample_poly")
from scipy.signal import resample_poly  # noqa

step("torch.utils.data")
from torch.utils.data import Dataset, DataLoader  # noqa

step("tqdm")
from tqdm import tqdm  # noqa

step("transformers (классы)")
from transformers import (AutoFeatureExtractor,
                          AutoModelForAudioClassification,
                          TrainingArguments, Trainer)  # noqa

step("config")
from config import (TRAIN, ARTIFACTS, PRETRAINED_MODEL,
                    NUM_CLASSES, LABEL2ID, ID2LABEL, AUDIO)  # noqa
step(f"  PRETRAINED_MODEL={PRETRAINED_MODEL}")

step("dataset")
from dataset import build_dusha_dataframes, compute_class_weights  # noqa

step("AutoFeatureExtractor.from_pretrained (как в train_hubert на уровне модуля)")
fe = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL)

step("ВСЁ ОК — ни один импорт не уронил процесс")
print("[diag] DONE", file=sys.stderr, flush=True)