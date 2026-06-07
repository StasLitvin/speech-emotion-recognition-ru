"""
config.py — единая конфигурация эксперимента.
Все пути и гиперпараметры задаются здесь, чтобы их не приходилось менять в коде.
"""
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Базовые пути. ИЗМЕНИТЕ DUSHA_ROOT и RESD_ROOT под свою файловую систему.
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
DUSHA_ROOT = DATA_ROOT / "dusha"      # сюда распаковывается датасет Dusha
RESD_ROOT = DATA_ROOT / "resd"        # сюда распаковывается датасет RESD
ARTIFACTS = PROJECT_ROOT / "artifacts"   # чекпоинты, логи, рисунки
ARTIFACTS.mkdir(exist_ok=True)
(ARTIFACTS / "figures").mkdir(exist_ok=True)
(ARTIFACTS / "checkpoints").mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Классы эмоций датасета Dusha (5 классов).
# ----------------------------------------------------------------------------
EMOTIONS = ["angry", "sad", "neutral", "positive", "other"]
EMOTION_RU = {
    "angry": "злость",
    "sad": "грусть",
    "neutral": "нейтральное",
    "positive": "радость",
    "other": "другое",
}
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}
NUM_CLASSES = len(EMOTIONS)


@dataclass
class AudioConfig:
    """Параметры обработки аудиосигнала (раздел 2.1 работы)."""
    sample_rate: int = 16_000      # частота дискретизации, Гц
    max_duration: float = 4.0      # максимальная длительность фрагмента, с
    n_fft: int = 1024              # размер БПФ (окно дополняется нулями)
    hop_length: int = 160          # шаг 10 мс при 16 кГц
    win_length: int = 400          # окно 25 мс при 16 кГц
    n_mels: int = 128              # число мел-каналов (см. вывод раздела 2.1)
    fmin: int = 20
    fmax: int = 8000
    preemphasis: float = 0.97      # коэффициент предискажающего фильтра

    @property
    def max_samples(self) -> int:
        return int(self.sample_rate * self.max_duration)

    @property
    def max_frames(self) -> int:
        return self.max_samples // self.hop_length + 1


@dataclass
class TrainConfig:
    """Гиперпараметры обучения (раздел 3.2 работы)."""
    batch_size: int = 64
    num_epochs: int = 60
    lr: float = 3e-4               # для CNN-моделей
    lr_hubert: float = 1e-5        # для дообучения HuBERT
    weight_decay: float = 1e-4     # коэффициент L2-регуляризации
    dropout: float = 0.3
    early_stopping_patience: int = 10
    num_workers: int  = 4  # ≈ число физ. ядер CPU
    pin_memory: bool = True
    prefetch_factor: int  = 4
    persistent_workers: bool = True
    use_feature_cache: bool = True  # включить дисковый кэш logmel
    cache_for_train: bool = True  # если в train важны wave-аугментации — поставьте False
    use_amp: bool = True           # обучение в смешанной точности
    use_class_weights: bool = True # взвешенная кросс-энтропия
    seed: int = 42

    # Режим быстрой подвыборки: если subset_fraction < 1.0, берётся часть данных.
    # Полезно для проверки кода и слабого железа (CPU/Colab).
    subset_fraction: float = 1.0


AUDIO = AudioConfig()
TRAIN = TrainConfig()

# Предобученная самоконтрольная модель для дообучения (раздел 2.2).
# Возможные варианты: facebook/hubert-base-ls960, facebook/wav2vec2-base,
# microsoft/wavlm-base. По умолчанию — HuBERT.
PRETRAINED_MODEL = "facebook/hubert-base-ls960"
