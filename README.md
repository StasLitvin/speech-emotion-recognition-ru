# Распознавание эмоций по русской речи (SER)

Практическая часть ВКР: распознавание эмоций по акустическим признакам русскоязычной речи на датасете **Dusha**. Реализованы три подхода и единый конвейер обучения/оценки:

- **2D-CNN** по логарифмическим мел-спектрограммам;
- **CNN-BiLSTM с механизмом внимания**;
- **дообучение предобученной модели HuBERT** (`facebook/hubert-base-ls960`).

Классификация по пяти классам Dusha: злость, грусть, нейтральное, радость, другое.

## Структура проекта

```
PyCharmMiscProject/
├── config.py             # единая конфигурация: пути, классы, гиперпараметры (AUDIO, TRAIN)
├── preprocessing.py      # загрузка аудио, нормализация, предыскажение, log-Mel, аугментации
├── dataset.py            # разметка Dusha/RESD, кэш признаков, SERDataset, DataLoader-ы
├── models.py             # архитектуры CNN2D и CNN-BiLSTM-Attention
├── train.py              # обучение CNN-моделей (AMP, ранняя остановка, кривые обучения)
├── train_hubert.py       # дообучение HuBERT через transformers.Trainer
├── evaluate.py           # метрики (accuracy, macro/weighted F1) + отчёт по классам
├── confusion_plots.py    # матрицы ошибок (доли, счётчики, топ-N сложных классов)
├── inference.py          # инференс одного файла и потоковый режим скользящим окном
├── app_demo.py           # веб-демо на Gradio
├── precompute_features.py# предрасчёт кэша признаков (обёртка над warmup_feature_cache)
├── eda.py                # разведочный анализ: распределение классов, примеры спектрограмм
├── diagnose.py           # диагностика структуры данных Dusha
├── resd.py               # выгрузка датасета RESD (Aniemore) для кросс-корпусной оценки
├── diag_imports.py       # диагностика «падающего» импорта (нативный краш на Windows)
└── requirements.txt
```

Каталоги `data/` и `artifacts/` (включая `artifacts/feature_cache/`) создаются автоматически и в репозиторий не коммитятся (см. `.gitignore`).

## Установка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

На Windows для чтения wav используется `soundfile`.

## Подготовка данных

1. Распакуйте crowd-часть **Dusha** в `data/dusha/` (папки `crowd_train/`, `crowd_test/` с TSV-разметкой и каталогом wav). Путь настраивается через `DUSHA_ROOT` в `config.py`.
2. (Необязательно) Выгрузите RESD для кросс-корпусной оценки: `python resd.py`.
3. Проверка данных и меток: `python diagnose.py`.
4. (Рекомендуется) Прогреть кэш признаков: `python precompute_features.py --workers 8`.

## Запуск

```bash
# обучение
python train.py --model cnn2d
python train.py --model cnn_bilstm --warmup-cache
python train_hubert.py

# оценка
python evaluate.py --model cnn_bilstm --weights artifacts/checkpoints/cnn_bilstm_best.pt
python evaluate.py --model hubert --weights artifacts/checkpoints/hubert_best --testset resd

# инференс и демо
python inference.py --weights artifacts/checkpoints/cnn_bilstm_best.pt --audio sample.wav
python app_demo.py --weights artifacts/checkpoints/cnn_bilstm_best.pt
```

## Ключевые решения

Параметры аудио (16 кГц, окно 25 мс / шаг 10 мс, 128 мел-каналов, предыскажение 0.97) и обучения (AdamW, косинусный отжиг, смешанная точность, ранняя остановка, взвешенная кросс-энтропия по обратной частоте классов) собраны в `config.py`. Разметка Dusha формируется мажоритарным голосованием аннотаторов с приоритетом числового `golden_emo` (`dataset.py`). Кэш log-Mel хранится в `artifacts/feature_cache/` и ключуется по параметрам аудио.

## Внесённые исправления

- `precompute_features.py` переписан: раньше он импортировал несуществующую `CACHE_ROOT` (падал на запуске) и считал признаки в каталог/формат, отличный от того, что читает `SERDataset`. Теперь он вызывает штатный `warmup_feature_cache`, так что кэш согласован с обучением.
- Файл `test.py` переименован в `diag_imports.py` (это диагностика нативного краша импорта, а не юнит-тесты).
- В `requirements.txt` добавлен `scipy` (используется в `diag_imports.py`) и поправлен перенос строки.

## Зависимости

Список зависимостей — в `requirements.txt` (`pip install -r requirements.txt`).
