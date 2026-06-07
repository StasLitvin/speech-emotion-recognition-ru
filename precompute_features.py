"""
precompute_features.py — однократный предрасчёт кэша log-Mel признаков
для всех wav-файлов (train/val/test).

Использует штатный warmup_feature_cache из dataset.py, поэтому кэш пишется
ровно туда и в том формате, откуда его затем читает SERDataset
(ARTIFACTS/feature_cache). Это устраняет прежнее рассогласование, когда
признаки считались в отдельный каталог в другом формате.

Запуск:
    python precompute_features.py
    python precompute_features.py --workers 8
"""
import argparse
import os

import pandas as pd

from dataset import build_dusha_dataframes, warmup_feature_cache


def main(workers: int):
    print("Сбор DataFrame-ов...")
    train_df, val_df, test_df = build_dusha_dataframes()
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
    print(f"Всего уникальных wav для кэша: {len(all_df)}")
    if len(all_df) == 0:
        print("Нет данных — проверьте DUSHA_ROOT в config.py.")
        return
    warmup_feature_cache(all_df, num_workers=workers, desc="logmel")
    print("Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2),
                        help="Число процессов DataLoader для прогрева кэша")
    args = parser.parse_args()
    main(args.workers)
