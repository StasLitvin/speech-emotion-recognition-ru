import os
from datasets import load_dataset

# Создаём директорию (аналог mkdir -p data/resd)
os.makedirs("data/resd", exist_ok=True)

# Загружаем датасет
ds = load_dataset("Aniemore/resd", split="test")

# Сохраняем в CSV
output_path = os.path.join("data", "resd", "test.csv")
ds.to_csv(output_path)

print(f"Готово! Файл сохранён: {output_path}")
