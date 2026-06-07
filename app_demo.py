"""
app_demo.py — веб-демонстрация прототипа SER на Gradio (раздел 3.4).
Позволяет загрузить или записать аудио и получить распределение вероятностей
по пяти эмоциям.

Запуск:
    python app_demo.py --weights artifacts/checkpoints/cnn_bilstm_best.pt
"""
import argparse

import gradio as gr
import torchaudio

from config import AUDIO
from inference import load_model, predict_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cnn_bilstm",
                    choices=["cnn2d", "cnn_bilstm"])
parser.add_argument("--weights", required=True)
ARGS = parser.parse_args()

MODEL = load_model(ARGS.model, ARGS.weights)


def classify(audio_path):
    if audio_path is None:
        return {}
    # Gradio отдаёт путь к временному файлу; приведение к 16 кГц при необходимости
    return predict_file(MODEL, audio_path)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(type="filepath", label="Загрузите или запишите речь"),
    outputs=gr.Label(num_top_classes=5, label="Эмоции"),
    title="Распознавание эмоций по речи (русский язык)",
    description="Прототип SER-модели, обученной на датасете Dusha.",
)

if __name__ == "__main__":
    demo.launch()
