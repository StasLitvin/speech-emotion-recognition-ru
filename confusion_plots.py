"""
confusion_plots.py — расширенная визуализация матриц ошибок (раздел 3.3).

Строит три варианта по предсказаниям модели:
  1) нормализованную по строкам (доли)        -> cm_norm_<tag>.png
  2) с абсолютными счётчиками                  -> cm_count_<tag>.png
  3) по топ-N классам с наибольшим числом ошибок (в стиле присланной
     матрицы; для задач с большим числом классов)  -> cm_top<N>_<tag>.png

Функции принимают массивы истинных и предсказанных меток (целые номера
классов) и список названий классов. Подходят для любого числа классов.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import ARTIFACTS


def plot_confusion(y_true, y_pred, labels, tag, normalize=False):
    """Полная матрица ошибок: доли (normalize=True) или счётчики."""
    cm = confusion_matrix(y_true, y_pred,
                          normalize="true" if normalize else None)
    fmt = ".2f" if normalize else "d"
    n = len(labels)
    plt.figure(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", square=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"shrink": 0.8})
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    kind = "доли" if normalize else "счётчики"
    plt.title(f"Матрица ошибок ({kind}): {tag}")
    plt.tight_layout()
    out = ARTIFACTS / "figures" / f"cm_{'norm' if normalize else 'count'}_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Сохранено: {out}")
    return out


def plot_topn_confusion(y_true, y_pred, labels, tag, top_n=10):
    """Матрица по топ-N классам, на которых модель чаще всего ошибается.

    «Сложность» класса измеряется числом ошибок по соответствующей строке
    (сумма недиагональных элементов). Для 5 классов Dusha будут показаны
    все классы, упорядоченные по числу ошибок.
    """
    cm = confusion_matrix(y_true, y_pred)
    errors = cm.sum(axis=1) - np.diag(cm)          # ошибки по каждому истинному классу
    order = np.argsort(errors)[::-1][:top_n]       # индексы топ-N «сложных»
    order = np.sort(order)                          # вернуть естественный порядок
    sub = cm[np.ix_(order, order)]
    sub_labels = [labels[i] for i in order]
    k = len(order)
    plt.figure(figsize=(max(6, k * 0.7), max(5, k * 0.6)))
    sns.heatmap(sub, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=sub_labels, yticklabels=sub_labels)
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    plt.title(f"Матрица ошибок по топ-{k} «сложным» классам: {tag}")
    plt.tight_layout()
    out = ARTIFACTS / "figures" / f"cm_top{top_n}_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Сохранено: {out}")
    return out


def plot_all(y_true, y_pred, labels, tag, top_n=10):
    """Удобная обёртка: строит все три варианта матрицы."""
    plot_confusion(y_true, y_pred, labels, tag, normalize=True)
    plot_confusion(y_true, y_pred, labels, tag, normalize=False)
    plot_topn_confusion(y_true, y_pred, labels, tag, top_n=top_n)
