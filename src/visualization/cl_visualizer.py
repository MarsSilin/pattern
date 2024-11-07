import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pylab as plt
from tslearn.clustering import KShape
from src.models.metrics import indices_std, std


class Visualizer:
    """Визуализация лучших класстеров, предсказанных моделью."""

    def __init__(self, model_path: str):
        """Загружает обученную модель.

        Args:
            model_path (str): путь к модели.
        """
        self.model = KShape.from_pickle(model_path)

    def visualize(self, data: np.ndarray, n_best: int = 10):
        """Визуализация класстеров без предикта.

        Args:
            data (np.ndarray): данные свеч
            n_best (int, optional): Сколько лучших кластеров брать.
            Defaults to 10.
        """
        y_pred = self.model.predict(data)
        best_idx = indices_std(
            self.model.cluster_centers_, data, y_pred, n_best
        )
        list_std = std(self.model.cluster_centers_, data, y_pred)
        # длина временного ряда
        sz = data.shape[1]
        # число кластеров
        nc = len(best_idx)

        plt.figure(figsize=(12, nc * 4))
        for yi, cl in enumerate(best_idx):
            plt.subplot(nc, 1, 1 + yi)
            for xx in data[y_pred == cl]:
                plt.plot(xx.ravel(), "k-", alpha=0.2)
            plt.plot(self.model.cluster_centers_[cl].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.title(
                f"Cluster {str(cl)} has {str(data[y_pred == cl].shape[0])}"
                + " timeseries and std is %.2f" % list_std[cl]
            )

        plt.tight_layout()
        plt.savefig("reports/figures/visual.png")

    def visualize_with_predict(
        self, df: pd.DataFrame, data: np.ndarray, n_best: int = 10
    ):
        """Визуализация  с периодом для предсказания.

        Args:
            df (pd.DataFrame): данные периода для предсказания.
            data (np.ndarray):данные свеч.
            n_best (int, optional): кол-во кластеров для визуализации.
            Defaults to 10.
        """
        y_pred = self.model.predict(data)
        best_idx = indices_std(
            self.model.cluster_centers_, data, y_pred, n_best
        )
        list_std = std(self.model.cluster_centers_, data, y_pred)
        # длина временного ряда
        sz = df.shape[1]
        # число кластеров
        nc = len(best_idx)

        plt.figure(figsize=(12, nc * 4))
        for yi, cl in enumerate(best_idx):
            plt.subplot(nc, 1, 1 + yi)
            cl_df = df[y_pred == cl].reset_index(drop=True)
            for xx in range(len(cl_df)):
                plt.plot(cl_df.iloc[xx, :], "k-", alpha=0.2)
            plt.plot(self.model.cluster_centers_[cl].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.title(
                f"Cluster {str(cl)} has {str(df[y_pred == cl].shape[0])}"
                + " timeseries and std is %.2f" % list_std[cl]
            )

        plt.tight_layout()
        plt.savefig("reports/figures/visual_pred.png")
