import logging
import logging.config

import click
import numpy as np
import pandas as pd

from src.logger.log_settings import LOGGING_CONFIG
from src.visualization.cl_visualizer import Visualizer


@click.command()
@click.option("--model_path", required=True, type=str, help="Путь к модели")
@click.option("--data_path", required=True, type=str, help="Путь к данным")
@click.option(
    "--pred_df_path", required=True, type=str, help="Путь к предикту"
)
@click.option("--n_best", type=int, help="Кол-во лучших кластеров")
def main(model_path: str, data_path: str, pred_df_path: str, n_best: int):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("file_logger")

    logger.debug("Начало визуализации.")

    try:
        logger.debug("Загрузка данных для визуализации...")
        input_df = np.load(data_path)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    try:
        logger.debug("Загрузка данных для предсказания...")
        df = pd.read_csv(pred_df_path)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    try:
        logger.debug("Инициализация класса...")
        visualizer = Visualizer(model_path)
    except Exception as ex:
        logger.exception(f"Ошибка при создании класса: {ex}")

    try:
        logger.debug("Построение графиков...")
        visualizer.visualize(input_df, n_best)
        visualizer.visualize_with_predict(df, input_df, n_best)
    except Exception as ex:
        logger.exception(f"Ошибка при построении графиков(: {ex}")


if __name__ == "__main__":
    main()
