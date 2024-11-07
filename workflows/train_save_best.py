import logging
import logging.config

import click
import numpy as np
import yaml

from src.logger.log_settings import LOGGING_CONFIG
from src.models.clustering import get_clustering_model


@click.command()
@click.option(
    "--in_file",
    required=True,
    help="Input file full path to data",
)
@click.option("--out_file", type=str, help="Output model file")
@click.option("--model_name", required=True, type=str, help="Имя модели")
@click.option("--par_file", required=True, type=str, help="Имя модели")
def train_save(in_file: str, out_file: str, model_name: str, par_file: str):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("file_logger")

    logger.debug("Обучение модели с лучшими параметрами..")

    try:
        with open(par_file) as f:
            model_prmt = yaml.safe_load(f)
            logger.debug("Параметры модели загружены!!")

    except Exception as ex:
        logger.exception(f"Ошибка при загрузке параметров: {ex}")

    try:
        input_df = np.load(in_file)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    try:
        model = get_clustering_model(model_name, model_prmt)
        model.fit(input_df)
        model.save(out_file)
        logger.info("Модель обучена исохранена")
    except Exception as ex:
        logger.exception(f"Ошибка при обучении модели: {ex}")


if __name__ == "__main__":
    train_save()
