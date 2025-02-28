import logging
import time

import numpy as np
import optuna

from src.models.clustering import get_clustering_model

logger = logging.getLogger("file_logger")


def optimize(
    data: np.ndarray,
    model_name: str,
    model_prmt: dict,
    opt_clusters_min_max_step: tuple,
    best_cl: int = 10,
    n_trials: int = 10,
) -> dict:
    """Подбирает параметры для датасета и модели кластеризации.

    Args:
         data (np.ndarray): Стандартизированный датасет.

         all_clusters_min_max_step (tuple): Число кластеров, которые
             подбирает модель (ОТ, ДО, ШАГ).

         model_name (str): Имя модели для подбора параметров.

         model_prmt (dict): Параметры для инициализации модели.
            max_iter,
            n_init,
            n_clusters,

         best_clusters (int): Число хороших кластеров, которые ищет модель.
             Defaults to 10.

         n_trials (int, optional): Количество проходов оптимизации.
             Defaults to 10.
             (если хочется пройти все варианты, то надо ставить хотя бы число
             равное кол-ву сочетаний *4, но обычно хватает кол-ва сочетаний /2)

     Returns:
         best_params (dict): Подобранные параметры (общее число кластеров).

    """

    param_history = []
    iteration = [1]  # в objective можно извне передать только list
    seed = 0

    logger.debug(f"Имя модели: {model_name}")
    logger.debug(f"Параметры для модели: {model_prmt}")

    def objective(trial):
        try:
            n_clusters_min = opt_clusters_min_max_step[0]
            n_clusters_max = opt_clusters_min_max_step[1]
            n_clusters_step = opt_clusters_min_max_step[2]

            n_clusters = trial.suggest_int(
                "n_clusters",
                n_clusters_min,
                n_clusters_max,
                step=n_clusters_step,
            )
            logger.info(f"n_clusters = {n_clusters}:")

            model_prmt["n_clusters"] = n_clusters

            # test for repeated params
            # we use tuple because that's easier to add more params this way
            if (n_clusters) in param_history:
                logger.info(
                    f"Итерация {np.sum(iteration)} / {n_trials} завершена"
                    + "в связи с повтором параметров"
                )
                iteration.append(1)
                raise optuna.exceptions.TrialPruned()
            param_history.append(n_clusters)
        except Exception as ex:
            logger.exception(
                f"Ошибка при тесте повторяющихся параметров: {ex}"
            )

        np.random.seed(seed)
        try:
            ts = time.time()

            model = get_clustering_model(model_name, model_prmt)
            y_pred = model.fit_predict(data)
            score = model.get_metric_std(data, y_pred, best_cl)

            sec = time.time() - ts

            logger.info(
                f"Итерация {np.sum(iteration)} / {n_trials} завершена, "
                + f"обработка {n_clusters} кластеров заняла "
                + f"{np.round(sec/60,2)} минут"
            )
            iteration.append(1)

            return score

        except Exception as ex:
            logger.debug(
                f"Итерация {np.sum(iteration)} / {n_trials} завершена"
                + " из-за слишком большого  числа заданных кластеров"
            )
            iteration.append(1)
            logger.exception(ex)
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Лучшие параметры: {study.best_params}")

    return study.best_params
