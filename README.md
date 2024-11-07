# Pattern_or_coincidence

   Назначение проекта заключается в разработке автоматизированных процессов обработки данных (пайплайнов),
которые будут являться основой для создания одного из модулей будущей системы прогнозирования цен на биржевые активы.
Этот модуль позволит искать статистически значимые паттерны движения цен и использовать их для прогнозирования цен на биржевые активы.


## Getting started
1. Создаем виртуальное окружение на Python 3.9 с помощью Poetry
Устанасливаем Poetry в терминале pip install poetry 

или с помощью инструкции по ссылке https://python-poetry.org/docs/

2. В терминале заходим в папку с проектом (пример: /home/Nikolay/PycharmProjects/pattern_or_coincidence) и вводим команду:

poetry env use python3.9 

создается виртуальное окружение на основе python3.9

3. далее устанавливаем зависимости

poetry install

---------------

Для воспроизведения эксперимента используем следующие команды:

1. poetry update
2. dvc exp run



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │   
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


# pattern
