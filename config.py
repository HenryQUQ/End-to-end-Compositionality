from dataclasses import dataclass
import os
import torch


@dataclass
class Config:

    # Paths
    ROOT_PATH: str = os.path.dirname(os.path.abspath(__file__))

    CACHE_PATH: str = os.path.join(ROOT_PATH, "cache")

    DATA_PATH: str = os.path.join(ROOT_PATH, "dataset")
    MNIST_DATASET_PATH: str = os.path.join(DATA_PATH, "MNIST")

    LOG_FOLDER: str = os.path.join(ROOT_PATH, "logs")
    TRAINING_LOG_FOLDER: str = os.path.join(LOG_FOLDER, "train")
    EVALUATION_LOG_FOLDER: str = os.path.join(LOG_FOLDER, "eval")
