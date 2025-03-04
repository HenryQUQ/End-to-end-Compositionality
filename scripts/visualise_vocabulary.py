import os.path
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from config import Config
from torch.optim.lr_scheduler import StepLR
import wandb

# Hugging Face Accelerate
from accelerate import Accelerator
from dataloader.MINST.raw import load_MINST_dataloader
from tqdm import tqdm

from pipelines.composition.pipeline_compostion import CompositionalPipeline

from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Hyperparameters:
    IN_CHANNEL: int = 1
    VOCAB_SIZES = [100,]
    PATCH_SIZE: int = 3
    STRIDE: int = 3
    IMAGE_SIZE: int = 81



if __name__ == '__main__':

    pretrained_model_path = r'/home/vieunite/Researchs/End-to-end-Compositionality/logs/train/Composition-E2E-2025-02-27_18-20-42/pipeline_final.pth'
    pipeline = CompositionalPipeline(
        in_channels=Hyperparameters.IN_CHANNEL,
        vocab_sizes=Hyperparameters.VOCAB_SIZES,
        patch_size=Hyperparameters.PATCH_SIZE,
        stride=Hyperparameters.STRIDE,
        image_size=Hyperparameters.IMAGE_SIZE,
    )

    checkpoint = torch.load(pretrained_model_path)
    pipeline.load_state_dict(checkpoint)

    pipeline.eval()

    # Visualise the vocabulary
    vocs = pipeline.layers[0].vocabulary

    # for i, vocs in enumerate(voc_list):
    for j in range(vocs.shape[0]):
        torchvision.utils.save_image(vocs[j], f'{j}.png')


