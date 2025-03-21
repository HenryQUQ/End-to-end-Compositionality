import os.path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributed as dist
from tqdm import tqdm
from config import Config
from torch.optim.lr_scheduler import StepLR
import wandb

# Hugging Face Accelerate
from accelerate import Accelerator
from dataloader.MINST.raw import load_MINST_dataloader
from tqdm import tqdm
import torchvision.utils as vutils

from pipelines.composition.pipeline_compostion import CompositionalPipeline

from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import torchvision

# Solve potential NCCL issues
# dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5400))

@dataclass
class Hyperparameters:
    IN_CHANNEL: int = 1
    VOCAB_SIZES = [100, 1000]
    PATCH_SIZE: int = 3
    STRIDE: int = 3
    IMAGE_SIZE: int = 81

    BATCH_SIZE: int = 8
    INITIAL_LR: float = 1e-4
    LEARNING_RATE_DECAY: float = 0.9
    LR_DECAY_STEP: int = 1000

    EPOCHS: int = 10

    CHECKPOINT_FREQ: int = 1


def train_one_epoch(
    pipeline,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
):
    pipeline.train()
    total_loss = 0.0
    num_samples = 0

    tqdm_loader = tqdm(
        enumerate(dataloader),
        desc=f"Epoch {epoch}",
        total=len(dataloader),
        disable=not accelerator.is_main_process,
    )
    for step, (images, _) in tqdm_loader:

        final_feat, info_list, reconstructed = pipeline(images)
        max_value = info_list[0]['composition_matrix'].max()
        min_value = info_list[0]['composition_matrix'].min()

        loss = 0
        for i in range(len(info_list)):
            loss += F.mse_loss(reconstructed[i], info_list[i]["featmap_bchw"])
        if accelerator.is_main_process and step % 100 == 0:
            torchvision.utils.save_image(images[0, 0, :, :], f"input.png")
            image = pipeline._visualize_composition_matrix(info_list[-1]['composition_matrix'], len(Hyperparameters.VOCAB_SIZES))
            torchvision.utils.save_image(image[0, 0, :, :], f"composition_matrix_{epoch}_{step}.png")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

        # wandb
        current_loss = loss.item()
        if accelerator.is_main_process:
            wandb.log(
                {
                    "train_loss": current_loss,
                    "epoch": epoch,
                    "step": step + 1 + epoch * len(dataloader),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

            tqdm_loader.set_postfix(loss=current_loss)

    avg_loss = total_loss / num_samples
    return avg_loss


def visualize_reconstruction(pipeline, dataloader, accelerator, epoch, max_samples=4):


    # Visualize the reconstruction for one batch
    for images, _ in dataloader:
        images, _ = images, _
        break
    if not accelerator.is_main_process:
        return
    images = images.to(accelerator.device)

    pipeline.eval()
    with torch.no_grad():
        final_feat, info_list, reconstructed = pipeline(images)

    n_visual = min(max_samples, images.size(0))

    gt_grid = vutils.make_grid(images[:n_visual], nrow=n_visual, normalize=True)
    recon_grid = vutils.make_grid(
        reconstructed[:n_visual], nrow=n_visual, normalize=True
    )

    wandb.log(
        {
            "GT_Images": wandb.Image(gt_grid, caption=f"Epoch {epoch} - Ground Truth"),
            "Reconstruction": wandb.Image(
                recon_grid, caption=f"Epoch {epoch} - Reconstruction"
            ),
        },
        step=epoch,
    )


def main():
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"Composition-E2E-{time}"

    accelerator = Accelerator()

    dataloader = load_MINST_dataloader(
        batch_size=Hyperparameters.BATCH_SIZE, num_workers=0
    )

    pipeline = CompositionalPipeline(
        in_channels=Hyperparameters.IN_CHANNEL,
        vocab_sizes=Hyperparameters.VOCAB_SIZES,
        patch_size=Hyperparameters.PATCH_SIZE,
        stride=Hyperparameters.STRIDE,
        image_size=Hyperparameters.IMAGE_SIZE,
    )

    # if True:
    #     pretained_path = r'/home/vieunite/Researchs/End-to-end-Compositionality/logs/train/Composition-E2E-2025-03-13_13-35-14/pipeline_final.pth'
    #     checkpoint = torch.load(pretained_path)
    #     pipeline.load_state_dict(checkpoint)


    optimizer = optim.Adam(pipeline.parameters(), lr=Hyperparameters.INITIAL_LR)

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Hyperparameters.LR_DECAY_STEP,
        gamma=Hyperparameters.LEARNING_RATE_DECAY,
    )

    pipeline, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipeline, optimizer, dataloader, lr_scheduler
    )

    checkpoint_dir = os.path.join(Config.TRAINING_LOG_FOLDER, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:
        wandb.init(project="Composition-E2E", name=name, dir=checkpoint_dir)

        # Log Hyperparameters
        wandb.config.update(asdict(Hyperparameters()))

    for epoch in range(Hyperparameters.EPOCHS):
        avg_loss = train_one_epoch(
            pipeline, dataloader, optimizer, lr_scheduler, accelerator, epoch
        )

        accelerator.print(
            f"Epoch [{epoch + 1}/{Hyperparameters.EPOCHS}] | Loss: {avg_loss:.6f}"
        )


        # visualize_reconstruction(pipeline, dataloader, accelerator, epoch + 1)

        if (epoch + 1) % Hyperparameters.CHECKPOINT_FREQ == 0:
            accelerator.wait_for_everyone()
            # unwrap model
            unwrapped_pipeline = accelerator.unwrap_model(pipeline)

            ckpt_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            accelerator.save(unwrapped_pipeline.state_dict(), ckpt_path)

    accelerator.wait_for_everyone()
    unwrapped_pipeline = accelerator.unwrap_model(pipeline)
    final_path = os.path.join(checkpoint_dir, "pipeline_final.pth")
    accelerator.save(unwrapped_pipeline.state_dict(), final_path)

    wandb.finish()


if __name__ == "__main__":
    main()
