import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from models.gap_net import Sentinel2ResUNet
from datasets.raster_datasets import S2S1GapFractionTileFolderDataset
from datasets.npz_dataset import CachedNPZGapFractionDataset
import config


def sse_and_count(pred: torch.Tensor, target: torch.Tensor):
    diff = pred - target
    sse = torch.sum(diff * diff).item()
    n = diff.numel()
    return sse, n


# -------------------------
# Logging setup
# -------------------------
log_path = getattr(config, "LOG_PATH", "logs/train_gap_spline.log")
Path(log_path).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
logging.info("Starting gap-fraction spline training run")

tb_log_dir = getattr(config, "TB_LOG_DIR", "runs/gap_spline")
writer = SummaryWriter(log_dir=tb_log_dir)


# -------------------------
# Device + threads
# -------------------------
torch.set_num_threads(getattr(config, "NUM_THREADS", 30))

requested_device = getattr(config, "DEVICE", "cuda")
if requested_device == "cuda" and not torch.cuda.is_available():
    device = torch.device("cpu")
    logging.warning("CUDA requested but not available, falling back to CPU.")
else:
    device = torch.device(requested_device)

logging.info(f"Using device: {device}")


# -------------------------
# Datasets
# -------------------------
train_ds = CachedNPZGapFractionDataset(config.TRAIN_CACHE_ROOT)
val_ds = CachedNPZGapFractionDataset(config.VAL_CACHE_ROOT)

logging.info(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")


# -------------------------
# DataLoaders
# -------------------------
num_workers = getattr(config, "NUM_WORKERS", 4)


val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=False,
    persistent_workers=(num_workers > 0),
)


# -------------------------
# Model / loss / optimizer
# -------------------------
model = Sentinel2ResUNet(
    in_channels=config.NUM_BANDS,   # should be 220
    s1_in_channels=config.S1_BANDS, # should be 3
).to(device)

criterion = torch.nn.SmoothL1Loss(beta=0.05).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

accum_steps = getattr(config, "ACCUM_STEPS", 4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-4,
    min_lr=1e-6,
)

best_val_rmse = float("inf")

model_out = getattr(config, "MODEL_OUT", "models/output/gap_spline_best.pth")
Path(model_out).parent.mkdir(parents=True, exist_ok=True)



# -------------------------
# Training Loop
# -------------------------
global_step = 0

for epoch in range(config.EPOCHS):

    samples_per_epoch = getattr(config, "TRAIN_SAMPLES_PER_EPOCH", len(train_ds))
    samples_per_epoch = min(samples_per_epoch, len(train_ds))

    epoch_indices = torch.randperm(len(train_ds))[:samples_per_epoch].tolist()
    train_epoch_ds = Subset(train_ds, epoch_indices)

    train_loader = DataLoader(
        train_epoch_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=False,
    )

    model.train()

    train_loss_sum = 0.0
    train_sse = 0.0
    train_n = 0
    train_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        s2 = batch["s2"].to(device, non_blocking=True).float()
        s1 = batch["s1"].to(device, non_blocking=True).float()
        y  = batch["label"].to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(s2, s1)
            raw_loss = criterion(pred, y)
            loss = raw_loss / accum_steps

        scaler.scale(loss).backward()

        train_loss_sum += raw_loss.item()

        pred_rmse = pred.detach().float().clamp(0.0, 1.0)
        diff = pred_rmse - y.float()
        train_sse += torch.sum(diff * diff).item()
        train_n += diff.numel()
        train_batches += 1

        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))

        if do_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            writer.add_scalar("GradNorm/train", float(grad_norm), global_step)

        if step % 100 == 0:
            logging.info(
                f"Epoch {epoch + 1} Step {step}/{len(train_loader)}"
            )

    avg_train_loss = train_loss_sum / max(train_batches, 1)
    avg_train_rmse = (train_sse / max(train_n, 1)) ** 0.5

    # -------------------------
    # Validation
    # -------------------------
    val_every = getattr(config, "VAL_EVERY", 1)
    do_val = ((epoch + 1) % val_every == 0) or ((epoch + 1) == config.EPOCHS)
    
    if do_val:
        model.eval()

        val_loss_sum = 0.0
        val_batches = 0
        val_sse = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                s2 = batch["s2"].to(device, non_blocking=True).float()
                s1 = batch["s1"].to(device, non_blocking=True).float()
                y  = batch["label"].to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(s2, s1)
                    vloss = criterion(pred, y)

                val_loss_sum += vloss.item()
                val_batches += 1

                pred_rmse = pred.float().clamp(0.0, 1.0)
                batch_sse, batch_n = sse_and_count(pred_rmse, y.float())
                val_sse += batch_sse
                val_n += batch_n

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_rmse = (val_sse / max(val_n, 1)) ** 0.5

        scheduler.step(avg_val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("RMSE/train", avg_train_rmse, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("RMSE/val", avg_val_rmse, epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)

        logging.info(
            f"Epoch {epoch + 1}/{config.EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_rmse": best_val_rmse,
            }

            torch.save(checkpoint, model_out)
            logging.info(f"Saved new best model to {model_out}")
    else:
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("RMSE/train", avg_train_rmse, epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)

        logging.info(
            f"Epoch {epoch + 1}/{config.EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f} | "
            f"Val skipped | "
            f"LR: {current_lr:.2e}"
        )

writer.close()