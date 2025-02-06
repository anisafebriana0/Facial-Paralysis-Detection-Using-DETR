import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from roboflow import Roboflow
from transformers import DetrImageProcessor, DetrForObjectDetection
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from config import *
from data.dataset import CocoDetection, collate_fn

# implementasi training model Detection Transformer (DETR) menggunakan framework PyTorch Lightning.
# pytorch_lightning Untuk mengatur training, validasi, logging, dan pengelolaan model
# DetrForObjectDetection: Model DETR dari HuggingFace Transformers.
class Detr(pl.LightningModule):
    def __init__(self, num_labels=91, lr=1e-2, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        
        # Initialize loss tracking
        self.train_losses = []
        
        # Create directory for saving logs
        self.log_dir = Path('training_logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created log directory: {self.log_dir}")

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        # Log losses
        loss_dict = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'total_loss': outputs.loss.item(),
            'loss_ce': outputs.loss_dict['loss_ce'].item(),
            'loss_bbox': outputs.loss_dict['loss_bbox'].item(),
            'loss_giou': outputs.loss_dict['loss_giou'].item()
        }
        self.train_losses.append(loss_dict)
        
        # Log to tensorboard or other logger
        for name, value in outputs.loss_dict.items():
            self.log(f"train_{name}", value.item())
        
        return outputs.loss

    def on_train_epoch_end(self):
        # Convert losses to DataFrame
        df = pd.DataFrame(self.train_losses)
        
        # Save losses to CSV
        csv_path = self.log_dir / 'training_losses.csv'
        df.to_csv(csv_path, index=False)
        
        # Calculate and print epoch statistics
        epoch_stats = df[df['epoch'] == self.current_epoch].mean()
        print(f"\nEpoch {self.current_epoch} statistics:")
        print(f"Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"Classification Loss: {epoch_stats['loss_ce']:.4f}")
        print(f"Bbox Loss: {epoch_stats['loss_bbox']:.4f}")
        print(f"GIoU Loss: {epoch_stats['loss_giou']:.4f}")

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_path, num_labels=91):
        model = cls(num_labels=num_labels)
        model.model = DetrForObjectDetection.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        return model

def setup_data():
    # Download dataset
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("anisa-liipc").project("paralysis-face")
    version = project.version(12)
    dataset = version.download("coco")

    # Setup directories
    train_dir = os.path.join(dataset.location, "train")
    val_dir = os.path.join(dataset.location, "valid")
    test_dir = os.path.join(dataset.location, "test")

    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

    # Create datasets
    train_dataset = CocoDetection(train_dir, image_processor, train=True)
    val_dataset = CocoDetection(val_dir, image_processor, train=False)
    test_dataset = CocoDetection(test_dir, image_processor, train=False)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset, 
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.coco.cats

def main():
    # Replace ClearML with TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="detr_training",
        default_hp_metric=False
    )

    # Setup data
    train_dataloader, val_dataloader, test_dataloader, categories = setup_data()
    id2label = {k: v['name'] for k,v in categories.items()}

    # Initialize model with all parameters in constructor
    # id2label mapping antara ID kelas ke nama kelas objek.
    model = Detr(
        num_labels=len(id2label),
        lr=1e-2,
        lr_backbone=1e-5,
        weight_decay=1e-4
    )

    # Train model with logger
    trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu", 
        max_epochs=MAX_EPOCHS, 
        gradient_clip_val=0.1, 
        accumulate_grad_batches=8, 
        log_every_n_steps=5,
        logger=logger
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Save model
    model.model.save_pretrained(MODEL_PATH)

if __name__ == "__main__":
    main() 