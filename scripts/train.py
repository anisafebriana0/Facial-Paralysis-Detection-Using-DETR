import os
import pytorch_lightning as pl
from clearml import Task
from roboflow import Roboflow
from transformers import DetrImageProcessor

from src.config.default import config
from src.data_handling.dataset import CocoDetection
from src.data_handling.dataloader import create_dataloaders
from src.model.detr import Detr

def setup_data():
    # Download dataset
    rf = Roboflow(api_key=config['data'].roboflow_api_key)
    project = rf.workspace(config['data'].workspace).project(config['data'].project)
    version = project.version(config['data'].version)
    dataset = version.download("coco")

    # Setup image processor
    image_processor = DetrImageProcessor.from_pretrained(config['model'].checkpoint)

    # Create datasets
    train_dataset = CocoDetection(
        os.path.join(dataset.location, "train"),
        image_processor,
        train=True
    )
    val_dataset = CocoDetection(
        os.path.join(dataset.location, "valid"),
        image_processor,
        train=False
    )
    test_dataset = CocoDetection(
        os.path.join(dataset.location, "test"),
        image_processor,
        train=False
    )

    return (
        create_dataloaders(train_dataset, val_dataset, test_dataset, image_processor),
        train_dataset.coco.cats
    )

def main():
    # Initialize ClearML
    task = Task.init(project_name='detr', task_name='detr_api')

    # Setup data
    (train_loader, val_loader, test_loader), categories = setup_data()
    id2label = {k: v['name'] for k, v in categories.items()}

    # Initialize model
    model = Detr(num_labels=len(id2label))

    # Train model
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=config['training'].max_epochs,
        gradient_clip_val=config['training'].gradient_clip_val,
        accumulate_grad_batches=config['training'].accumulate_grad_batches,
        log_every_n_steps=config['training'].log_every_n_steps
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Save model
    model.model.save_pretrained('detr_api')

if __name__ == "__main__":
    main() 