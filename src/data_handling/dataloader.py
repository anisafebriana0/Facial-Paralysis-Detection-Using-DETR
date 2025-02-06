from torch.utils.data import DataLoader
from .dataset import CocoDetection
from ..config.default import config

def collate_fn(batch, image_processor):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

def create_dataloaders(train_dataset, val_dataset, test_dataset, image_processor):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].batch_size,
        num_workers=config['training'].num_workers,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, image_processor)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].batch_size,
        num_workers=config['training'].num_workers,
        collate_fn=lambda b: collate_fn(b, image_processor)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].batch_size,
        num_workers=config['training'].num_workers,
        collate_fn=lambda b: collate_fn(b, image_processor)
    )

    return train_loader, val_loader, test_loader 