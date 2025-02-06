import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection
from ..config.default import config

class Detr(pl.LightningModule):
    def __init__(self, num_labels: int):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=config['model'].checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, 
            pixel_mask=pixel_mask, 
            labels=labels
        )
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if "backbone" in n and p.requires_grad],
                "lr": config['training'].lr_backbone,
            },
        ]
        return torch.optim.AdamW(
            param_dicts,
            lr=config['training'].learning_rate,
            weight_decay=config['training'].weight_decay
        ) 