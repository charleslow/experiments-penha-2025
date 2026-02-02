"""Bi-encoder models for search and recommendation tasks."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sentence_transformers import SentenceTransformer
from torchmetrics import MeanMetric

from src.data.datamodule import SearchBatch, RecBatch, MultiTaskBatch

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with in-batch negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query_embeddings: (batch_size, embedding_dim)
            key_embeddings: (batch_size, embedding_dim) or (num_keys, embedding_dim)
            labels: Optional positive indices (default: diagonal)

        Returns:
            Scalar loss tensor
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        key_embeddings = F.normalize(key_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(query_embeddings, key_embeddings.t()) / self.temperature

        # Default: diagonal positives (in-batch)
        if labels is None:
            labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device)

        loss = F.cross_entropy(similarity, labels)
        return loss


class BiEncoderModule(L.LightningModule):
    """
    Bi-encoder module for search and recommendation tasks.

    Supports three training modes:
    - search: Query-to-item contrastive learning
    - rec: Item-to-item co-occurrence contrastive learning
    - multi_task: Joint training on both tasks
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        task: str = "multi_task",
        temperature: float = 0.07,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        search_loss_weight: float = 1.0,
        rec_loss_weight: float = 1.0,
    ):
        """
        Initialize the bi-encoder.

        Args:
            model_name: Name of the sentence transformer model
            task: Task type ("search", "rec", or "multi_task")
            temperature: Temperature for contrastive loss
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            search_loss_weight: Weight for search loss in multi-task
            rec_loss_weight: Weight for rec loss in multi-task
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.task = task
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.search_loss_weight = search_loss_weight
        self.rec_loss_weight = rec_loss_weight

        # Load sentence transformer
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Loss function
        self.loss_fn = ContrastiveLoss(temperature=temperature)

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_search_loss = MeanMetric()
        self.train_rec_loss = MeanMetric()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts to embeddings."""
        return self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass - encode texts."""
        return self.encode(texts)

    def compute_search_loss(self, batch: SearchBatch) -> torch.Tensor:
        """Compute contrastive loss for search task."""
        query_emb = self.encode(batch.queries)
        item_emb = self.encode(batch.items)
        labels = batch.labels.to(query_emb.device)
        return self.loss_fn(query_emb, item_emb, labels)

    def compute_rec_loss(self, batch: RecBatch) -> torch.Tensor:
        """Compute contrastive loss for recommendation task."""
        item1_emb = self.encode(batch.items1)
        item2_emb = self.encode(batch.items2)
        labels = batch.labels.to(item1_emb.device)
        return self.loss_fn(item1_emb, item2_emb, labels)

    def training_step(self, batch, batch_idx):
        """Training step."""
        if self.task == "search":
            loss = self.compute_search_loss(batch)
            self.train_loss(loss)
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        elif self.task == "rec":
            loss = self.compute_rec_loss(batch)
            self.train_loss(loss)
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        elif self.task == "multi_task":
            # batch is a dict with "search" and "rec" keys
            search_loss = self.compute_search_loss(batch["search"])
            rec_loss = self.compute_rec_loss(batch["rec"])

            loss = (
                self.search_loss_weight * search_loss
                + self.rec_loss_weight * rec_loss
            )

            self.train_loss(loss)
            self.train_search_loss(search_loss)
            self.train_rec_loss(rec_loss)

            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/search_loss", self.train_search_loss, on_step=True, on_epoch=True)
            self.log("train/rec_loss", self.train_rec_loss, on_step=True, on_epoch=True)

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        if self.task == "search":
            loss = self.compute_search_loss(batch)
        elif self.task == "rec":
            loss = self.compute_rec_loss(batch)
        elif self.task == "multi_task":
            if dataloader_idx == 0:  # search
                loss = self.compute_search_loss(batch)
            else:  # rec
                loss = self.compute_rec_loss(batch)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        """Reset metrics at epoch start."""
        self.train_loss.reset()
        self.train_search_loss.reset()
        self.train_rec_loss.reset()

    def on_validation_epoch_start(self):
        """Reset metrics at validation start."""
        self.val_loss.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def get_item_embeddings(
        self,
        items: List[str],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Get embeddings for a list of items.

        Args:
            items: List of item texts
            batch_size: Batch size for encoding

        Returns:
            Tensor of shape (num_items, embedding_dim)
        """
        embeddings = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            emb = self.encode(batch)
            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)


def train_bi_encoder(
    datamodule: L.LightningDataModule,
    model_name: str = "all-MiniLM-L6-v2",
    task: str = "multi_task",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    temperature: float = 0.07,
    accelerator: str = "auto",
    devices: int = 1,
    output_dir: Optional[str] = None,
) -> BiEncoderModule:
    """
    Train a bi-encoder model.

    Args:
        datamodule: Lightning DataModule
        model_name: Name of the sentence transformer model
        task: Task type ("search", "rec", or "multi_task")
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature for contrastive loss
        accelerator: Accelerator to use
        devices: Number of devices
        output_dir: Optional output directory for checkpoints

    Returns:
        Trained BiEncoderModule
    """
    model = BiEncoderModule(
        model_name=model_name,
        task=task,
        temperature=temperature,
        learning_rate=learning_rate,
    )

    callbacks = []
    if output_dir:
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f"bi_encoder_{task}_{{epoch}}",
                save_top_k=1,
                monitor="val/loss",
                mode="min",
            )
        )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
    )

    trainer.fit(model, datamodule=datamodule)

    return model
