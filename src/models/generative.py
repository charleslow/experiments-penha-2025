"""Generative retrieval model using T5 with semantic IDs."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from torchmetrics import MeanMetric

logger = logging.getLogger(__name__)


class GenerativeRetrievalModule(L.LightningModule):
    """
    Generative retrieval model using T5 with semantic ID vocabulary.

    The model takes a query as input and generates semantic IDs as output.
    The semantic ID vocabulary is added to the model's vocabulary.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        n_hierarchies: int = 3,
        codebook_size: int = 256,
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_input_length: int = 128,
        max_output_length: int = 32,
        num_beams: int = 10,
    ):
        """
        Initialize the generative retrieval model.

        Args:
            model_name: Name of the T5 model to use
            n_hierarchies: Number of hierarchical levels in semantic IDs
            codebook_size: Size of each codebook
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for scheduler
            weight_decay: Weight decay
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            num_beams: Number of beams for beam search
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.n_hierarchies = n_hierarchies
        self.codebook_size = codebook_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Add semantic ID tokens to vocabulary
        self._add_semantic_id_tokens()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def _add_semantic_id_tokens(self):
        """Add semantic ID tokens to the vocabulary."""
        # Create tokens for each hierarchy and code
        new_tokens = []
        for h in range(self.n_hierarchies):
            for c in range(self.codebook_size):
                token = f"<SID_{h}_{c}>"
                new_tokens.append(token)

        # Add special tokens for hierarchy separators
        new_tokens.extend([f"<H{h}>" for h in range(self.n_hierarchies)])

        # Add tokens to tokenizer
        num_added = self.tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {num_added} semantic ID tokens to vocabulary")

        # Resize model embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Store token mappings
        self.sid_token_start = self.tokenizer.convert_tokens_to_ids(f"<SID_0_0>")

    def semantic_ids_to_tokens(self, semantic_ids: torch.Tensor) -> List[str]:
        """
        Convert semantic IDs to token strings.

        Args:
            semantic_ids: Tensor of shape (batch_size, n_hierarchies)

        Returns:
            List of token strings
        """
        if semantic_ids.dim() == 1:
            semantic_ids = semantic_ids.unsqueeze(0)

        token_strs = []
        for ids in semantic_ids:
            tokens = []
            for h, code in enumerate(ids):
                tokens.append(f"<SID_{h}_{code.item()}>")
            token_strs.append(" ".join(tokens))

        return token_strs

    def tokens_to_semantic_ids(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert generated token IDs back to semantic IDs.

        Args:
            token_ids: Generated token IDs from the model

        Returns:
            Tuple of (semantic_ids, valid_mask)
        """
        batch_size = token_ids.shape[0]
        semantic_ids = torch.zeros(batch_size, self.n_hierarchies, dtype=torch.long)
        valid_mask = torch.ones(batch_size, dtype=torch.bool)

        for b in range(batch_size):
            decoded = self.tokenizer.decode(token_ids[b], skip_special_tokens=True)
            tokens = decoded.split()

            h = 0
            for token in tokens:
                if token.startswith("<SID_") and token.endswith(">"):
                    try:
                        parts = token[5:-1].split("_")
                        hierarchy = int(parts[0])
                        code = int(parts[1])

                        if hierarchy == h and code < self.codebook_size:
                            semantic_ids[b, h] = code
                            h += 1

                        if h >= self.n_hierarchies:
                            break
                    except (ValueError, IndexError):
                        continue

            if h < self.n_hierarchies:
                valid_mask[b] = False

        return semantic_ids, valid_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs (for training)

        Returns:
            Dictionary with loss and/or logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def training_step(self, batch, batch_idx):
        """Training step."""
        queries, semantic_ids, item_ids = batch

        # Tokenize queries
        inputs = self.tokenizer(
            queries,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Convert semantic IDs to target tokens
        target_strs = self.semantic_ids_to_tokens(semantic_ids)
        targets = self.tokenizer(
            target_strs,
            max_length=self.max_output_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        outputs = self.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=targets.input_ids,
        )

        loss = outputs["loss"]
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        queries, semantic_ids, item_ids = batch

        # Tokenize queries
        inputs = self.tokenizer(
            queries,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Convert semantic IDs to target tokens
        target_strs = self.semantic_ids_to_tokens(semantic_ids)
        targets = self.tokenizer(
            target_strs,
            max_length=self.max_output_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        outputs = self.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=targets.input_ids,
        )

        loss = outputs["loss"]
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def generate(
        self,
        queries: List[str],
        num_return_sequences: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic IDs for queries.

        Args:
            queries: List of query strings
            num_return_sequences: Number of sequences to generate per query

        Returns:
            Tuple of (semantic_ids, scores)
            - semantic_ids: (batch_size, num_return_sequences, n_hierarchies)
            - scores: (batch_size, num_return_sequences)
        """
        # Tokenize queries
        inputs = self.tokenizer(
            queries,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate with beam search
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.max_output_length,
            num_beams=max(self.num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Parse generated sequences
        batch_size = len(queries)
        all_semantic_ids = []
        all_scores = []

        sequences = outputs.sequences.view(batch_size, num_return_sequences, -1)
        scores = outputs.sequences_scores.view(batch_size, num_return_sequences)

        for b in range(batch_size):
            batch_ids = []
            for s in range(num_return_sequences):
                sid, valid = self.tokens_to_semantic_ids(sequences[b, s : s + 1])
                batch_ids.append(sid[0])
            all_semantic_ids.append(torch.stack(batch_ids))
            all_scores.append(scores[b])

        return torch.stack(all_semantic_ids), torch.stack(all_scores)

    def on_train_epoch_start(self):
        """Reset metrics at epoch start."""
        self.train_loss.reset()

    def on_validation_epoch_start(self):
        """Reset metrics at validation start."""
        self.val_loss.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Linear warmup then decay
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def train_generative_model(
    datamodule: L.LightningDataModule,
    model_name: str = "google/flan-t5-small",
    n_hierarchies: int = 3,
    codebook_size: int = 256,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    accelerator: str = "auto",
    devices: int = 1,
    output_dir: Optional[str] = None,
) -> GenerativeRetrievalModule:
    """
    Train a generative retrieval model.

    Args:
        datamodule: Lightning DataModule
        model_name: Name of the T5 model
        n_hierarchies: Number of hierarchical levels
        codebook_size: Codebook size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        accelerator: Accelerator to use
        devices: Number of devices
        output_dir: Optional output directory

    Returns:
        Trained GenerativeRetrievalModule
    """
    model = GenerativeRetrievalModule(
        model_name=model_name,
        n_hierarchies=n_hierarchies,
        codebook_size=codebook_size,
        learning_rate=learning_rate,
    )

    callbacks = []
    if output_dir:
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="generative_{epoch}",
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
