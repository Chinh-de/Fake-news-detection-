"""
Integrated SLM (Small Language Model) wrapper — FTT edition.
Fake-news detector based on the FTT-ACL23 BERT model:
    BERT (fully trainable) + Average Pooling + MLP binary head
    trained with instance-weighted cross-entropy loss.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertModel, BertTokenizer

from src.config import MODEL_PATH, SLM_BACKEND
from src.utils import preprocess_text


# ============================================================
# Building-block layers
# ============================================================

class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations and dropout."""

    def __init__(self, input_dim: int, embed_dims: list, dropout: float, output_layer: bool = True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================
# Core FTT Model (exactly as described in FTT-ACL23 paper)
# ============================================================

class FTTBertModel(nn.Module):
    """BERT + Average Pooling + MLP binary classifier (FTT architecture).

    As described in the paper:
    - BERT is trainable (no freezing)
    - Output = average representation of non-padded tokens
    - MLP with sigmoid for final prediction
    """

    def __init__(self, emb_dim: int, mlp_dims: list, dropout: float, bert_path: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # BERT is trainable (no freezing) - as per paper
        # (All parameters have requires_grad=True by default)

        self.mlp = MLP(emb_dim, mlp_dims, dropout)

    def forward(self, **kwargs) -> torch.Tensor:
        inputs = kwargs["content"]           # (B, seq_len)
        masks  = kwargs["content_masks"]     # (B, seq_len)

        bert_out = self.bert(inputs, attention_mask=masks)[0]   # (B, seq_len, H)

        # Average pooling over non-padded tokens (as per paper)
        # Expand mask to same dimension as bert_out
        mask_expanded = masks.unsqueeze(-1).expand(bert_out.size()).float()  # (B, seq_len, H)
        sum_embeddings = torch.sum(bert_out * mask_expanded, dim=1)          # (B, H)
        sum_mask = torch.sum(mask_expanded, dim=1)                           # (B, H)
        pooled = sum_embeddings / sum_mask.clamp(min=1e-9)                   # (B, H)

        output = self.mlp(pooled)               # (B, 1)
        return torch.sigmoid(output.squeeze(1)) # (B,)


# ============================================================
# Instance-weighted BCE loss (identical to paper)
# ============================================================

class InstanceWeightedBCELoss(nn.Module):
    """BCE loss with per-sample weighting for FTT-style training."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, pred: torch.Tensor, ground: torch.Tensor,
                instance_weight: torch.Tensor) -> torch.Tensor:
        unweighted = self.loss(pred, ground)
        return torch.mean(unweighted * instance_weight)


# ============================================================
# IntegratedSLM — public wrapper used by MRCD pipeline
# ============================================================

class IntegratedSLM:
    """FTT-based SLM wrapper with inference and fine-tuning for MRCD.

    Args:
        model_path: HuggingFace model id or local path to BERT checkpoint.
        backend: Ignored (kept for API compatibility).
    """

    def __init__(self, model_path: str = MODEL_PATH, backend: str = None):
        self.backend = backend or SLM_BACKEND
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_model_path: Optional[str] = None

        resolved = model_path if os.path.exists(model_path) else model_path
        print(f"[FTT-SLM] Initialising from: {resolved}")
        self._init_model(resolved)

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _init_model(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = FTTBertModel(
            emb_dim=768,
            mlp_dims=[384],   # matches FTT-ACL23 default
            dropout=0.2,
            bert_path=model_path,
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded_model_path = model_path
        print(f"[FTT-SLM] Model loaded on {self.device}")

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    def _tokenise(self, texts: list, max_length: int = 170):
        """Tokenise a list of texts and return input tensors on device."""
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Inference  (unchanged public API for MRCD pipeline)
    # ------------------------------------------------------------------

    def inference(self, text: str) -> tuple:
        """Single-sample inference.

        Returns:
            (pred: int, conf: float, probs: [p_fake, p_real])
        """
        clean = preprocess_text(text)
        enc   = self._tokenise([clean])
        self.model.eval()
        with torch.no_grad():
            prob = self.model(
                content=enc["input_ids"],
                content_masks=enc["attention_mask"],
            ).item()
        pred = int(prob > 0.5)
        conf = max(prob, 1.0 - prob)
        return pred, conf, [prob, 1.0 - prob]

    def inference_batch(self, texts: list, batch_size: int = 32) -> list:
        """Batch inference.

        Returns:
            List of (pred, conf, probs) tuples, one per input text.
        """
        clean_texts = [preprocess_text(t) for t in texts]
        results = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch = clean_texts[i : i + batch_size]
                enc   = self._tokenise(batch)
                probs = self.model(
                    content=enc["input_ids"],
                    content_masks=enc["attention_mask"],
                ).cpu().numpy()
                for p in probs:
                    results.append((int(p > 0.5), float(max(p, 1.0 - p)), [float(p), float(1.0 - p)]))
        return results

    # ------------------------------------------------------------------
    # FTT-style weighted training  (Phase 1 — called by the notebook)
    # ------------------------------------------------------------------

    def finetune_weighted(
        self,
        train_texts: list,
        train_labels: list,
        train_weights: list,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 2e-5,
        weight_decay: float = 5e-5,
        early_stop: int = 5,
        val_texts: list = None,
        val_labels: list = None,
        save_path: str = None,
    ) -> dict:
        """Train the FTT detector with instance-weighted BCE loss.

        This implements the training procedure from FTT paper:
        - Minimises weighted cross-entropy loss (InstanceWeightedBCELoss).
        - Adam optimiser, lr=2e-5, early-stop on validation macro-F1.

        Args:
            train_texts:   Raw news texts for training.
            train_labels:  Binary labels (0=real, 1=fake).
            train_weights: Per-sample weights from FTT reweighting pipeline.
            val_texts / val_labels: Optional validation set for early-stop.
            save_path: Directory to save best model checkpoint.

        Returns:
            dict with training statistics.
        """
        from sklearn.metrics import f1_score

        assert len(train_texts) == len(train_labels) == len(train_weights), \
            "texts, labels and weights must have the same length"

        loss_fn   = InstanceWeightedBCELoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Preprocess texts
        clean_texts = [preprocess_text(t) for t in train_texts]
        clean_val   = [preprocess_text(t) for t in val_texts] if val_texts else None

        best_f1   = 0.0
        no_improve = 0
        history   = {"train_loss": [], "val_f1": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            # Shuffle order each epoch
            idx = np.random.permutation(len(clean_texts)).tolist()
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                batch_texts   = [clean_texts[i] for i in batch_idx]
                batch_labels  = torch.tensor(
                    [train_labels[i]  for i in batch_idx], dtype=torch.float, device=self.device
                )
                batch_weights = torch.tensor(
                    [train_weights[i] for i in batch_idx], dtype=torch.float, device=self.device
                )

                enc  = self._tokenise(batch_texts)
                pred = self.model(content=enc["input_ids"], content_masks=enc["attention_mask"])
                loss = loss_fn(pred, batch_labels, batch_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history["train_loss"].append(avg_loss)

            # Validation / early-stop
            val_f1 = 0.0
            if clean_val is not None and val_labels is not None:
                val_f1 = self._eval_f1(clean_val, val_labels)
                history["val_f1"].append(val_f1)
                print(f"[FTT] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | val_f1={val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    no_improve = 0
                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(save_path, "parameter_bert.pkl"),
                        )
                else:
                    no_improve += 1
                    if no_improve >= early_stop:
                        print(f"[FTT] Early stop at epoch {epoch+1}")
                        break
            else:
                print(f"[FTT] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f}")

        # Load best checkpoint if saved
        ckpt = os.path.join(save_path, "parameter_bert.pkl") if save_path else None
        if ckpt and os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f"[FTT] Loaded best checkpoint from {ckpt}")

        self.model.eval()
        return {
            "trained": True,
            "samples": len(train_texts),
            "epochs_run": epoch + 1,
            "best_val_f1": best_f1,
            "train_loss_history": history["train_loss"],
            "val_f1_history": history.get("val_f1", []),
        }

    def _eval_f1(self, texts: list, labels: list, batch_size: int = 64) -> float:
        """Compute macro-F1 on a given text/label set."""
        from sklearn.metrics import f1_score as sk_f1
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenise(batch)
                probs = self.model(
                    content=enc["input_ids"],
                    content_masks=enc["attention_mask"],
                ).cpu().numpy()
                preds.extend([int(p > 0.5) for p in probs])
        return sk_f1(labels, preds, average="macro")

    # ------------------------------------------------------------------
    # MRCD fine-tuning on D_clean  (called by pipeline/finetune.py)
    # ------------------------------------------------------------------

    def finetune_on_clean(
        self,
        clean_samples: list,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        val_texts: list = None,
        val_labels: list = None,
    ) -> dict:
        """Fine-tune on MRCD clean pool with confidence-based weights.

        Each sample in clean_samples is a dict with keys 'text', 'label',
        and optionally 'conf_slm' for confidence-weighted training.

        Uses a very low learning rate (2e-6) to preserve FTT-learned
        temporal representations while adapting to clean pool supervision.
        """
        valid = [s for s in clean_samples if s.get("text") and s.get("label") in [0, 1]]
        if not valid:
            return {"trained": False, "reason": "no_valid_samples"}

        texts   = [preprocess_text(s["text"])  for s in valid]
        labels  = [int(s["label"]) for s in valid]
        # Use SLM confidence as sample weight (preserves FTT spirit)
        weights = [float(s.get("conf_slm", 0.8)) for s in valid]

        # Save pre-finetune state for possible rollback
        pre_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        pre_f1 = None
        if val_texts is not None and val_labels is not None:
            pre_f1 = self._eval_f1(val_texts, val_labels)

        loss_fn   = InstanceWeightedBCELoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_loss, total_steps = 0.0, 0

        for _ in range(epochs):
            self.model.train()
            idx = np.random.permutation(len(texts)).tolist()
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                batch_texts   = [texts[i]   for i in batch_idx]
                batch_labels  = torch.tensor([labels[i]  for i in batch_idx], dtype=torch.float, device=self.device)
                batch_weights = torch.tensor([weights[i] for i in batch_idx], dtype=torch.float, device=self.device)

                enc  = self._tokenise(batch_texts)
                pred = self.model(content=enc["input_ids"], content_masks=enc["attention_mask"])
                loss = loss_fn(pred, batch_labels, batch_weights)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss  += loss.item()
                total_steps += 1

        # Rollback if validation F1 dropped (prevents catastrophic forgetting)
        rolled_back = False
        post_f1 = None
        if pre_f1 is not None:
            self.model.eval()
            post_f1 = self._eval_f1(val_texts, val_labels)
            if post_f1 < pre_f1 - 0.01:  # allow tiny margin
                self.model.load_state_dict(pre_state)
                rolled_back = True
                print(f"[MRCD] Rollback! val_f1 dropped {pre_f1:.4f} → {post_f1:.4f}")
            else:
                print(f"[MRCD] Finetune kept: val_f1 {pre_f1:.4f} → {post_f1:.4f}")

        self.model.eval()
        return {
            "trained": True,
            "samples": len(valid),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "avg_loss": total_loss / max(1, total_steps),
            "pre_f1": pre_f1,
            "post_f1": post_f1,
            "rolled_back": rolled_back,
        }

    def _eval_f1(self, texts: list, labels: list) -> float:
        """Compute binary F1 on given texts/labels (used for rollback check)."""
        from sklearn.metrics import f1_score
        preds = []
        results = self.inference_batch(texts, batch_size=64)
        for pred, conf, _ in results:
            preds.append(pred)
        return f1_score(labels, preds, average="binary", zero_division=0)

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def save(self, save_path: str):
        """Save model weights to directory."""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "parameter_bert.pkl"))
        print(f"[FTT-SLM] Checkpoint saved → {save_path}")

    def load(self, checkpoint_path: str):
        """Load model weights from a .pkl file."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"[FTT-SLM] Checkpoint loaded ← {checkpoint_path}")