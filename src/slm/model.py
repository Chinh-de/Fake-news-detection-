"""
Integrated SLM (Small Language Model) wrapper — FTT edition (Feature Extractor Mode).
Fake-news detector based on the FTT-ACL23 BERT model:
    BERT (frozen) + Average Pooling + MLP binary head
    trained with instance-weighted cross-entropy loss.
Only the MLP head is trainable.
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
# Building-block layers (giữ nguyên)
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
# Core FTT Model (Feature Extractor: BERT frozen, MLP trainable)
# ============================================================

class FTTBertModel(nn.Module):
    """BERT (frozen) + Average Pooling + MLP binary classifier.

    - BERT is frozen (feature extractor)
    - MLP is trainable
    """
    def __init__(self, emb_dim: int, mlp_dims: list, dropout: float, bert_path: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # Đóng băng toàn bộ BERT (feature extractor)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.mlp = MLP(emb_dim, mlp_dims, dropout)

    def forward(self, **kwargs) -> torch.Tensor:
        inputs = kwargs["content"]           # (B, seq_len)
        masks  = kwargs["content_masks"]     # (B, seq_len)

        with torch.no_grad():  # BERT hoàn toàn không tính gradient
            bert_out = self.bert(inputs, attention_mask=masks)[0]   # (B, seq_len, H)

        # Average pooling over non-padded tokens
        mask_expanded = masks.unsqueeze(-1).expand(bert_out.size()).float()
        sum_embeddings = torch.sum(bert_out * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        pooled = sum_embeddings / sum_mask.clamp(min=1e-9)

        output = self.mlp(pooled)
        return torch.sigmoid(output.squeeze(1))


# ============================================================
# Instance-weighted BCE loss (giữ nguyên)
# ============================================================

class InstanceWeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, pred: torch.Tensor, ground: torch.Tensor,
                instance_weight: torch.Tensor) -> torch.Tensor:
        unweighted = self.loss(pred, ground)
        return torch.mean(unweighted * instance_weight)


# ============================================================
# IntegratedSLM wrapper (chỉ train MLP)
# ============================================================

class IntegratedSLM:
    """FTT-based SLM wrapper with inference and fine-tuning (Feature Extractor Mode)."""

    def __init__(self, model_path: str = MODEL_PATH, backend: str = None):
        self.backend = backend or SLM_BACKEND
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_model_path: Optional[str] = None

        resolved = model_path if os.path.exists(model_path) else model_path
        print(f"[FTT-SLM] Initialising from: {resolved}")
        self._init_model(resolved)

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
        print(f"[FTT-SLM] Model loaded on {self.device} (feature extractor mode: BERT frozen)")

    def _tokenise(self, texts: list, max_length: int = 170):
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ================================================================
    # Inference (giữ nguyên)
    # ================================================================
    def inference(self, text: str) -> tuple:
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

    # ================================================================
    # FTT-style weighted training (chỉ train MLP)
    # ================================================================
    def finetune_weighted(
        self,
        train_texts: list,
        train_labels: list,
        train_weights: list,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,          # Tăng learning rate vì chỉ train MLP (ít tham số)
        weight_decay: float = 1e-4,
        save_path: str = None,
    ) -> dict:
        """Train only the MLP head with instance-weighted BCE loss."""

        assert len(train_texts) == len(train_labels) == len(train_weights), \
            "texts, labels and weights must have the same length"

        # Chỉ lấy các tham số của MLP (phần classification head)
        trainable_params = self.model.mlp.parameters()
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        loss_fn = InstanceWeightedBCELoss()

        clean_texts = [preprocess_text(t) for t in train_texts]
        history = {"train_loss": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            idx = np.random.permutation(len(clean_texts)).tolist()
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_texts   = [clean_texts[i] for i in batch_idx]
                batch_labels  = torch.tensor([train_labels[i] for i in batch_idx], dtype=torch.float, device=self.device)
                batch_weights = torch.tensor([train_weights[i] for i in batch_idx], dtype=torch.float, device=self.device)

                enc = self._tokenise(batch_texts)
                pred = self.model(content=enc["input_ids"], content_masks=enc["attention_mask"])
                loss = loss_fn(pred, batch_labels, batch_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history["train_loss"].append(avg_loss)
            print(f"[FTT] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} (training only MLP)")

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(save_path, "parameter_bert.pkl"))

        if save_path:
            ckpt = os.path.join(save_path, "parameter_bert.pkl")
            if os.path.exists(ckpt):
                self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
                print(f"[FTT] Loaded best checkpoint from {ckpt}")

        self.model.eval()
        return {
            "trained": True,
            "samples": len(train_texts),
            "epochs_run": epoch + 1,
            "train_loss_history": history["train_loss"],
        }

    # ================================================================
    # MRCD fine-tuning on D_clean (chỉ train MLP)
    # ================================================================
    def finetune_on_clean(
        self,
        clean_samples: list,
        epochs: int = 2,
        batch_size: int = 32,
        lr: float = 1e-3,          # Learning rate cho MLP (cao hơn full fine-tune)
        weight_decay: float = 1e-4,
    ) -> dict:
        """Fine-tune only the MLP head on clean pool with confidence-based weights."""
        valid = [s for s in clean_samples if s.get("text") and s.get("label") in [0, 1]]
        if not valid:
            return {"trained": False, "reason": "no_valid_samples"}

        texts   = [preprocess_text(s["text"]) for s in valid]
        labels  = [int(s["label"]) for s in valid]
        weights = [float(s.get("conf_slm", 0.8)) for s in valid]

        loss_fn = InstanceWeightedBCELoss()
        # Chỉ train MLP
        optimizer = Adam(self.model.mlp.parameters(), lr=lr, weight_decay=weight_decay)
        total_loss, total_steps = 0.0, 0

        for _ in range(epochs):
            self.model.train()
            idx = np.random.permutation(len(texts)).tolist()
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_texts   = [texts[i] for i in batch_idx]
                batch_labels  = torch.tensor([labels[i] for i in batch_idx], dtype=torch.float, device=self.device)
                batch_weights = torch.tensor([weights[i] for i in batch_idx], dtype=torch.float, device=self.device)

                enc = self._tokenise(batch_texts)
                pred = self.model(content=enc["input_ids"], content_masks=enc["attention_mask"])
                loss = loss_fn(pred, batch_labels, batch_weights)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.mlp.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        self.model.eval()
        return {
            "trained": True,
            "samples": len(valid),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "avg_loss": total_loss / max(1, total_steps),
        }

    # ================================================================
    # Helper functions (giữ nguyên)
    # ================================================================
    def _eval_f1(self, texts: list, labels: list) -> float:
        from sklearn.metrics import f1_score
        preds = []
        results = self.inference_batch(texts, batch_size=64)
        for pred, conf, _ in results:
            preds.append(pred)
        return f1_score(labels, preds, average="macro", zero_division=0)

    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "parameter_bert.pkl"))
        print(f"[FTT-SLM] Checkpoint saved → {save_path}")

    def load(self, checkpoint_path: str):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"[FTT-SLM] Checkpoint loaded ← {checkpoint_path}")