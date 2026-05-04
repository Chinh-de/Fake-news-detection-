"""
Integrated SLM (Small Language Model) wrapper — EANN edition.
Fake-news detector with Domain Adaptation (Event Adversarial Neural Network).
- BERT backbone (frozen except last layer)
- Attention pooling
- MLP binary classifier
- Domain classifier with Gradient Reversal Layer
Training: binary CE + adversarial domain loss.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import BertModel, BertTokenizer

from src.config import MODEL_PATH, SLM_BACKEND
from src.utils import preprocess_text


# ============================================================
# Helper layers
# ============================================================

class MLP(nn.Module):
    """Multi-layer perceptron with optional output layer."""
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


class MaskAttention(nn.Module):
    """Simple attention pooling over sequence with mask."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> tuple:
        scores = self.attn(sequence).squeeze(-1)           # (B, L)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)                 # (B, L)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class ReverseLayerF(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ============================================================
# EANN Model (BERT + attention + domain classifier)
# ============================================================

class EANNBertModel(nn.Module):
    """BERT (frozen except last layer) + Attention + MLP classifier + Domain classifier."""
    def __init__(self, emb_dim: int, mlp_dims: list, dropout: float, bert_path: str, domain_num: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)

        # Freeze BERT except the last transformer layer
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.attention = MaskAttention(emb_dim)
        self.classifier = MLP(emb_dim, mlp_dims, dropout, output_layer=True)
        self.domain_classifier = nn.Sequential(
            MLP(emb_dim, mlp_dims, dropout, output_layer=False),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], domain_num)
        )

    def forward(self, alpha: float, **kwargs) -> tuple:
        inputs = kwargs["content"]
        masks  = kwargs["content_masks"]

        bert_out = self.bert(inputs, attention_mask=masks)[0]   # (B, L, H)
        pooled, _ = self.attention(bert_out, masks)            # (B, H)

        # Binary classification
        logits = self.classifier(pooled)
        pred = torch.sigmoid(logits.squeeze(1))

        # Domain adaptation with gradient reversal
        reverse_feat = ReverseLayerF.apply(pooled, alpha)
        domain_logits = self.domain_classifier(reverse_feat)
        return pred, domain_logits


# ============================================================
# Instance‑weighted BCE loss (for clean samples with confidence)
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
# Integrated SLM Wrapper (EANN version)
# ============================================================

class IntegratedSLM:
    """EANN‑based SLM wrapper with inference, initial fine‑tuning, and MRCD fine‑tuning."""

    def __init__(self, model_path: str = MODEL_PATH, backend: str = None, domain_num: int = 4):
        self.backend = backend or SLM_BACKEND
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.domain_num = domain_num
        self._loaded_model_path: Optional[str] = None

        resolved = model_path if os.path.exists(model_path) else model_path
        print(f"[EANN-SLM] Initialising from: {resolved}")
        self._init_model(resolved)

    def _init_model(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = EANNBertModel(
            emb_dim=768,
            mlp_dims=[384],
            dropout=0.2,
            bert_path=model_path,
            domain_num=self.domain_num,
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded_model_path = model_path
        print(f"[EANN-SLM] Model loaded on {self.device} (domain_num={self.domain_num})")

    def _tokenise(self, texts: list, max_length: int = 170):
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            'content': enc['input_ids'].to(self.device),
            'content_masks': enc['attention_mask'].to(self.device)
        }

    # ================================================================
    # Inference
    # ================================================================
    def inference(self, text: str) -> tuple:
        clean = preprocess_text(text)
        enc   = self._tokenise([clean])
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(alpha=-1.0, **enc)
            prob = pred.item()
        pred_label = int(prob > 0.5)
        conf = max(prob, 1.0 - prob)
        return pred_label, conf, [prob, 1.0 - prob]

    def inference_batch(self, texts: list, batch_size: int = 32) -> list:
        clean_texts = [preprocess_text(t) for t in texts]
        results = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch = clean_texts[i : i + batch_size]
                enc   = self._tokenise(batch)
                preds, _ = self.model(alpha=-1.0, **enc)
                probs = preds.cpu().numpy()
                for p in probs:
                    results.append((int(p > 0.5), float(max(p, 1.0 - p)), [float(p), float(1.0 - p)]))
        return results

    # ================================================================
    # Initial fine‑tuning from pre‑trained BERT (with optional domain labels)
    # ================================================================
    def finetune(
        self,
        train_texts: list,
        train_labels: list,
        train_domain_labels: Optional[list] = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        lambda_adv: float = 0.1,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Fine‑tune the EANN model from pre‑trained BERT.
        - If train_domain_labels is provided, uses domain adaptation loss.
        - Otherwise, domain loss is disabled.
        """
        assert len(train_texts) == len(train_labels), "Mismatched texts and labels"
        has_domain = train_domain_labels is not None
        if has_domain:
            assert len(train_texts) == len(train_domain_labels), "Domain labels length mismatch"

        clean_texts = [preprocess_text(t) for t in train_texts]
        labels_t = torch.tensor(train_labels, dtype=torch.float, device=self.device)
        # Instance weights default to 1 (no weighting in initial training)
        weights = torch.ones(len(train_texts), dtype=torch.float, device=self.device)

        if has_domain:
            domain_t = torch.tensor(train_domain_labels, dtype=torch.long, device=self.device)

        # Trainable parameters: all parameters of the model (BERT last layer, attention, heads)
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_bce = InstanceWeightedBCELoss()
        loss_domain = nn.CrossEntropyLoss()

        history = {"train_loss": []}

        for epoch in range(epochs):
            self.model.train()
            idx = np.random.permutation(len(clean_texts)).tolist()
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_texts = [clean_texts[i] for i in batch_idx]
                batch_labels = labels_t[batch_idx]
                batch_weights = weights[batch_idx]

                enc = self._tokenise(batch_texts)
                # Increase alpha gradually from 0 to 1 over epochs
                alpha = max(2. / (1. + np.exp(-10 * epoch / epochs)) - 1, 1e-1)
                pred, domain_pred = self.model(alpha=alpha, **enc)

                loss = loss_bce(pred, batch_labels, batch_weights)
                if has_domain:
                    batch_domain = domain_t[batch_idx]
                    loss_adv = loss_domain(domain_pred, batch_domain)
                    loss = loss + lambda_adv * loss_adv

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history["train_loss"].append(avg_loss)
            print(f"[EANN] Initial fine‑tune epoch {epoch+1}/{epochs} | loss={avg_loss:.4f}")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_path, "parameter_eann.pkl"))
            self.tokenizer.save_pretrained(save_path)
            print(f"[EANN] Initial model saved to {save_path}")

        self.model.eval()
        return {
            "trained": True,
            "samples": len(train_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_adv": lambda_adv if has_domain else 0,
            "train_loss_history": history["train_loss"],
        }

    # ================================================================
    # MRCD fine‑tuning on clean pool (only MLP and domain head)
    # ================================================================
    def finetune_on_clean(
        self,
        clean_samples: list,
        epochs: int = 2,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_adv: float = 0.1,
    ) -> dict:
        """
        Fine‑tune only the MLP and domain classifier heads on the clean pool.
        Each sample should contain 'text', 'label', 'conf_slm', and optionally 'domain_label'.
        """
        valid = [s for s in clean_samples if s.get("text") and s.get("label") in [0, 1]]
        if not valid:
            return {"trained": False, "reason": "no_valid_samples"}

        texts   = [preprocess_text(s["text"]) for s in valid]
        labels  = torch.tensor([int(s["label"]) for s in valid], dtype=torch.float, device=self.device)
        weights = torch.tensor([float(s.get("conf_slm", 0.8)) for s in valid], dtype=torch.float, device=self.device)

        has_domain = all("domain_label" in s for s in valid)
        if has_domain:
            domain_labels = torch.tensor([int(s["domain_label"]) for s in valid], dtype=torch.long, device=self.device)
        else:
            domain_labels = torch.zeros(len(valid), dtype=torch.long, device=self.device)

        # Trainable parameters: only classifier and domain_classifier (heads)
        trainable_params = list(self.model.classifier.parameters()) + list(self.model.domain_classifier.parameters())
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        loss_bce = InstanceWeightedBCELoss()
        loss_domain = nn.CrossEntropyLoss()

        total_loss, total_steps = 0.0, 0

        for epoch in range(epochs):
            self.model.train()
            idx = np.random.permutation(len(texts)).tolist()
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_texts   = [texts[i] for i in batch_idx]
                batch_labels  = labels[batch_idx]
                batch_weights = weights[batch_idx]
                batch_domain  = domain_labels[batch_idx]

                enc = self._tokenise(batch_texts)
                alpha = 1.0  # fixed for MRCD fine‑tuning
                pred, domain_pred = self.model(alpha=alpha, **enc)

                loss_cls = loss_bce(pred, batch_labels, batch_weights)
                loss_adv = loss_domain(domain_pred, batch_domain) if has_domain else torch.tensor(0.0, device=self.device)
                loss = loss_cls + lambda_adv * loss_adv

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
            "lambda_adv": lambda_adv,
            "avg_loss": total_loss / max(1, total_steps),
        }

    # ================================================================
    # Save / Load
    # ================================================================
    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "parameter_eann.pkl"))
        self.tokenizer.save_pretrained(save_path)
        print(f"[EANN-SLM] Checkpoint saved → {save_path}")

    def load(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[EANN-SLM] Checkpoint loaded ← {checkpoint_path}")