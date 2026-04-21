"""
Integrated SLM (Small Language Model) wrapper.
EANN-based binary classifier for fake news detection (from FTT-ACL23).

Supports HuggingFace Transformers backend.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from src.config import MODEL_PATH, SLM_BACKEND
from src.slm.dataset import FakeNewsDataset
from src.utils import preprocess_text


# Copy necessary classes from FTT-ACL23
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MaskAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class EANNBertModel(torch.nn.Module):
    """ EANN Model Using BERT as Text Encoder
    """
    def __init__(self, emb_dim, mlp_dims, dropout, bert_path, domain_num):
        super(EANNBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.classifier = MLP(emb_dim, mlp_dims, dropout)
        self.domain_classifier = nn.Sequential(MLP(emb_dim, mlp_dims, dropout, False), torch.nn.ReLU(),
                        torch.nn.Linear(mlp_dims[-1], domain_num))
        self.attention = MaskAttention(emb_dim)
    
    def forward(self, alpha, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        bert_feature = self.bert(inputs, attention_mask = masks)[0]
        bert_feature, _ = self.attention(bert_feature, masks)

        output = self.classifier(bert_feature)
        reverse = ReverseLayerF.apply

        reverse_res = reverse(bert_feature, alpha)

        domain_pred = self.domain_classifier(reverse(bert_feature, alpha))
        return torch.sigmoid(output.squeeze(1)), domain_pred


class IntegratedSLM:
    """
    Wrapper for EANN-based SLM with inference and fine-tuning capabilities.

    Args:
        model_path: Path to pre-trained BERT model.
        domain_num: Number of domains for adversarial training.
    """

    def __init__(self, model_path: str = MODEL_PATH, backend: str = None, domain_num: int = 4):
        self.backend = backend or SLM_BACKEND
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.domain_num = domain_num
        self._loaded_model_path = None

        local_model_path = model_path if os.path.exists(model_path) else None
        if local_model_path is not None:
            print(f"Loading SLM from local path: {local_model_path}")
            resolved_model_path = local_model_path
        else:
            print(f"Loading SLM from Hugging Face model id: {model_path}")
            resolved_model_path = model_path

        self._init_hf(resolved_model_path)

    def _init_hf(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = EANNBertModel(
            emb_dim=768,
            mlp_dims=[256],
            dropout=0.1,
            bert_path=model_path,
            domain_num=self.domain_num
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded_model_path = model_path
        print(f"SLM loaded (EANN HF backend) on {self.device}")

    def _inference_hf(self, text: str) -> tuple:
        clean_text = preprocess_text(text)
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output, _ = self.model(alpha=-1, content=inputs['input_ids'], content_masks=inputs['attention_mask'])
            probs = output.cpu().numpy()
        pred = int(probs[0] > 0.5)
        conf = max(probs[0], 1 - probs[0])
        return pred, conf, [probs[0], 1 - probs[0]]

    def _inference_hf_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        clean_texts = [preprocess_text(t) for t in texts]
        results = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs, _ = self.model(alpha=-1, content=inputs['input_ids'], content_masks=inputs['attention_mask'])
                probs = outputs.cpu().numpy()

                for prob in probs:
                    pred = int(prob > 0.5)
                    conf = max(prob, 1 - prob)
                    results.append((pred, conf, [prob, 1 - prob]))

        return results

    # ================================================================
    # Public Interface
    # ================================================================
    def inference(self, text: str) -> tuple:
        return self._inference_hf(text)

    def inference_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        return self._inference_hf_batch(texts, batch_size)

    def finetune_on_clean(
        self,
        clean_samples: list,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> dict:
        valid_samples = [
            s
            for s in clean_samples
            if s.get("text") is not None and s.get("label") in [0, 1]
        ]
        if not valid_samples:
            return {"trained": False, "reason": "no_valid_samples"}

        texts = [preprocess_text(s["text"]) for s in valid_samples]
        labels = [int(s["label"]) for s in valid_samples]
        domains = [s.get("domain", 0) for s in valid_samples]  # Default domain if not provided

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCELoss()

        total_loss = 0.0
        total_steps = 0

        for _ in range(epochs):
            self.model.train()
            alpha = max(2. / (1. + np.exp(-10 * _ / epochs)) - 1, 1e-1)

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.float).to(self.device)
                batch_domains = torch.tensor(domains[i:i+batch_size], dtype=torch.long).to(self.device)

                inputs = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                pred, domain_pred = self.model(alpha=alpha, content=inputs['input_ids'], content_masks=inputs['attention_mask'])
                loss = loss_fn(pred, batch_labels)
                loss_adv = F.nll_loss(F.log_softmax(domain_pred, dim=1), batch_domains)
                loss = loss + loss_adv

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_steps += 1

        self.model.eval()
        avg_loss = total_loss / max(1, total_steps)
        return {
            "trained": True,
            "samples": len(valid_samples),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "avg_loss": avg_loss,
        }

    def finetune(
        self,
        train_texts: list[str],
        train_labels: list[int],
        model_init: str = "bert-base-uncased",
        epochs: int = 4,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_path: str | None = None,
    ) -> dict:
        if len(train_texts) != len(train_labels):
            raise ValueError("train_texts vÃ  train_labels pháº£i cÃ¹ng sá»‘ lÆ°á»£ng")
        if len(train_texts) == 0:
            return {"trained": False, "reason": "no_train_data"}

        if self._loaded_model_path != model_init:
            self._init_hf(model_init)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        history = {"train_loss": []}
        loss_fn = torch.nn.BCELoss()

        for _ in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            alpha = max(2. / (1. + np.exp(-10 * _ / epochs)) - 1, 1e-1)

            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i+batch_size]
                batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.float).to(self.device)
                batch_domains = torch.tensor([0] * len(batch_texts), dtype=torch.long).to(self.device)  # Default domain

                inputs = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                pred, domain_pred = self.model(alpha=alpha, content=inputs['input_ids'], content_masks=inputs['attention_mask'])
                loss = loss_fn(pred, batch_labels)
                loss_adv = F.nll_loss(F.log_softmax(domain_pred, dim=1), batch_domains)
                loss = loss + loss_adv

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += float(loss.item())

            avg_train_loss = epoch_loss / max(1, len(train_texts) // batch_size)
            history["train_loss"].append(avg_train_loss)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_path, 'parameter_eann.pkl'))

        self.model.eval()

        result = {
            "trained": True,
            "samples": len(train_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_loss_history": history["train_loss"],
        }
        if save_path:
            result["save_path"] = save_path

        return result

    def fnetune(
        self,
        train_texts: list[str],
        train_labels: list[int],
        model_init: str = "bert-base-uncased",
        epochs: int = 4,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_path: str | None = None,
    ) -> dict:
        return self.finetune(
            train_texts=train_texts,
            train_labels=train_labels,
            model_init=model_init,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
            save_path=save_path,
        )

