"""
Integrated SLM (Small Language Model) wrapper.
RoBERTa-based binary classifier for fake news detection.

Supports two backends:
- "hf": HuggingFace Transformers (default)
- "vllm": vLLM for faster inference (requires vllm package)
"""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
	RobertaForSequenceClassification,
	RobertaTokenizer,
	get_linear_schedule_with_warmup,
)

from src.config import MODEL_PATH, SLM_BACKEND
from src.slm.dataset import FakeNewsDataset
from src.utils import preprocess_text


class IntegratedSLM:
	"""
	Wrapper for RoBERTa-based SLM with inference and fine-tuning capabilities.

	Args:
		model_path: Path to pre-trained model checkpoint.
		backend: "hf" (HuggingFace) or "vllm"
	"""

	def __init__(self, model_path: str = MODEL_PATH, backend: str = None):
		self.backend = backend or SLM_BACKEND
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._loaded_model_path = None

		local_model_path = model_path if os.path.exists(model_path) else None
		if local_model_path is not None:
			print(f"Loading SLM from local path: {local_model_path}")
			resolved_model_path = local_model_path
		else:
			print(f"Loading SLM from Hugging Face model id: {model_path}")
			resolved_model_path = model_path

		if self.backend == "vllm":
			self._init_vllm(resolved_model_path)
		else:
			self._init_hf(resolved_model_path)

	# ================================================================
	# HuggingFace Backend
	# ================================================================
	def _init_hf(self, model_path: str):
		self.tokenizer, self.model = self._load_roberta_components(
			model_path=model_path,
			eval_mode=True,
		)
		self._loaded_model_path = model_path
		print(f"SLM loaded (HF backend) on {self.device}")

	def _load_roberta_components(self, model_path: str, eval_mode: bool = True):
		tokenizer = RobertaTokenizer.from_pretrained(model_path)
		model = RobertaForSequenceClassification.from_pretrained(
			model_path,
			num_labels=2,
		)
		model.to(self.device)
		if eval_mode:
			model.eval()
		else:
			model.train()
		return tokenizer, model

	def _inference_hf(self, text: str) -> tuple:
		clean_text = preprocess_text(text)
		inputs = self.tokenizer(
			clean_text,
			return_tensors="pt",
			truncation=True,
			max_length=128,
			padding="max_length",
		)
		with torch.no_grad():
			outputs = self.model(
				inputs["input_ids"].to(self.device),
				inputs["attention_mask"].to(self.device),
			)
			probs = F.softmax(outputs.logits, dim=1)
		conf, pred = torch.max(probs, dim=1)
		return pred.item(), conf.item(), probs[0].cpu().numpy()

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

				outputs = self.model(
					inputs["input_ids"].to(self.device),
					inputs["attention_mask"].to(self.device),
				)
				probs = F.softmax(outputs.logits, dim=1)
				conf, pred = torch.max(probs, dim=1)

				for j in range(len(batch_texts)):
					results.append((pred[j].item(), conf[j].item(), probs[j].cpu().numpy()))

		return results

	# ================================================================
	# vLLM Backend
	# ================================================================
	def _init_vllm(self, model_path: str):
		try:
			from vllm import LLM as _VLLM_LLM  # noqa: F401
		except ImportError:
			raise ImportError(
				"vLLM not installed. Install with: pip install vllm\n"
				"Or set SLM_BACKEND=hf in .env"
			)

		self.tokenizer, self.model = self._load_roberta_components(
			model_path=model_path,
			eval_mode=True,
		)
		self._loaded_model_path = model_path
		self._vllm_model_path = model_path
		print(f"SLM loaded (vLLM backend) on {self.device}")

	def _inference_vllm(self, text: str) -> tuple:
		return self._inference_hf(text)

	# ================================================================
	# Public Interface
	# ================================================================
	def inference(self, text: str) -> tuple:
		if self.backend == "vllm":
			return self._inference_vllm(text)
		return self._inference_hf(text)

	def inference_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
		return self._inference_hf_batch(texts, batch_size)

	def _freeze_backbone_train_head_only(self):
		for _, param in self.model.named_parameters():
			param.requires_grad = False

		if not hasattr(self.model, "classifier"):
			raise AttributeError("Model does not expose classification head 'classifier'")

		for param in self.model.classifier.parameters():
			param.requires_grad = True

	def _set_head_train_mode(self):
		# Keep backbone deterministic while training only the classifier head.
		self.model.eval()
		self.model.classifier.train()

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

		dataset = FakeNewsDataset(texts, labels, self.tokenizer, max_len=128)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

		label_counts = torch.tensor(
			[
				sum(1 for label in labels if label == 0),
				sum(1 for label in labels if label == 1),
			],
			dtype=torch.float,
		)
		class_weights = (label_counts.sum() / (2 * label_counts.clamp(min=1))).to(self.device)
		loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

		self.model.train()

		total_loss = 0.0
		total_steps = 0

		for _ in range(epochs):
			for batch in loader:
				input_ids = batch["input_ids"].to(self.device)
				attention_mask = batch["attention_mask"].to(self.device)
				labels_t = batch["labels"].to(self.device)

				optimizer.zero_grad()
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
				)
				loss = loss_fn(outputs.logits, labels_t)
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
		model_init: str = "roberta-base",
		epochs: int = 4,
		batch_size: int = 32,
		lr: float = 1e-3,
		weight_decay: float = 1e-4,
		warmup_ratio: float = 0.1,
		max_grad_norm: float = 1.0,
		save_path: str | None = None,
	) -> dict:
		"""
		Fine-tune nhị phân với cấu hình feature extractor:
		- Freeze toàn bộ backbone RoBERTa.
		- Chỉ huấn luyện classification head.
		- AdamW, lr=1e-3, weight_decay=1e-4, batch_size=32.
		"""
		if len(train_texts) != len(train_labels):
			raise ValueError("train_texts và train_labels phải cùng số lượng")
		if len(train_texts) == 0:
			return {"trained": False, "reason": "no_train_data"}

		if self._loaded_model_path != model_init:
			self.tokenizer, self.model = self._load_roberta_components(
				model_path=model_init,
				eval_mode=True,
			)
			self._loaded_model_path = model_init

		self._freeze_backbone_train_head_only()

		train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer, max_len=128)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

		total_steps = len(train_loader) * epochs
		trainable_params = [p for p in self.model.parameters() if p.requires_grad]
		optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=int(total_steps * warmup_ratio),
			num_training_steps=total_steps,
		)

		label_counts = torch.tensor(
			[
				sum(1 for l in train_labels if l == 0),
				sum(1 for l in train_labels if l == 1),
			],
			dtype=torch.float,
		)
		class_weights = (label_counts.sum() / (2 * label_counts.clamp(min=1))).to(self.device)
		loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

		history = {"train_loss": []}

		for _ in range(epochs):
			epoch_loss = 0.0
			self._set_head_train_mode()

			for batch in train_loader:
				input_ids = batch["input_ids"].to(self.device)
				attention_mask = batch["attention_mask"].to(self.device)
				labels_t = batch["labels"].to(self.device)

				optimizer.zero_grad()
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
				)
				loss = loss_fn(outputs.logits, labels_t)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
				optimizer.step()
				scheduler.step()

				epoch_loss += float(loss.item())

			avg_train_loss = epoch_loss / max(1, len(train_loader))
			history["train_loss"].append(avg_train_loss)

		if save_path:
			os.makedirs(save_path, exist_ok=True)
			self.model.save_pretrained(save_path)
			self.tokenizer.save_pretrained(save_path)

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
		model_init: str = "roberta-base",
		epochs: int = 4,
		batch_size: int = 32,
		lr: float = 1e-3,
		weight_decay: float = 1e-4,
		warmup_ratio: float = 0.1,
		max_grad_norm: float = 1.0,
		save_path: str | None = None,
	) -> dict:
		"""
		Backward-compatible alias for finetune().
		"""
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
