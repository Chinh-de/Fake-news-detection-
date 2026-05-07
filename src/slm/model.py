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
import numpy as np
from transformers import (
	RobertaForSequenceClassification,
	RobertaTokenizer,
	get_linear_schedule_with_warmup,
	TrainingArguments,
	Trainer,
	EarlyStoppingCallback,
)
import evaluate

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
		warmup_steps: int = 500,
		early_stopping_patience: int = 2,
	) -> dict:
		"""
		Fine-tune trên D_clean với HuggingFace Trainer API.
		Sử dụng feature extractor approach (đóng băng backbone, train head).
		"""
		valid_samples = [
			s
			for s in clean_samples
			if s.get("text") is not None and s.get("label") in [0, 1]
		]
		if not valid_samples:
			return {"trained": False, "reason": "no_valid_samples"}

		print(f"\n[SLM Fine-tune on clean] INIT: clean_samples={len(valid_samples)}, epochs={epochs}, batch_size={batch_size}, lr={lr}")

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
		epochs: int = 50,
		batch_size: int = 32,
		lr: float = 1e-3,
		weight_decay: float = 0.01,
		warmup_steps: int = 500,
		early_stopping_patience: int = 2,
		val_texts: list[str] | None = None,
		val_labels: list[int] | None = None,
		save_path: str | None = None,
	) -> dict:
		"""
		Fine-tune with HuggingFace Trainer API (feature extractor approach):
		- Freeze toàn bộ backbone RoBERTa.
		- Chỉ huấn luyện classification head.
		- Hỗ trợ validation set và early stopping.
		- AdamW, eval_strategy="epoch", save_strategy="epoch".
		"""
		if len(train_texts) != len(train_labels):
			raise ValueError("train_texts và train_labels phải cùng số lượng")
		if len(train_texts) == 0:
			return {"trained": False, "reason": "no_train_data"}
			
		print(f"\n[SLM Fine-tune] INIT: train_samples={len(train_texts)}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
		
		# Load model
		if self._loaded_model_path != model_init:
			self.tokenizer, self.model = self._load_roberta_components(
				model_path=model_init,
				eval_mode=False,
			)
			self._loaded_model_path = model_init

		# Freeze backbone, train only head
		self._freeze_backbone_train_head_only()

		# Prepare datasets
		train_texts_clean = [preprocess_text(t) for t in train_texts]
		train_dataset = FakeNewsDataset(train_texts_clean, train_labels, self.tokenizer, max_len=128)
		
		eval_dataset = None
		if val_texts is not None and val_labels is not None:
			if len(val_texts) != len(val_labels):
				raise ValueError("val_texts và val_labels phải cùng số lượng")
			val_texts_clean = [preprocess_text(t) for t in val_texts]
			eval_dataset = FakeNewsDataset(val_texts_clean, val_labels, self.tokenizer, max_len=128)
			print(f"Validation dataset: {len(eval_dataset)} samples")

		# Compute metrics function
		metric_f1 = evaluate.load("f1")
		metric_acc = evaluate.load("accuracy")
		
		def compute_metrics(eval_pred):
			logits, labels = eval_pred
			predictions = np.argmax(logits, axis=-1)
			f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
			acc = metric_acc.compute(predictions=predictions, references=labels)
			return {**f1, **acc}

		# Training arguments
		output_dir = save_path or "./results_slm_finetune"
		training_args = TrainingArguments(
			output_dir=output_dir,
			num_train_epochs=epochs,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=64,
			learning_rate=lr,
			weight_decay=weight_decay,
			optim="adamw_torch",
			warmup_steps=warmup_steps,
			logging_dir="./logs_slm",
			logging_steps=50,
			eval_strategy="epoch" if eval_dataset else "no",
			save_strategy="epoch" if eval_dataset else "no",
			load_best_model_at_end=eval_dataset is not None,
			metric_for_best_model="eval_loss" if eval_dataset else None,
			greater_is_better=False,
			save_total_limit=2,
		)

		# Trainer with callbacks
		callbacks = []
		if eval_dataset:
			callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			compute_metrics=compute_metrics if eval_dataset else None,
			callbacks=callbacks,
		)

		# Train
		trainer.train()
		self.model.eval()

		result = {
			"trained": True,
			"samples": len(train_texts),
			"epochs": epochs,
			"batch_size": batch_size,
			"lr": lr,
			"weight_decay": weight_decay,
			"warmup_steps": warmup_steps,
			"val_samples": len(val_texts) if val_texts else None,
		}
		if save_path:
			result["save_path"] = save_path

		return result

	def fnetune(
		self,
		train_texts: list[str],
		train_labels: list[int],
		model_init: str = "roberta-base",
		epochs: int = 50,
		batch_size: int = 32,
		lr: float = 1e-3,
		weight_decay: float = 0.01,
		warmup_steps: int = 500,
		early_stopping_patience: int = 2,
		val_texts: list[str] | None = None,
		val_labels: list[int] | None = None,
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
			warmup_steps=warmup_steps,
			early_stopping_patience=early_stopping_patience,
			val_texts=val_texts,
			val_labels=val_labels,
			save_path=save_path,
		)
