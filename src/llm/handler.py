"""
Generic Local LLM handler using HuggingFace Transformers.
Model name loaded from .env (LLM_MODEL_NAME), default: Meta-Llama-3-8B-Instruct.
"""

import logging
import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from src.llm.base import BaseLLM

# Thiết lập logging
logger = logging.getLogger(__name__)


class LocalLLM(BaseLLM):
    """
    Generic local LLM running via HuggingFace Transformers.
    Supports any causal LM (Llama, Qwen, Mistral, etc.)
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or LLM_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

            logger.info(f"Loaded LLM: {self.model_name}")
        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")

    def generate_text(
        self,
        prompt: str,
        max_output_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Sinh văn bản từ một prompt đầu vào.

        Args:
            prompt: Chuỗi prompt đầu vào.
            max_output_tokens: Số token tối đa sinh thêm.
            temperature: Nhiệt độ sampling (ghi đè giá trị config nếu được cung cấp).
            top_p: Top-p sampling (ghi đè config nếu cung cấp).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model is not initialized.")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Sửa lỗi: do_sample phụ thuộc vào temperature
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        top_p_val = top_p if top_p is not None else LLM_TOP_P
        do_sample = temp > 0.0  # nếu temperature > 0 thì bật sampling

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max(1, int(max_output_tokens)),
                do_sample=do_sample,
                temperature=temp if do_sample else 1.0,  # temperature chỉ dùng khi do_sample=True
                top_p=top_p_val if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_only_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        response = self.tokenizer.batch_decode(
            gen_only_ids, skip_special_tokens=True
        )[0]
        return response.strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_output_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Sinh văn bản cho batch các prompt, tự động xử lý theo lô nhỏ để tránh OOM.

        Args:
            prompts: Danh sách các prompt đầu vào.
            max_output_tokens: Số token tối đa sinh thêm cho mỗi prompt.
            temperature: Nhiệt độ sampling.
            top_p: Top-p sampling.
            batch_size: Kích thước lô nhỏ để xử lý (mặc định 8).

        Returns:
            Danh sách các chuỗi sinh ra tương ứng.
        """
        results = []
        total = len(prompts)
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        top_p_val = top_p if top_p is not None else LLM_TOP_P
        do_sample = temp > 0.0

        for i in range(0, total, batch_size):
            batch_prompts = prompts[i:i + batch_size]
            # Tạo chat template cho từng prompt
            batch_texts = []
            for p in batch_prompts:
                messages = [{"role": "user", "content": p}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)

            # Tokenize batch (padding=True để đồng nhất độ dài)
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max(1, int(max_output_tokens)),
                    do_sample=do_sample,
                    temperature=temp if do_sample else 1.0,
                    top_p=top_p_val if do_sample else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Loại bỏ phần input tokens
            gen_only_ids = []
            for inp, out in zip(inputs.input_ids, output_ids):
                # Tìm độ dài thực của input (bỏ qua padding)
                inp_len = (inp != self.tokenizer.pad_token_id).sum().item()
                gen_only_ids.append(out[inp_len:])

            batch_responses = self.tokenizer.batch_decode(
                gen_only_ids, skip_special_tokens=True
            )
            results.extend([resp.strip() for resp in batch_responses])

            # Log tiến độ
            logger.info(f"Generated batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")

        return results


# ============================================================
# Singleton Accessor with reset capability
# ============================================================
_current_llm = None


def get_llm(model_name: str = None) -> BaseLLM:
    global _current_llm
    if _current_llm is None:
        _current_llm = LocalLLM(model_name)
    return _current_llm


def reset_llm():
    """Reset singleton LLM instance, useful for switching models."""
    global _current_llm
    _current_llm = None


def set_llm(model_name: str):
    """Force create a new LLM instance with given model name."""
    global _current_llm
    _current_llm = LocalLLM(model_name)
    return _current_llm