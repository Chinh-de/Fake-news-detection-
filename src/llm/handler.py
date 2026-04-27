"""
Generic Local LLM handler using HuggingFace Transformers.
Model name loaded from .env (LLM_MODEL_NAME), default: Meta-Llama-3-8B-Instruct.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_BACKEND,
)
from src.llm.base import BaseLLM


class LocalLLM(BaseLLM):
    """
    Generic local LLM running via HuggingFace Transformers.
    Supports any causal LM (Llama, Qwen, Mistral, etc.)
    """

    def __init__(self, model_name: str = None):
        """
        Khởi tạo lớp LocalLLM.
        
         
        1. Xác định tên mô hình từ tham số truyền vào hoặc file .env.
        2. Khởi tạo thuộc tính model và tokenizer là None.
        3. Gọi phương thức _load_model() để nạp mô hình vào bộ nhớ.
        """
        self.model_name = model_name or LLM_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """
        Nạp mô hình và tokenizer từ HuggingFace AutoClasses.
        
         
        1. Thử nạp AutoTokenizer từ model_name.
        2. Đảm bảo có pad_token (nếu chưa có thì dùng eos_token).
        3. Nạp AutoModelForCausalLM với cấu hình tự động chọn device (GPU/CPU) và kiểu dữ liệu.
        4. Gán pad_token_id cho cấu hình tạo văn bản của mô hình.
        5. In thông báo xác nhận đã nạp thành công.
        6. Bắt lỗi nếu quá trình nạp thất bại và đưa ra RuntimeError.
        """
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

            print(f" Loaded LLM: {self.model_name}")
        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")

    def generate_text(
        self, prompt: str, max_output_tokens: int = LLM_MAX_NEW_TOKENS
    ) -> str:
        """
        Sinh văn bản từ một prompt đầu vào.
        
         
        1. Kiểm tra mô hình đã được khởi tạo chưa.
        2. Tạo cấu trúc tin nhắn hội thoại (chat message) với prompt người dùng.
        3. Sử dụng chat template của tokenizer để định dạng prompt.
        4. Tokenize văn bản đã định dạng và chuyển sang device của mô hình.
        5. Sử dụng torch.inference_mode() để tối ưu bộ nhớ khi dự đoán.
        6. Gọi mô hình tạo văn bản (model.generate) với các tham số: temperature, top_p, max_new_tokens.
        7. Giải mã (decode) các token đã tạo ra thành chuỗi văn bản, loại bỏ các ký tự đặc biệt.
        8. Trả về kết quả đã được cắt bỏ khoảng trắng thừa.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model is not initialized.")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max(1, int(max_output_tokens)),
                do_sample=False,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
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


# # ============================================================
# # Singleton Accessor
# # ============================================================
# _current_llm = None


# def get_llm(model_name: str = None) -> BaseLLM:
#     """
#     Lấy hoặc tạo mới một singleton Global LLM.
    
     
#     1. Kiểm tra biến toàn cục _current_llm.
#     2. Nếu chưa tồn tại (_current_llm is None), khởi tạo instance LocalLLM mới.
#     3. Trả về instance LLM hiện tại.
    
#     Args:
#         model_name: Ghi đè tên mô hình. Nếu None, sử dụng LLM_MODEL_NAME từ .env.
#     """
#     global _current_llm
#     if _current_llm is None:
#         _current_llm = LocalLLM(model_name)
#     return _current_llm
# ============================================================
# vLLM Backend (optional dependency)
# ============================================================
try:
    from vllm import LLM, SamplingParams

    class VLLMHandler(BaseLLM):
        """vLLM backend for high-performance inference."""

        def __init__(self, model_name: str = None):
            self.model_name = model_name or LLM_MODEL_NAME
            self.llm = None
            self.sampling_params = None
            self._load_model()

        def _load_model(self):
            print(f"[vLLM] Initializing with model: {self.model_name}")
            self.llm = LLM(
                model=self.model_name,
                max_model_len=8192,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                swap_space=4,
                enforce_eager=True,
            )
            self.sampling_params = SamplingParams(
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_NEW_TOKENS,
                top_p=LLM_TOP_P,
            )
            print("[vLLM] Model loaded successfully")

        def generate_text(self, prompt: str, max_output_tokens: int = LLM_MAX_NEW_TOKENS) -> str:
            if self.llm is None:
                raise RuntimeError("vLLM model not initialized.")
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()

except ImportError:
    VLLMHandler = None


# ============================================================
# Singleton Accessor with Backend Selection
# ============================================================
_current_llm = None

def get_llm(model_name: str = None) -> BaseLLM:
    """
    Get singleton LLM instance based on LLM_BACKEND config.
    Supported backends: "hf" (default) or "vllm".
    """
    global _current_llm
    if _current_llm is None:
        if LLM_BACKEND == "vllm":
            if VLLMHandler is None:
                raise ImportError("vLLM is not installed. Run: pip install vllm")
            _current_llm = VLLMHandler(model_name)
        else:
            _current_llm = LocalLLM(model_name)
    return _current_llm