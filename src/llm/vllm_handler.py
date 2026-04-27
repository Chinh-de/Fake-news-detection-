# src/llm/vllm_handler.py
from vllm import LLM, SamplingParams
from src.config import LLM_MODEL_NAME

class VLLMHandler:
    # Đảm bảo chỉ có một instance của model được tải trên toàn bộ pipeline
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print(f"Initializing vLLM with model: {LLM_MODEL_NAME}")
        # Khởi tạo LLM với các setting phù hợp cho nhu cầu của bạn
        self.llm = LLM(
            model=LLM_MODEL_NAME,
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
        )
        print("vLLM model loaded successfully")

    def generate(self, prompts: list[str]) -> list[str]:
        """Nhận vào một list các prompt (string), trả về list các câu trả lời (string)."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]