"""
Label management for MRCD Framework.

- LLM classification: chỉ dùng "Real" / "Fake" → parse đơn giản
- Synonym labels: chỉ dùng để gán nhãn giả cho demonstrations (training task)
"""

import re
import random


# ============================================================
# LLM Classification Labels (dùng trong prompt)
# LLM chỉ được trả về 1 trong 2 giá trị này
# ============================================================
LLM_LABEL_REAL = "Real"
LLM_LABEL_FAKE = "Fake"

# ============================================================
# Synonym Labels (CHỈ dùng cho gán nhãn giả demo)
# Không đưa vào prompt LLM classification
# ============================================================
REAL_SYNONYM_LABELS = [
    "Authentic", "Reliable", "True", "Genuine", "Credible",
    "Fact-based", "Verified", "Accurate", "Trustworthy", "Factual",
    "Legitimate", "Proven", "Substantiated", "Valid", "Confirmed",
    "Authoritative", "Honest", "Objective",
]

FAKE_SYNONYM_LABELS = [
    "Hoax", "False", "Fabricated", "Untrue", "Misleading",
    "Bogus", "Inaccurate", "Dubious", "Deceptive", "Unverified",
    "Rumor", "Satire", "Propaganda", "Manipulated", "Distorted",
    "Baseless", "Phony", "Clickbait", "Disinformation", "Misinformation",
]

ALL_SYNONYM_LABELS = REAL_SYNONYM_LABELS + FAKE_SYNONYM_LABELS


# ============================================================
# Demo Label Generation (for training/retrieval demos)
# ============================================================
def generate_demo_label(text: str = None) -> str:
    """
    Gán nhãn giả (synonym) ngẫu nhiên cho các ví dụ minh họa.
    CHỈ sử dụng để gán nhãn giả cho demo, KHÔNG dùng trong prompt LLM.
    
     
    1. Lấy danh sách tổng hợp tất cả các nhãn đồng nghĩa (ALL_SYNONYM_LABELS).
    2. Sử dụng random.choice để chọn ngẫu nhiên một nhãn.
    3. Trả về nhãn được chọn dưới dạng chuỗi (str).
    """
    if random.random() < 0.5:
        return random.choice(REAL_SYNONYM_LABELS)
    else:
        return random.choice(FAKE_SYNONYM_LABELS)

def to_clean_demo_label(binary_label: int) -> str:
    """
    Chuyển đổi nhãn nhị phân sang chuỗi nhãn trực tiếp ("Real"/"Fake").
    Sử dụng cho các ví dụ minh họa từ Round 2 trở đi (từ pool sạch D_clean).
    
     
    1. Chuyển đổi binary_label về kiểu số nguyên (int).
    2. Nếu label = 0: Trả về "Real" (LLM_LABEL_REAL).
    3. Nếu label = 1: Trả về "Fake" (LLM_LABEL_FAKE).
    """
    if int(binary_label) == 0:
        return LLM_LABEL_REAL
    else:
        return LLM_LABEL_FAKE


# ============================================================
# LLM Output Parsing (chỉ parse "Real" / "Fake")
# ============================================================
def _normalize_label_text(s: str) -> str:
    """
    Chuẩn hóa văn bản nhãn để so sánh.
    
     
    1. Chuyển văn bản thành chữ thường (lower).
    2. Loại bỏ khoảng trắng đầu/cuối (strip).
    3. Thay thế các khoảng trắng liên tiếp bằng một khoảng trắng duy nhất bằng regex.
    4. Trả về chuỗi văn bản đã chuẩn hóa.
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def parse_llm_label(
    llm_response: str,
    default_fake: int = 1,
    return_matched_label: bool = False,
):
    """
    Phân tích phản hồi từ LLM thành nhãn nhị phân (0/1).
    LLM chỉ trả về "Real" hoặc "Fake" -> parse đơn giản.
    
     
    1. Chuẩn hóa chuỗi phản hồi bằng `_normalize_label_text`.
    2. Loại bỏ các ký tự bọc (fencing) markdown như code blocks (```json, ```).
    3. Kiểm tra khớp trực tiếp (Direct match):
       - Nếu có 'real' và không có 'fake' -> Trả về 0 (Real).
       - Nếu có 'fake' và không có 'real' -> Trả về 1 (Fake).
    4. Nếu khớp trực tiếp thất bại, thử khớp với token đầu tiên (First token match).
    5. Nếu vẫn không khớp, trả về giá trị mặc định (default_fake).
    6. Tùy theo `return_matched_label`, trả về số nguyên hoặc tuple kèm chuỗi nhãn.
    """
    text = _normalize_label_text(llm_response)
    text = text.replace("```json", "").replace("```", "").strip()

    # Direct match
    if "real" in text and "fake" not in text:
        return (0, "Real") if return_matched_label else 0
    if "fake" in text and "real" not in text:
        return (1, "Fake") if return_matched_label else 1

    # First token match
    first_token = re.split(r"\s+|[,:;.!?]", text)[0].strip()
    if first_token == "real":
        return (0, "Real") if return_matched_label else 0
    if first_token == "fake":
        return (1, "Fake") if return_matched_label else 1

    # Default to fake
    return (default_fake, None) if return_matched_label else default_fake
