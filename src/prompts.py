"""
Prompt builders for MRCD Framework.

Two classification prompt variants:
- wiki_only: chỉ dùng K_wiki (entity definitions)
- full: dùng K_wiki + K_fact (verified reports)

LLM chỉ trả về "Real" hoặc "Fake" (không dùng synonym list).
"""


def build_dual_extraction_prompt(text: str) -> str:
    """
    Xây dựng prompt để thực hiện đồng thời việc trích xuất thực thể (entities) và tạo truy vấn tìm kiếm trung lập (neutral query).
    """
    prompt = (
        "You are an expert Fact-Checking Extraction Agent. Your task is to process a raw news text "
        "and generate two outputs simultaneously for a Two-Stage Retrieval System.\n\n"
        "TASK 1: WIKI ENTITIES (For Knowledge Retrieval)\n"
        "Extract 1 to 4 core named entities (People, Organizations, Locations, Events) from the text "
        "that are crucial for verifying the claim and are highly likely to have a Wikipedia page.\n\n"
        "TASK 2: NEUTRAL QUERY (For Fact-Check Search)\n"
        "Generate a single, concise search query to retrieve factual articles. "
        "STRICT RULES: Focus ONLY on factual core subjects. REMOVE all clickbait, sensational, "
        "or emotional words (e.g., 'breaking', 'urgent', 'cure', 'secret'). "
        "DO NOT use quotation marks (\"\") or any search operators.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object. Do NOT wrap in markdown tags (like ```json), no preamble, no explanations.\n"
        'Schema: {"entities": ["entity_1", "entity_2"], "query": "query"}\n\n'
        f"Input text: {text}"
    )
    return prompt

def build_entity_extraction_prompt(text: str) -> str:
    """
    Xây dựng prompt CHỈ trích xuất thực thể (phục vụ chế độ wiki_only giúp tiết kiệm token).
    """
    prompt = (
        "You are an expert Fact-Checking Extraction Agent.\n\n"
        "TASK: WIKI ENTITIES (For Knowledge Retrieval)\n"
        "Extract 1 to 4 core named entities (People, Organizations, Locations, Events) from the text "
        "that are crucial for verifying the claim and are highly likely to have a Wikipedia page.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object. Do NOT wrap in markdown tags (like ```json), no preamble, no explanations.\n"
        'Schema: {"entities": ["entity_1", "entity_2"]}\n\n'
        f"Input text: {text}"
    )
    return prompt


def build_classification_prompt(
    text: str,
    knowledge_k: str,
    demos: list,
) -> str:
    # Giới hạn knowledge để tránh quá dài
    max_knowledge_len = 2000
    if len(knowledge_k) > max_knowledge_len:
        knowledge_k = knowledge_k[:max_knowledge_len] + "... (truncated)"

    header = f"""You are an advanced AI fake news detector.

BACKGROUND KNOWLEDGE (use this as the source of truth):
{knowledge_k}

INSTRUCTIONS:
You will classify a news article as either Real or Fake.
- **Real** means the article is factually accurate and consistent with the background knowledge.
- **Fake** means the article is fabricated, misleading, unverified, or contradicts the background knowledge.

IMPORTANT: In the examples below, labels like "True", "Authentic", "Genuine", "Factual" correspond to **Real**.
Labels like "Fake", "False", "Hoax", "Misleading", "Unsubstantiated" correspond to **Fake**.

STRICT RULE: Output ONLY one word: "Real" or "Fake". No preamble, no punctuation. Just the word.

EXAMPLES:"""

    examples = _build_demo_section(demos)  # cần sửa _build_demo_section bỏ dấu ...

    tail = f"""
TARGET ARTICLE TO CLASSIFY:
Text: "{text.strip()}"

Label:"""
    return header + examples + tail


def _build_demo_section(demos: list) -> str:
    """
    Xây dựng phần danh sách các ví dụ few-shot cho prompt.
    
     
    1. Kiểm tra nếu danh sách demos trống, trả về thông báo rỗng.
    2. Lặp qua danh sách demos (bắt đầu từ index 1).
    3. Với mỗi demo:
       - Lấy nhãn (label) và nội dung văn bản (giới hạn 1000 ký tự đầu).
       - Định dạng theo cấu trúc: [Example n], Text: "...", Label: ...
    4. Ghép tất cả các ví dụ thành một chuỗi duy nhất và trả về.
    """
    if not demos:
        return "\n(No examples provided)\n"

    examples = ""
    for i, demo in enumerate(demos, start=1):
        raw_label = demo.get("label", "Unknown")
        # Chuyển đổi từ đồng nghĩa thành "Real" hoặc "Fake"
        if raw_label.lower() in ["true", "authentic", "genuine", "factual", "real"]:
            display_label = "Real"
        elif raw_label.lower() in ["fake", "false", "hoax", "misleading", "unsubstantiated"]:
            display_label = "Fake"
        else:
            display_label = raw_label  # fallback
        text_demo = demo.get("text", "")[:1000].strip()
        examples += f'\n[Example {i}]\nText: "{text_demo}..."\nLabel: {display_label}\n'
    return examples
    # for i, demo in enumerate(demos, start=1):
    #     label_str = demo.get("label", "Unknown")
    #     text_demo = demo.get("text", "")[:1000].strip()
    #     examples += f'\n[Example {i}]\nText: "{text_demo}..."\nLabel: {label_str}\n'
    # return examples
