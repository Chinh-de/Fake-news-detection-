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
        "You are an expert Fact-Checking Extraction Agent analyzing a social media (Twitter) post. "
        "Your task is to process the raw text and generate two outputs simultaneously for a Two-Stage Retrieval System.\n\n"
        
        "TASK 1: WIKI ENTITIES (For Knowledge Retrieval)\n"
        "Extract 1 to 4 core named entities (People, Organizations, Locations, Specific Events) from the text "
        "that are crucial for verifying the claim and are highly likely to have a Wikipedia page.\n\n"
        
        "TASK 2: NEWS SEARCH QUERY (For Exact Event Matching)\n"
        "Generate a single, concise search query specifically designed to retrieve mainstream news articles "
        "that directly report on the exact event or claim described in the post.\n"
        "STRICT RULES:\n"
        "- Capture the specific action, context, or incident (e.g., 'arrested', 'bankrupt', 'protest'), not just the entity names.\n"
        "- Translate informal social media language, slang, or abbreviations into formal journalistic keywords.\n"
        "- REMOVE all hashtags, emojis, clickbait, sensational, or emotional words (e.g., 'breaking', 'shocking', 'omg').\n"
        "- DO NOT use quotation marks (\"\") or any search operators.\n"
        "- The query should read like a factual, objective news headline summary.\n\n"
        
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object. Do NOT wrap in markdown tags (like ```json), no preamble, no explanations.\n"
        'Schema: {"entities": ["entity_1", "entity_2"], "query": "exact event news search query"}\n\n'
        
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
    """
    Xây dựng prompt phân loại chung. Nội dung tri thức đã được xử lý bên ngoài.
    """
    # 1. HEADER
    header = f"""You are an advanced AI fake news detector.

BACKGROUND KNOWLEDGE:
{knowledge_k}

INSTRUCTIONS:
Classify the following news article as either Real or Fake.
- "Real" means the article is factually accurate and trustworthy.
- "Fake" means the article is fabricated, misleading, or unverified.

STRICT RULE: Only output ONE word: "Real" or "Fake". 
No preamble, no explanation, no punctuation.
Just the word.

EXAMPLES:"""

    # 2. FEW-SHOT DEMOS
    examples = _build_demo_section(demos)

    # 3. TAIL & TARGET
    tail = f"""
----------------------------------------
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
        label_str = demo.get("label", "Unknown")
        text_demo = demo.get("text", "")[:1000].strip()
        examples += f'\n[Example {i}]\nText: "{text_demo}..."\nLabel: {label_str}\n'
    return examples
