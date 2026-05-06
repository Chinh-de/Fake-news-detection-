"""
Branch 2: Knowledge Retrieval (LLM Analysis + Parallel Crawl + Rerank)
- Entity extraction via LLM
- Trusted domain search
- Parallel web crawling with ThreadPoolExecutor
- Cross-encoder reranking
"""

import re
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from ddgs import DDGS
from sentence_transformers import CrossEncoder

from src.config import (
    TRUST_DOMAINS,
    LLM_MAX_OUTPUT_TOKENS_EXTRACTION,
    CRAWL_MAX_WORKERS,
)
import src.config as cfg
from src.utils import clean_query, truncate_text, log_retrieval_to_csv
from src.prompts import build_dual_extraction_prompt, build_entity_extraction_prompt
from src.llm.handler import get_llm


def analyze_claim_entities_and_query(text: str, mode: str = "full") -> dict:
    """
    Sử dụng LLM một lần duy nhất để trích xuất các thực thể và tạo truy vấn tìm kiếm.
    
     
    1. Kiểm tra mode: nếu 'wiki_only' thì dùng prompt rút gọn.
    2. Gọi LLM để sinh kết quả.
    3. Làm sạch phản hồi từ LLM, loại bỏ các ký tự thừa (như ```json).
    4. Sử dụng Regex để trích xuất đối tượng JSON từ nội dung đã làm sạch.
    5. Chuyển đổi chuỗi JSON thành dictionary Python.
    6. Chuẩn hóa danh sách các thực thể (xử lý cả kiểu chuỗi và kiểu đối tượng).
    7. Chuẩn hóa và giới hạn độ dài của truy vấn tìm kiếm.
    8. Trả về kết quả hoặc fallback nếu có lỗi xảy ra.
    """
    if mode == "wiki_only":
        prompt = build_entity_extraction_prompt(text)
        max_tokens = 48  # Giới hạn cực thấp do chỉ cần 1-4 thực thể
    else:
        prompt = build_dual_extraction_prompt(text)
        max_tokens = LLM_MAX_OUTPUT_TOKENS_EXTRACTION

    fallback = {
        "entities": [],
        "query": truncate_text(clean_query(text), max_length=80),
    }

    try:
        llm = get_llm()
        raw = llm.generate_text(
            prompt, max_output_tokens=max_tokens
        )
        clean = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)

        data = json.loads(clean)
        entities = data.get("entities", []) if isinstance(data, dict) else []
        query = data.get("query", "") if isinstance(data, dict) else ""

        normalized_entities = []
        for item in entities:
            if isinstance(item, str) and item.strip():
                normalized_entities.append(item.strip())
            elif isinstance(item, dict) and item.get("entity"):
                normalized_entities.append(str(item.get("entity", "")).strip())

        if not query:
            query = fallback["query"]
        else:
            query = truncate_text(clean_query(query), max_length=80)

        return {
            "entities": normalized_entities,
            "query": query,
        }
    except Exception:
        return fallback


def build_trusted_domain_query(domains: list = None) -> str:
    """
    Xây dựng chuỗi toán tử tìm kiếm cho các tên miền (domains) tin cậy.
    
     
    1. Lấy danh sách domains từ tham số hoặc từ cấu hình mặc định (TRUST_DOMAINS).
    2. Nếu danh sách trống, trả về chuỗi rỗng.
    3. Sử dụng toán tử `site:` của công cụ tìm kiếm kết hợp với logic `OR`.
    4. Trả về chuỗi định dạng: (site:domain1 OR site:domain2 OR ...).
    """
    domains = domains or TRUST_DOMAINS
    if not domains:
        return ""
    return "(" + " OR ".join([f"site:{d}" for d in domains]) + ")"


def scrape_full_article(url: str) -> str | None:
    """
    Cào nội dung toàn bộ bài báo và giữ lại các đoạn có tín hiệu (signal) cao.
    
     
    1. Chọn ngẫu nhiên một trình duyệt giả lập (impersonate) để tránh bị chặn.
    2. Gửi yêu cầu GET đến URL với timeout.
    3. Kiểm tra mã trạng thái phản hồi (chỉ tiếp tục nếu là 200).
    4. Sử dụng BeautifulSoup để phân tích HTML.
    5. Loại bỏ các thẻ rác không chứa nội dung chính (script, style, nav, footer, v.v.).
    6. Lấy tất cả các thẻ `<p>`.
    7. Lọc các đoạn văn bản có độ dài trên 8 từ để đảm bảo chất lượng thông tin.
    8. Làm sạch văn bản: xóa URL, xóa trích dẫn trong ngoặc vuông, chuẩn hóa khoảng trắng.
    9. Trả về nội dung bài báo hoặc None nếu thất bại.
    """
    browsers = ["chrome", "chrome110", "edge99", "safari15_3"]
    browser_choice = random.choice(browsers)

    try:
        response = curl_requests.get(url, impersonate=browser_choice, timeout=15)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(
            [
                "script", "style", "nav", "footer", "header",
                "aside", "form", "button", "iframe",
            ]
        ):
            element.decompose()

        paragraphs = soup.find_all("p")
        cleaned_paragraphs = []
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if len(text.split()) > 8:
                cleaned_paragraphs.append(text)

        raw_text = " ".join(cleaned_paragraphs)
        raw_text = re.sub(r"http[s]?://\S+", "", raw_text)
        raw_text = re.sub(r"www\.\S+", "", raw_text)
        raw_text = re.sub(r"\[.*?\]", "", raw_text)
        article_text = re.sub(r"\s+", " ", raw_text).strip()

        if not article_text:
            return None
        return article_text
    except Exception:
        return None


def _crawl_single_result(result: dict) -> dict | None:
    """
    Thực hiện cào nội dung cho một kết quả tìm kiếm duy nhất (hàm trợ giúp cho chạy song song).
    
     
    1. Gọi hàm `scrape_full_article` để lấy nội dung từ URL.
    2. Nếu không lấy được nội dung đầy đủ hoặc nội dung quá ngắn (< 30 từ):
       - Sử dụng đoạn trích (snippet) có sẵn từ kết quả tìm kiếm làm nội dung dự phòng.
    3. Nếu vẫn không có nội dung, trả về None.
    4. Trả về dictionary chứa tiêu đề, url và nội dung đã cào.
    """
    content = scrape_full_article(result["url"])
    if not content or len(content.split()) < 30:
        content = result.get("snippet", "")
    if not content:
        return None
    return {
        "title": result["title"],
        "url": result["url"],
        "content": content,
    }


def crawl_results_parallel(
    results: list, max_workers: int = CRAWL_MAX_WORKERS
) -> list:
    """
    Thực hiện cào nhiều URL song song sử dụng ThreadPoolExecutor.
    
     
    1. Kiểm tra danh sách kết quả đầu vào.
    2. Khởi tạo ThreadPoolExecutor với số lượng luồng (workers) tối đa.
    3. Gửi các tác vụ cào từng kết quả đơn lẻ (`_crawl_single_result`) vào executor.
    4. Sử dụng `as_completed` để thu thập kết quả ngay khi mỗi luồng hoàn thành.
    5. Tổng hợp các tài liệu đã cào thành công vào một danh sách.
    6. Trả về danh sách các tài liệu (documents).
    
    Args:
        results: Danh sách các dictionary chứa 'url', 'title', 'snippet'.
        max_workers: Số luồng đồng thời tối đa.
    """
    if not results:
        return []

    documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_crawl_single_result, r): r for r in results
        }
        for future in as_completed(futures):
            try:
                doc = future.result()
                if doc is not None:
                    documents.append(doc)
            except Exception:
                pass

    return documents


def chunk_text_by_sentences(
    text: str, max_words: int = 300, overlap_sentences: int = 1
) -> list:
    """
    Chia nhỏ văn bản thành các đoạn (chunks) theo ranh giới câu.
    
     
    1. Tách văn bản thành danh sách các câu dựa trên các dấu câu kết thúc (.!?).
    2. Lặp qua từng câu và tính toán số từ.
    3. Nếu câu hiện tại cộng với đoạn hiện tại vượt quá `max_words`:
       - Lưu đoạn hiện tại vào danh sách chunks.
       - Tạo đoạn mới với cơ chế gối đầu (overlap) bằng số lượng câu quy định.
    4. Nếu một câu đơn lẻ dài hơn `max_words`, coi nó là một chunk riêng biệt.
    5. Sau khi lặp hết, nếu còn đoạn văn bản chưa lưu thì thêm vào danh sách chunks.
    6. Trả về danh sách các chunks.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = len(sentence.split())

        if sentence_words > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            i += 1
            continue

        if current_word_count + sentence_words <= max_words:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            i += 1
        else:
            chunks.append(" ".join(current_chunk))

            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
                current_word_count = sum(len(s.split()) for s in current_chunk)

                while current_chunk and (
                    current_word_count + sentence_words > max_words
                ):
                    removed_sentence = current_chunk.pop(0)
                    current_word_count -= len(removed_sentence.split())
            else:
                current_chunk = []
                current_word_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ============================================================
# Cross-Encoder Singleton
# ============================================================
_fact_ranker = None


def get_fact_ranker() -> CrossEncoder:
    """
    Lấy hoặc tạo mới một singleton CrossEncoder dùng để xếp hạng lại (rerank).
    
     
    1. Kiểm tra biến toàn cục `_fact_ranker`.
    2. Nếu chưa tồn tại, khởi tạo mô hình CrossEncoder từ HuggingFace (mặc định dùng ms-marco-MiniLM).
    3. Trả về instance của mô hình.
    """
    global _fact_ranker
    if _fact_ranker is None:
        _fact_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _fact_ranker


def retrieve_fact_evidence(
    raw_query: str,
    max_urls: int = 12,
    top_k_chunks: int = 3,
    similarity_threshold: float = None,
    crawl_max_workers: int = CRAWL_MAX_WORKERS,
) -> dict:
    """
    Thực hiện quy trình truy xuất bằng chứng: Tìm kiếm từ domains tin cậy + Cào song song + Xếp hạng lại.
    
     
    1. Phân tích yêu cầu (analyzing) để lấy các thực thể và truy vấn tìm kiếm tối ưu.
    2. Xây dựng chuỗi truy vấn nâng cao kết hợp với các toán tử `site:`.
    3. Sử dụng DuckDuckGo (backend Bing) để lấy danh sách URL kết quả.
    4. Thực hiện cào nội dung các trang web song song (Parallel Crawl).
    5. Với mỗi tài liệu đã cào:
       - Chia nhỏ văn bản thành các chunks.
       - Sử dụng CrossEncoder để tính điểm tương đồng giữa yêu cầu gốc và từng chunk.
       - Lọc các chunk có điểm số trên ngưỡng `similarity_threshold`.
    6. Sắp xếp toàn bộ các chunk theo điểm số giảm dần.
    7. Lấy top-k chunk có điểm cao nhất.
    8. Trả về dictionary chứa thông tin phân tích và các chunks hàng đầu.
    """
    if similarity_threshold is None:
        similarity_threshold = cfg.RERANKER_THRESHOLD
    analysis = analyze_claim_entities_and_query(raw_query)
    query = analysis.get(
        "query", truncate_text(clean_query(raw_query), max_length=80)
    )

    trusted_domains = build_trusted_domain_query(TRUST_DOMAINS)
    search_query = f"{query} {trusted_domains}".strip()

    results = []
    try:
        with DDGS(timeout=20) as ddgs:
            results_gen = ddgs.text(
                search_query, backend="bing", max_results=max_urls
            )
            for i, r in enumerate(results_gen):
                if i >= max_urls:
                    break
                title = str(r.get("title", "")).strip()
                url = str(r.get("href", "")).strip()
                snippet = str(r.get("body", "")).strip()
                results.append({"title": title, "url": url, "snippet": snippet})
                log_retrieval_to_csv(
                    "retrieve_fact_evidence", search_query, title, url, snippet
                )
    except Exception:
        results = []

    # === PARALLEL CRAWL ===
    documents = crawl_results_parallel(results, max_workers=crawl_max_workers)

    if not documents:
        return {
            "analysis": analysis,
            "top_chunks": [],
        }

    ranker = get_fact_ranker()
    scored_chunks = []

    for doc in documents:
        chunks = chunk_text_by_sentences(
            doc["content"], max_words=300, overlap_sentences=1
        )
        if not chunks:
            continue

        pairs = [[raw_query, chunk] for chunk in chunks]
        scores = ranker.predict(pairs)

        for idx, score in enumerate(scores):
            if float(score) >= similarity_threshold:
                scored_chunks.append(
                    {
                        "score": float(score),
                        "chunk_text": chunks[idx],
                        "title": doc["title"],
                        "url": doc["url"],
                        "source": "trusted_crawled",
                    }
                )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:top_k_chunks]

    return {
        "analysis": analysis,
        "top_chunks": top_chunks,
    }


def format_fact_knowledge(top_chunks: list) -> str:
    """
    Định dạng kiến thức từ các chunk bằng chứng để hiển thị hoặc debug.
    
     
    1. Kiểm tra nếu danh sách trống, trả về thông báo lỗi.
    2. Lặp qua danh sách top_chunks, gán số thứ tự [Fi].
    3. Với mỗi chunk, định dạng chuỗi chứa: điểm số, tiêu đề, nội dung chunk và URL.
    4. Ghép các dòng lại và trả về chuỗi văn bản.
    """
    if not top_chunks:
        return "No trusted fact evidence found."

    lines = []
    for i, item in enumerate(top_chunks, start=1):
        lines.append(
            f"[F{i}] score={item['score']:.4f} | {item['title']} | "
            f"{item['chunk_text']} | {item['url']}"
        )
    return "\n".join(lines)
