"""
Evidence bundle construction and dual-model inference.
Handles round-aware demonstration retrieval and LLM+SLM assessment.
"""

import logging
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from src.config import (
    TOP_K_DEMOS,
    FACT_TOP_K,
    LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION,
    KNOWLEDGE_MODE,
)
from src.utils import preprocess_text
from src.labels import parse_llm_label, to_clean_demo_label
from src.prompts import build_classification_prompt
from src.retrieval.demo_retrieval import search_news, retrieve_demonstrations
from src.retrieval.knowledge_agent import (
    build_knowledge_bundle,
    get_cached_knowledge_bundle_local,
)

# Configure logger
logger = logging.getLogger(__name__)


def _tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize văn bản cho BM25: lowercase, tách từ, loại bỏ stopwords đơn giản.
    Có thể mở rộng dùng nltk nếu cần.
    """
    # Tách từ đơn giản
    tokens = text.lower().split()
    # Danh sách stopwords tiếng Anh cơ bản (có thể mở rộng)
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with"
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def retrieve_from_clean_pool(
    query: str,
    clean_pool: List[Dict[str, Any]],
    k: int = TOP_K_DEMOS
) -> List[Dict[str, Any]]:
    """
    Truy xuất các ví dụ minh họa gần nhất từ tập D_clean (pool sạch) sử dụng BM25.
    Các nhãn được gán trực tiếp ("Real" hoặc "Fake") theo yêu cầu tuyệt đối từ vòng 2.
    """
    if not clean_pool:
        logger.debug("clean_pool is empty, returning empty demos")
        return []

    # Kiểm tra các item có đủ trường 'text' không
    valid_items = []
    for item in clean_pool:
        if "text" not in item:
            logger.warning(f"Skipping clean_pool item without 'text': {item}")
            continue
        if "label" not in item and "label_slm" not in item:
            logger.warning(f"Skipping clean_pool item without label: {item}")
            continue
        valid_items.append(item)

    if not valid_items:
        logger.warning("No valid items in clean_pool after filtering")
        return []

    cleaned_query = preprocess_text(query)
    query_tokens = _tokenize_for_bm25(cleaned_query)

    # Tokenize corpus
    corpus_items = [preprocess_text(item["text"]) for item in valid_items]
    tokenized_corpus = [_tokenize_for_bm25(doc) for doc in corpus_items]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k = [idx for idx, _ in scored_indices[:k] if scores[idx] > 0]

    if not top_k:
        logger.debug(f"No BM25 positive scores for query, fallback to first {k} items")
        top_k = list(range(min(k, len(valid_items))))

    demos = []
    for idx in top_k:
        item = valid_items[idx]
        # Lấy label ưu tiên 'label' rồi 'label_slm', mặc định 1 (fake)
        clean_label = item.get("label", item.get("label_slm", 1))
        demos.append({
            "text": preprocess_text(item["text"]),
            "label": to_clean_demo_label(clean_label),
            "source": "D_clean",
        })

    logger.debug(f"Retrieved {len(demos)} demos from clean_pool")
    return demos


def prefetch_query_context(
    text: str,
    demo_k: int = TOP_K_DEMOS,
    fact_top_k: int = FACT_TOP_K,
    reuse_knowledge_cache: bool = True,
    knowledge_cache_local: dict = None,
    knowledge_mode: str = None,
    wiki_fetch_full: bool = False,
) -> dict:
    """
    Khởi tạo ngữ cảnh truy xuất (bootstrap context) trước khi chạy vòng lặp MRCD.
    """
    cleaned_text = preprocess_text(text)
    mode = knowledge_mode or KNOWLEDGE_MODE

    if reuse_knowledge_cache:
        knowledge_bundle = get_cached_knowledge_bundle_local(
            cleaned_text,
            knowledge_cache_local,
            fact_top_k=fact_top_k,
            mode=mode,
            wiki_fetch_full=wiki_fetch_full,
        )
    else:
        knowledge_bundle = build_knowledge_bundle(
            cleaned_text,
            fact_top_k=fact_top_k,
            mode=mode,
            wiki_fetch_full=wiki_fetch_full,
        )

    bing_seed_news = search_news(cleaned_text, max_results=demo_k)

    return {
        "text": cleaned_text,
        "knowledge_bundle": knowledge_bundle,
        "knowledge_text": knowledge_bundle.get("combined_text", "No info."),
        "knowledge_mode": knowledge_bundle.get("mode", mode),
        "bing_seed_news": bing_seed_news,
    }


def build_evidence_bundle(
    text: str,
    static_corpus: List[Dict[str, str]],
    clean_pool: List[Dict[str, Any]],
    round_id: int,
    query_context: Dict[str, Any],
    demo_k: int = TOP_K_DEMOS,
) -> tuple:
    """
    Xây dựng gói bằng chứng (evidence bundle) có sự phân hoá theo vòng (round-aware).
    Trả về: (demos, knowledge_text, retrieval_source)
    """
    cleaned_text = preprocess_text(text)
    knowledge_k = query_context.get("knowledge_text", "No info.")
    bing_seed_news = query_context.get("bing_seed_news", [])

    if round_id == 1:
        combined_corpus = static_corpus + bing_seed_news
        demos = retrieve_demonstrations(cleaned_text, combined_corpus, k=demo_k)
        retrieval_source = "external_prefetched"
        logger.info(f"Round {round_id}: using external demonstrations (pseudo labels)")
    else:
        demos = retrieve_from_clean_pool(cleaned_text, clean_pool, k=demo_k)
        if demos:
            retrieval_source = "d_clean"
            logger.info(f"Round {round_id}: using {len(demos)} demonstrations from D_clean")
        else:
            combined_corpus = static_corpus + bing_seed_news
            demos = retrieve_demonstrations(cleaned_text, combined_corpus, k=demo_k)
            retrieval_source = "fallback_external_prefetched"
            logger.warning(f"Round {round_id}: clean_pool empty, falling back to external demos")

    return demos, knowledge_k, retrieval_source


def assess_with_llm(
    text: str,
    demos: List[Dict[str, Any]],
    knowledge_k: str,
    llm,
    max_retries: int = 1,
    default_label: int = 1,
) -> Dict[str, Any]:
    """
    Đánh giá tin tức bằng mô hình LLM với cơ chế retry tùy chọn.
    
    Args:
        text: Nội dung tin tức
        demos: Danh sách các demonstration
        knowledge_k: Chuỗi kiến thức bổ sung
        llm: Đối tượng LLM (có method generate_text)
        max_retries: Số lần thử lại nếu parse thất bại (mặc định 1, tức không retry)
        default_label: Nhãn mặc định nếu parse thất bại sau retry (1 = Fake)
    
    Returns:
        Dict chứa y_llm, llm_raw, llm_label_matched, prompt
    """
    cleaned_text = preprocess_text(text)
    prompt = build_classification_prompt(
        text=cleaned_text,
        knowledge_k=knowledge_k,
        demos=demos,
    )

    llm_resp = None
    final_label = default_label
    matched_label_str = None

    for attempt in range(max_retries):
        llm_resp = llm.generate_text(
            prompt,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION,
        )
        y_llm, matched = parse_llm_label(
            llm_resp,
            default_fake=default_label,
            return_matched_label=True,
        )
        if matched is not None:  # Parse thành công
            final_label = y_llm
            matched_label_str = matched
            if attempt > 0:
                logger.info(f"LLM parse succeeded on retry {attempt}")
            break
        else:
            logger.warning(f"LLM parse failed (attempt {attempt+1}/{max_retries}): {llm_resp[:100]}")
            if attempt == max_retries - 1:
                # Lần cuối vẫn thất bại, dùng default_label và matched_label_str = None
                logger.error(f"LLM parse failed after {max_retries} attempts, using default label {default_label}")

    return {
        "y_llm": final_label,
        "llm_raw": llm_resp,
        "llm_label_matched": matched_label_str,
        "prompt": prompt,
    }