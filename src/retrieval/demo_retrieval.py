# """
# Branch 1: Demonstration Retrieval
# Bing News search + static AG News corpus + BM25 top-k selection.
# """

# import io

# import requests
# import pandas as pd
# from ddgs import DDGS
# from rank_bm25 import BM25Okapi

# from src.config import AG_NEWS_URL
# from src.utils import clean_query, truncate_text, log_retrieval_to_csv
# from src.labels import generate_demo_label


# def load_news_corpus(url: str = AG_NEWS_URL) -> list:
#     """
#     Tải tập dữ liệu AG News và nạp vào danh sách các tài liệu văn bản.
    
     
#     1. Gửi yêu cầu GET để tải CSV từ URL.
#     2. Chuyển nội dung phản hồi thành luồng dữ liệu (StringIO).
#     3. Sử dụng pandas để đọc CSV với các cột: class, title, desc.
#     4. Gộp 'title' và 'desc' thành một chuỗi duy nhất cho mỗi dòng.
#     5. Trả về danh sách các văn bản đã gộp.
#     """
#     print(f" Downloading News Corpus from {url}...")
#     try:
#         response = requests.get(url)
#         response.raise_for_status()

#         csv_content = io.StringIO(response.text)
#         df = pd.read_csv(csv_content, header=None, names=["class", "title", "desc"])
#         corpus_texts = (df["title"] + " " + df["desc"]).tolist()

#         print(f" Loaded {len(corpus_texts)} documents from AG News.")
#         return corpus_texts
#     except Exception as e:
#         print(f" Error downloading corpus: {e}")
#         return []


# def search_news(query: str, max_results: int = 10) -> list:
#     """
#     Tìm kiếm các đoạn tin tức mới nhất qua DuckDuckGo (backend Bing).
    
     
#     1. Làm sạch (clean) và cắt ngắn (truncate) truy vấn đầu vào.
#     2. Khởi tạo DuckDuckGo Search với timeout.
#     3. Gọi API tìm kiếm news với các tham số: region, safesearch, backend="bing".
#     4. Lặp qua các kết quả để lấy 'title' và 'body'.
#     5. Ghi nhật ký (log) thông tin tìm kiếm vào file CSV.
#     6. Trả về danh sách các đoạn văn bản tin tức.
#     """
#     query = clean_query(query)
#     query = truncate_text(query, max_length=50)

#     news_items = []
#     try:
#         with DDGS(timeout=20) as ddgs:
#             results_gen = ddgs.news(
#                 query=query,
#                 region="wt-wt",
#                 safesearch="off",
#                 timelimit=None,
#                 max_results=max_results,
#                 backend="bing",
#             )

#             for i, result in enumerate(results_gen):
#                 if i >= max_results:
#                     break

#                 title = result.get("title", "")
#                 body = result.get("body", "")
#                 news_items.append(f"{title}\n{body}")
#                 url = result.get("url", result.get("href", ""))
#                 log_retrieval_to_csv("search_news", query, title, url, body)
#     except Exception:
#         pass
#     return news_items


# def retrieve_demonstrations(query: str, corpus_items: list, k: int = 4) -> list:
#     """
#     Sử dụng thuật toán BM25 để lấy ra top-k ví dụ minh họa (demonstrations).
    
     
#     1. Kiểm tra nếu corpus trống thì trả về danh sách rỗng.
#     2. Tokenize (chia từ) và chuyển corpus sang chữ thường.
#     3. Khởi tạo mô hình xếp hạng BM25Okapi với corpus đã tokenize.
#     4. Tính toán điểm số BM25 cho truy vấn (đã tokenize).
#     5. Sắp xếp các tài liệu theo điểm số giảm dần và lấy top-k chỉ số (indices).
#     6. Với mỗi chỉ số trong top-k:
#        - Lấy nội dung văn bản.
#        - Tạo nhãn giả (pseudo-label) ngẫu nhiên bằng `generate_demo_label`.
#        - Đóng gói thành object kèm thông tin nguồn.
#     7. Trả về danh sách các demonstrations.
#     """
#     if not corpus_items:
#         return []

#     tokenized_corpus = [doc.lower().split() for doc in corpus_items]
#     bm25 = BM25Okapi(tokenized_corpus)
#     scores = bm25.get_scores(query.lower().split())

#     scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
#     top_k = [idx for idx, _ in scored_indices[:k]]

#     demonstrations = []
#     for i in top_k:
#         content = corpus_items[i]
#         demonstrations.append(
#             {
#                 "text": content,
#                 "label": generate_demo_label(content),
#                 "source": "Bing/Retrieved",
#             }
#         )
#     return demonstrations


"""
Branch 1: Demonstration Retrieval
Bing News search + static AG News corpus + BM25 top-k selection.
"""

import io
import os
import pickle
import hashlib
import logging
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from ddgs import DDGS
from rank_bm25 import BM25Okapi

from src.config import AG_NEWS_URL
from src.utils import clean_query, truncate_text, log_retrieval_to_csv
from src.labels import generate_demo_label

# Configure logging
logger = logging.getLogger(__name__)

# Try to import NLTK for better tokenization (optional)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except (LookupError, ImportError):
    nltk = None
    logger.warning("NLTK not fully configured. Fallback to simple split tokenization.")

# Cache file for AG News corpus
AG_NEWS_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "ag_news_corpus.pkl")
os.makedirs(os.path.dirname(AG_NEWS_CACHE), exist_ok=True)


def _tokenize(text: str) -> List[str]:
    """
    Tokenize văn bản: lowercasing, loại bỏ stopwords (nếu có), chỉ giữ token dài >=2.
    """
    text = text.lower()
    if nltk:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 1]
    else:
        tokens = [t for t in text.split() if len(t) > 1]
    return tokens


def load_news_corpus(url: str = AG_NEWS_URL, use_cache: bool = True) -> List[str]:
    """
    Tải tập dữ liệu AG News, có cache để tránh tải lại nhiều lần.
    """
    if use_cache and os.path.exists(AG_NEWS_CACHE):
        try:
            with open(AG_NEWS_CACHE, "rb") as f:
                corpus_texts = pickle.load(f)
            logger.info(f"Loaded {len(corpus_texts)} documents from cache: {AG_NEWS_CACHE}")
            return corpus_texts
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    logger.info(f"Downloading News Corpus from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content, header=None, names=["class", "title", "desc"])
        corpus_texts = (df["title"] + " " + df["desc"]).tolist()
        logger.info(f"Loaded {len(corpus_texts)} documents from AG News.")

        # Save to cache
        try:
            with open(AG_NEWS_CACHE, "wb") as f:
                pickle.dump(corpus_texts, f)
            logger.info(f"Cached corpus to {AG_NEWS_CACHE}")
        except Exception as e:
            logger.warning(f"Failed to cache corpus: {e}")

        return corpus_texts
    except Exception as e:
        logger.error(f"Error downloading corpus: {e}")
        return []


def search_news(query: str, max_results: int = 10, retries: int = 2) -> List[str]:
    """
    Tìm kiếm tin tức qua DuckDuckGo (Bing backend) với cơ chế retry.
    """
    query = clean_query(query)
    # Chỉ truncate nếu quá dài (trên 200 ký tự)
    if len(query) > 200:
        query = truncate_text(query, max_length=200)

    news_items = []
    for attempt in range(retries):
        try:
            with DDGS(timeout=20) as ddgs:
                results_gen = ddgs.news(
                    query=query,
                    region="wt-wt",
                    safesearch="off",
                    timelimit=None,
                    max_results=max_results,
                    backend="bing",
                )
                for i, result in enumerate(results_gen):
                    if i >= max_results:
                        break
                    title = result.get("title", "")
                    body = result.get("body", "")
                    full_text = f"{title}\n{body}".strip()
                    if full_text:
                        news_items.append(full_text)
                    url = result.get("url", result.get("href", ""))
                    log_retrieval_to_csv("search_news", query, title, url, body)
                break  # Thành công
        except Exception as e:
            logger.warning(f"Search attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                logger.error("All search attempts failed. Returning empty list.")
    return news_items


def retrieve_demonstrations(query: str, corpus_items: List[str], k: int = 4) -> List[Dict[str, Any]]:
    """
    Sử dụng BM25 để chọn top-k demonstrations. Hỗ trợ deduplication.
    """
    if not corpus_items:
        return []

    # Tokenize query
    query_tokens = _tokenize(query)
    if not query_tokens:
        # Fallback: nếu query không có token, trả về rỗng
        return []

    # Tokenize corpus và xây dựng BM25
    tokenized_corpus = [_tokenize(doc) for doc in corpus_items]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    # Lọc các mẫu có điểm > 0 (nếu không, lấy top k)
    valid_indices = [(idx, score) for idx, score in scored_indices if score > 0]
    if not valid_indices:
        # Fallback: lấy top k bất kỳ
        valid_indices = scored_indices[:k]
    top_k = [idx for idx, _ in valid_indices[:k]]

    # Deduplication dựa trên hash nội dung (tránh trùng lặp trong corpus_items)
    seen_hashes = set()
    demonstrations = []
    for idx in top_k:
        content = corpus_items[idx]
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        demonstrations.append({
            "text": content,
            "label": generate_demo_label(content),
            "source": "Bing/Retrieved",
        })
        if len(demonstrations) >= k:
            break

    # Log BM25 scores (debug)
    logger.debug(f"BM25 scores for query '{query[:50]}...': top scores = {[s for _, s in valid_indices[:k]]}")

    return demonstrations