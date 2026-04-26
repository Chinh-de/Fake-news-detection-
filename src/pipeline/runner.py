"""
MRCD Pipeline Runner - Main orchestrator.
Implements the multi-round collaborative detection flow:

1. Bootstrap: Prefetch knowledge + Bing seed for all events
2. Round 1: External retrieval + LLM/SLM assessment + clean/noisy split
3. Rounds 2-N: D_clean retrieval + SLM fine-tune + re-assessment
4. Final Judgment: SLM force-labeling for remaining noisy samples
"""

from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from src.config import (
    NUM_LOOP,
    CONFIDENCE_THRESHOLD,
    TOP_K_DEMOS,
    FACT_TOP_K,
    KNOWLEDGE_MODE,
    BOOTSTRAP_ENABLE_PARALLEL,
    BOOTSTRAP_MAX_WORKERS,
    ENABLE_SLM_FINETUNE,
    SLM_FINETUNE_EPOCHS,
    SLM_FINETUNE_BATCH_SIZE,
    SLM_FINETUNE_LR,
    SLM_FINETUNE_WEIGHT_DECAY,
    SLM_FINETUNE_MIN_SAMPLES,
    WIKI_FETCH_FULL,
)
from src.utils import (
    preprocess_text,
    log_prediction_to_csv,
    log_round_trace_to_csv,
)
from src.llm.handler import get_llm
from src.retrieval.demo_retrieval import load_news_corpus
from src.pipeline.evidence import (
    prefetch_query_context,
    build_evidence_bundle,
    assess_with_llm,
)
from src.pipeline.selection import split_clean_noisy, finalize_remaining_noisy_with_slm
from src.pipeline.finetune import maybe_finetune_slm_on_clean


def run_mrcd_pipeline(
    events: List[str],
    slm,
    max_rounds: int = NUM_LOOP,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    knowledge_mode: str = None,
    reuse_knowledge_cache: bool = True,
    bootstrap_parallel: bool = BOOTSTRAP_ENABLE_PARALLEL,
    bootstrap_max_workers: int = BOOTSTRAP_MAX_WORKERS,
    enable_slm_finetune: bool = ENABLE_SLM_FINETUNE,
    slm_finetune_epochs: int = SLM_FINETUNE_EPOCHS,
    slm_finetune_batch_size: int = SLM_FINETUNE_BATCH_SIZE,
    slm_finetune_lr: float = SLM_FINETUNE_LR,
    slm_finetune_weight_decay: float = SLM_FINETUNE_WEIGHT_DECAY,
    slm_finetune_min_samples: int = SLM_FINETUNE_MIN_SAMPLES,
    ground_truth: List[int] = None,
):
    """
    Run the full MRCD multi-round pipeline.
    
    Args:
        events: List of news texts to classify
        slm: IntegratedSLM instance (injected dependency)
        max_rounds: Maximum number of rounds (default: 3)
        confidence_threshold: SLM confidence threshold for clean selection
        knowledge_mode: "wiki_only" or "full" (default from config)
        reuse_knowledge_cache: Whether to cache knowledge bundles
        bootstrap_parallel: Enable parallel bootstrap context fetching
        enable_slm_finetune: Whether to fine-tune SLM each round
        
    Returns:
        dict with keys: results, clean, noisy, finalized_noisy,
                       history, finetune_history, knowledge_cache_size
    """
    mode = knowledge_mode or KNOWLEDGE_MODE

    print(f"Starting MRCD pipeline for {len(events)} events...")
    print(f"Knowledge mode: {mode}")
    llm = get_llm()

    print("Loading news corpus (fact-checking base)...")
    static_corpus = load_news_corpus()
    print(f"Corpus loaded: {len(static_corpus)} documents")

    cleaned_events = [preprocess_text(e) for e in events]

    event_states = [
        {
            "event_id": idx,
            "text": text,
            "round": 0,
            "status": "unprocessed",
            "label": None,
            "label_llm": None,
            "label_slm": None,
            "conf_slm": None,
            "llm_raw": None,
            "llm_label_matched": None,
            "retrieval_source": None,
            "knowledge": None,
            "query_context": None,
            "ground_truth": ground_truth[idx] if ground_truth is not None else None,
        }
        for idx, text in enumerate(cleaned_events)
    ]

    d_clean = []
    d_noisy = []
    round_history = []
    finetune_history = []
    knowledge_cache_local = {}

    # ================================================================
    # Bootstrap: Prefetch knowledge + Bing seed
    # ================================================================
    print("\n=== Bootstrap Retrieval Context ===")
    unique_texts = list(dict.fromkeys([s["text"] for s in event_states]))
    context_map = {}

    if bootstrap_parallel and len(unique_texts) > 1:
        workers = max(1, int(bootstrap_max_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_text = {
                executor.submit(
                    prefetch_query_context,
                    text,
                    TOP_K_DEMOS,
                    FACT_TOP_K,
                    reuse_knowledge_cache,
                    knowledge_cache_local,
                    mode,
                    WIKI_FETCH_FULL,
                ): text
                for text in unique_texts
            }

            for future in tqdm(
                as_completed(future_to_text),
                total=len(future_to_text),
                desc="Bootstrap Context",
            ):
                text = future_to_text[future]
                try:
                    context_map[text] = future.result()
                except Exception:
                    context_map[text] = {
                        "text": text,
                        "knowledge_bundle": {"combined_text": "No info."},
                        "knowledge_text": "No info.",
                        "knowledge_mode": mode,
                        "bing_seed_news": [],
                    }
    else:
        for text in tqdm(unique_texts, desc="Bootstrap Context"):
            context_map[text] = prefetch_query_context(
                text=text,
                demo_k=TOP_K_DEMOS,
                fact_top_k=FACT_TOP_K,
                reuse_knowledge_cache=reuse_knowledge_cache,
                knowledge_cache_local=knowledge_cache_local,
                knowledge_mode=mode,
                wiki_fetch_full=WIKI_FETCH_FULL,
            )

    for state in event_states:
        qctx = context_map.get(state["text"])
        if qctx is None:
            qctx = {
                "text": state["text"],
                "knowledge_bundle": {"combined_text": "No info."},
                "knowledge_text": "No info.",
                "knowledge_mode": mode,
                "bing_seed_news": [],
            }
        state["query_context"] = qctx
        state["knowledge"] = qctx.get("knowledge_text", "No info.")

    # ================================================================
    # Round 1: External Retrieval + Assessment + Selection
    # ================================================================
    round_id = 1
    print("\n=== Round 1: Retrieval + Assessment + Selection ===")
    
    # 1. LLM Assessment
    for state in tqdm(event_states, desc="Round 1 - LLM Processing"):
        text = preprocess_text(state["text"])
        demos, knowledge_k, retrieval_source = build_evidence_bundle(
            text=text,
            static_corpus=static_corpus,
            clean_pool=d_clean,
            round_id=round_id,
            query_context=state["query_context"],
            demo_k=TOP_K_DEMOS,
        )
        assess = assess_with_llm(
            text=text, demos=demos, knowledge_k=knowledge_k,
            llm=llm,
        )

        state.update(
            {
                "round": round_id,
                "label": assess["y_llm"],
                "label_llm": assess["y_llm"],
                "llm_raw": assess["llm_raw"],
                "llm_label_matched": assess["llm_label_matched"],
                "retrieval_source": retrieval_source,
                "knowledge": knowledge_k,
                "prompt": assess["prompt"],
            }
        )

    # 2. SLM Batch Inference
    print(f"Round 1 - SLM Batch Inference for {len(event_states)} items")
    slm_texts = [state["text"] for state in event_states]
    slm_results = slm.inference_batch(slm_texts)

    # 3. Merge, Log, and Split
    for state, res in zip(event_states, slm_results):
        pred, conf, probs = res
        state["label_slm"] = pred
        state["y_slm"] = pred
        state["conf_slm"] = conf

        # Ghi log vết (trace) cho từng sự kiện trong vòng này
        log_round_trace_to_csv(
            round_id=round_id,
            event_id=state["event_id"],
            text=state["text"],
            y_slm=state["y_slm"],
            y_llm=state["label_llm"],
            ground_truth=state["ground_truth"],
            conf_slm=state["conf_slm"],
            prompt=state["prompt"],
        )

        if split_clean_noisy(state, confidence_threshold):
            # Use SLM label (not LLM) since FTT-SLM is more accurate (~70% vs ~50%)
            state["label"] = pred
            state["status"] = "clean"
            d_clean.append(state)
            # Ghi log kết quả ngay lập tức để tiết kiệm RAM
            log_prediction_to_csv(
                event_id=state["event_id"],
                text=state["text"],
                label=state["label"],
                conf=state["conf_slm"],
                round_id=round_id,
                status=state["status"],
            )
        else:
            state["status"] = "noisy"
            d_noisy.append(state)

    round_history.append(
        {
            "round": round_id,
            "clean_count": len(d_clean),
            "noisy_count": len(d_noisy),
        }
    )
    print(f"Round {round_id} summary -> Clean: {len(d_clean)}, Noisy: {len(d_noisy)}")

    # ================================================================
    # Rounds 2-N: D_clean Retrieval + Fine-tune + Re-assessment
    # ================================================================
    round_id = 2
    while d_noisy and round_id <= max_rounds:
        print(f"\n=== Round {round_id}: Re-Assessment + SLM Fine-tune ===")

        # Finetune ở đầu mỗi vòng lặp tiếp theo (dựa trên D_clean đã thu thập)
        ft_stats = maybe_finetune_slm_on_clean(
            slm=slm,
            clean_pool=d_clean,
            round_id=round_id,
            enable_slm_finetune=enable_slm_finetune,
            slm_finetune_epochs=slm_finetune_epochs,
            slm_finetune_batch_size=slm_finetune_batch_size,
            slm_finetune_lr=slm_finetune_lr,
            slm_finetune_weight_decay=slm_finetune_weight_decay,
            slm_finetune_min_samples=slm_finetune_min_samples,
        )
        finetune_history.append({"round": round_id, **ft_stats})

        next_noisy = []
        promoted_clean = 0

        # 1. LLM Assessment
        for state in tqdm(d_noisy, desc=f"Round {round_id} - LLM Processing"):
            text = preprocess_text(state["text"])
            demos, knowledge_k, retrieval_source = build_evidence_bundle(
                text=text,
                static_corpus=static_corpus,
                clean_pool=d_clean,
                round_id=round_id,
                query_context=state["query_context"],
                demo_k=TOP_K_DEMOS,
            )
            assess = assess_with_llm(
                text=text, demos=demos, knowledge_k=knowledge_k,
                llm=llm,
            )

            state.update(
                {
                    "round": round_id,
                    "label": assess["y_llm"],
                    "label_llm": assess["y_llm"],
                    "llm_raw": assess["llm_raw"],
                    "llm_label_matched": assess["llm_label_matched"],
                    "retrieval_source": retrieval_source,
                    "knowledge": knowledge_k,
                    "prompt": assess["prompt"],
                }
            )

        # 2. SLM Batch Inference
        print(f"Round {round_id} - SLM Batch Inference for {len(d_noisy)} items")
        slm_texts = [state["text"] for state in d_noisy]
        slm_results = slm.inference_batch(slm_texts)

        # 3. Merge, Log, and Split
        for state, res in zip(d_noisy, slm_results):
            pred, conf, probs = res
            state["label_slm"] = pred
            state["y_slm"] = pred
            state["conf_slm"] = conf

            # Ghi log vết (trace) cho từng sự kiện trong vòng này
            log_round_trace_to_csv(
                round_id=round_id,
                event_id=state["event_id"],
                text=state["text"],
                y_slm=state["y_slm"],
                y_llm=state["label_llm"],
                ground_truth=state["ground_truth"],
                conf_slm=state["conf_slm"],
                prompt=state["prompt"],
            )

            if split_clean_noisy(state, confidence_threshold):
                # Use SLM label (not LLM) since FTT-SLM is more accurate
                state["label"] = pred
                state["status"] = f"clean@round{round_id}"
                d_clean.append(state)
                promoted_clean += 1
                # Ghi log kết quả ngay lập tức
                log_prediction_to_csv(
                    event_id=state["event_id"],
                    text=state["text"],
                    label=state["label"],
                    conf=state["conf_slm"],
                    round_id=round_id,
                    status=state["status"],
                )
            else:
                state["status"] = f"noisy@round{round_id}"
                next_noisy.append(state)

        d_noisy = next_noisy
        round_history.append(
            {
                "round": round_id,
                "promoted_to_clean": promoted_clean,
                "clean_count": len(d_clean),
                "noisy_count": len(d_noisy),
            }
        )
        print(
            f"Round {round_id} summary -> +Clean: {promoted_clean}, "
            f"Total Clean: {len(d_clean)}, Remaining Noisy: {len(d_noisy)}"
        )

        round_id += 1

    # ================================================================
    # Final Judgment: SLM force-labeling remaining noisy
    # ================================================================
    finalized_noisy = []
    if d_noisy:
        print(
            f"\n=== Final Judgment: SLM force-labeling "
            f"{len(d_noisy)} unresolved noisy samples ==="
        )
        finalized_noisy = finalize_remaining_noisy_with_slm(d_noisy, slm)
        for final_sample in finalized_noisy:
            final_sample["status"] = "finalized_by_slm"
            # Ghi log vết cho bước chốt hạ cuối cùng
            log_round_trace_to_csv(
                round_id="final_judgment",
                event_id=final_sample["event_id"],
                text=final_sample["text"],
                y_slm=final_sample["label"],
                y_llm=None,
                ground_truth=final_sample["ground_truth"],
                conf_slm=final_sample["conf_slm"],
                prompt="N/A (Final SLM Judgment)",
            )
            # Ghi log kết quả cuối cùng
            log_prediction_to_csv(
                event_id=final_sample["event_id"],
                text=final_sample["text"],
                label=final_sample["label"],
                conf=final_sample["conf_slm"],
                round_id=max_rounds + 1,  # Final judgment round
                status=final_sample["status"],
            )
        round_history.append(
            {
                "round": "final_judgment",
                "force_labeled_by_slm": len(finalized_noisy),
                "clean_count": len(d_clean),
                "remaining_noisy_after_final": 0,
            }
        )

    ordered_results = sorted(event_states, key=lambda x: x["event_id"])

    return {
        "results": ordered_results,
        "clean": d_clean,
        "noisy": d_noisy,
        "finalized_noisy": finalized_noisy,
        "history": round_history,
        "finetune_history": finetune_history,
        "knowledge_cache_size": len(knowledge_cache_local),
    }
