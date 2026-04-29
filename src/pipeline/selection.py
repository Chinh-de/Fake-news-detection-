"""
Data selection logic for MRCD pipeline.
Handles clean/noisy splitting and final judgment.
"""

import logging
from typing import List, Dict, Any

from src.utils import preprocess_text

logger = logging.getLogger(__name__)


def split_clean_noisy(sample: Dict[str, Any], confidence_threshold: float) -> bool:
    """
    Selection rule: LLM-SLM agreement + SLM confidence threshold.
    
    Returns True if sample should go to D_clean, False for D_noisy.
    """
    # Lấy các giá trị với fallback an toàn (nếu thiếu key thì coi như không đạt)
    label_llm = sample.get("label_llm")
    label_slm = sample.get("label_slm")
    conf_slm = sample.get("conf_slm", 0.0)

    if label_llm is None or label_slm is None:
        logger.warning(f"Missing label in sample: LLM={label_llm}, SLM={label_slm}")
        return False

    is_clean = (label_llm == label_slm) and (conf_slm >= confidence_threshold)
    if is_clean:
        logger.debug(f"Sample {sample.get('event_id')} -> clean (conf={conf_slm:.3f})")
    else:
        logger.debug(f"Sample {sample.get('event_id')} -> noisy (conf={conf_slm:.3f}, agree={label_llm==label_slm})")
    return is_clean


def finalize_remaining_noisy_with_slm(
    d_noisy: List[Dict[str, Any]],
    slm,
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Final judgment: force SLM labels for all unresolved noisy samples.
    Uses batch inference for efficiency.
    
    Args:
        d_noisy: List of noisy sample dicts (must have 'text' key)
        slm: IntegratedSLM instance with inference_batch method
        batch_size: Batch size for SLM inference
        
    Returns:
        List of finalized sample dicts with SLM-assigned labels
    """
    if not d_noisy:
        return []

    # Extract texts
    texts = [preprocess_text(sample["text"]) for sample in d_noisy]
    
    # Batch inference
    results = slm.inference_batch(texts, batch_size=batch_size)  # returns list of (pred, conf, probs)
    
    finalized = []
    for sample, (y_slm, conf_slm, _) in zip(d_noisy, results):
        final_sample = dict(sample)
        final_sample["label"] = y_slm
        final_sample["label_final"] = y_slm
        final_sample["conf_slm_final"] = conf_slm
        final_sample["status"] = "finalized_by_slm"
        final_sample["label_source"] = "SLM_final"
        finalized.append(final_sample)
        logger.debug(f"Finalized sample {sample.get('event_id')} with SLM label {y_slm} (conf={conf_slm:.3f})")
    
    logger.info(f"Finalized {len(finalized)} noisy samples using SLM batch inference")
    return finalized