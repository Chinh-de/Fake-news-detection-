"""
SLM fine-tuning wrapper for MRCD pipeline.
Conditional fine-tuning on D_clean after each round.
"""

from src.config import (
    ENABLE_SLM_FINETUNE,
    SLM_FINETUNE_EPOCHS,
    SLM_FINETUNE_BATCH_SIZE,
    SLM_FINETUNE_LR,
    SLM_FINETUNE_WEIGHT_DECAY,
    SLM_FINETUNE_MIN_SAMPLES,
)


def maybe_finetune_slm_on_clean(
    slm,
    clean_pool: list,
    round_id: int,
    enable_slm_finetune: bool = ENABLE_SLM_FINETUNE,
    slm_finetune_epochs: int = SLM_FINETUNE_EPOCHS,
    slm_finetune_batch_size: int = SLM_FINETUNE_BATCH_SIZE,
    slm_finetune_lr: float = SLM_FINETUNE_LR,
    slm_finetune_weight_decay: float = SLM_FINETUNE_WEIGHT_DECAY,
    slm_finetune_min_samples: int = SLM_FINETUNE_MIN_SAMPLES,
) -> dict:
    """
    Conditionally fine-tune SLM on D_clean.
    
    Skips if:
    - Fine-tuning is disabled
    - Not enough clean samples (< min_samples)
    
    Args:
        slm: IntegratedSLM instance to fine-tune
        clean_pool: List of clean sample dicts
        round_id: Current round number (for logging)
        
    Returns:
        dict with training statistics
    """
    if not enable_slm_finetune:
        return {"trained": False, "reason": "disabled"}
    if len(clean_pool) < slm_finetune_min_samples:
        return {
            "trained": False,
            "reason": "insufficient_samples",
            "available_samples": len(clean_pool),
            "min_samples": slm_finetune_min_samples,
        }

    print(
        f"Fine-tuning SLM on D_clean after round {round_id} "
        f"with {len(clean_pool)} samples..."
    )
    stats = slm.finetune_on_clean(
        clean_samples=clean_pool,
        epochs=slm_finetune_epochs,
        batch_size=slm_finetune_batch_size,
        lr=slm_finetune_lr,
        weight_decay=slm_finetune_weight_decay,
    )

    if stats.get("trained", False):
        print(
            f"SLM fine-tune done | round={round_id} samples={stats['samples']} "
            f"epochs={stats['epochs']} avg_loss={stats['avg_loss']:.4f}"
        )
    else:
        print(f"Skip SLM fine-tune at round {round_id}: {stats}")

    return stats
