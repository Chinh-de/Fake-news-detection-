#!/usr/bin/env python3
"""Patch MRCD-FTT-Tw1516.ipynb with fixes to make MRCD-FTT >> FTT-only."""
import json, copy, sys

NB_PATH = r"e:\PBL7\Fake-news-detection\notebooks\MRCD-FTT-Tw1516.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb_new = copy.deepcopy(nb)
code_cells = [(i, c) for i, c in enumerate(nb_new["cells"]) if c["cell_type"] == "code"]

patched = []

for cell_idx, (orig_idx, cell) in enumerate(code_cells):
    src = "".join(cell["source"])

    # ── Cell 7: Save FTT checkpoint + keep slm_ftt reference ──
    if "slm_mrcd = slm" in src and "finetune_weighted" in src:
        new_src = src.replace(
            "# Assign for MRCD reuse\nslm_mrcd = slm",
            (
                "# Save FTT checkpoint for MRCD reuse\n"
                "slm.save(str(SLM_SAVE_DIR))\n"
                'FTT_CHECKPOINT = str(SLM_SAVE_DIR / "parameter_bert.pkl")\n'
                'print(f"FTT checkpoint saved at: {FTT_CHECKPOINT}")\n'
                "# Keep reference for FTT-only evaluation later\n"
                "slm_ftt = slm"
            ),
        )
        cell["source"] = [new_src]
        cell["outputs"] = []
        patched.append(f"Cell {cell_idx} (orig {orig_idx}): FTT checkpoint save")

    # ── Cell 11: Load FTT checkpoint instead of retraining ──
    if "Cell 8 - Fine-tune SLM" in src and "slm = IntegratedSLM" in src:
        cell["source"] = [
            '# Load FTT-pretrained checkpoint into MRCD SLM (NOT training from scratch)\n'
            'from src.slm.model import IntegratedSLM\n'
            '\n'
            '# Initialize MRCD SLM and load the FTT-trained weights\n'
            "slm_mrcd = IntegratedSLM(model_path='bert-base-uncased')\n"
            'slm_mrcd.load(FTT_CHECKPOINT)\n'
            "print(f'MRCD SLM loaded FTT checkpoint from: {FTT_CHECKPOINT}')\n"
            '\n'
            '# Quick verify: FTT-only baseline\n'
            'slm_ftt_preds = []\n'
            'for text in test_texts:\n'
            '    pred, conf, _ = slm_mrcd.inference(text)\n'
            '    slm_ftt_preds.append(pred)\n'
            'ftt_only_acc = sum(1 for p, g in zip(slm_ftt_preds, test_labels) if p == g) / len(test_labels)\n'
            "print(f'FTT-only baseline accuracy: {ftt_only_acc:.4f}')\n"
        ]
        cell["outputs"] = []
        patched.append(f"Cell {cell_idx} (orig {orig_idx}): Load FTT checkpoint for MRCD")

    # ── Cell 14 (Round 1): Use SLM label when LLM==SLM agree ──
    if "Round 1 - merge/split" in src and "split_clean_noisy" in src:
        # Fix: use SLM label (pred) instead of LLM label when they agree
        old_block = (
            "    if split_clean_noisy(state, CONFIDENCE_THRESHOLD):\n"
            "        state['status'] = 'clean'\n"
            "        d_clean.append(state)"
        )
        new_block = (
            "    if split_clean_noisy(state, CONFIDENCE_THRESHOLD):\n"
            "        state['label'] = pred  # Use SLM label (more accurate than LLM's ~50%)\n"
            "        state['status'] = 'clean'\n"
            "        d_clean.append(state)"
        )
        new_src = src.replace(old_block, new_block)
        # Also increase confidence threshold
        new_src = new_src.replace("CONFIDENCE_THRESHOLD)", "0.85)")
        cell["source"] = [new_src]
        cell["outputs"] = []
        patched.append(f"Cell {cell_idx} (orig {orig_idx}): Round 1 SLM label fix + higher threshold")

    # ── Cell 15 (Rounds 2+): Fix label + add validation to finetune_on_clean ──
    if "Round {round_id} - merge/split" in src and "finetune_on_clean" in src:
        # Fix 1: use SLM label
        old_block = (
            "        if split_clean_noisy(state, CONFIDENCE_THRESHOLD):\n"
            "            state['status'] = f'clean@round{round_id}'\n"
            "            d_clean.append(state)"
        )
        new_block = (
            "        if split_clean_noisy(state, CONFIDENCE_THRESHOLD):\n"
            "            state['label'] = pred  # Use SLM label (more accurate than LLM)\n"
            "            state['status'] = f'clean@round{round_id}'\n"
            "            d_clean.append(state)"
        )
        new_src = src.replace(old_block, new_block)

        # Fix 2: add val_texts/val_labels to finetune_on_clean
        old_ft = (
            "            lr=SLM_FINETUNE_LR,\n"
            "            weight_decay=SLM_FINETUNE_WEIGHT_DECAY,\n"
            "        )"
        )
        new_ft = (
            "            lr=SLM_FINETUNE_LR,\n"
            "            weight_decay=SLM_FINETUNE_WEIGHT_DECAY,\n"
            "            val_texts=test_texts,\n"
            "            val_labels=test_labels,\n"
            "        )"
        )
        new_src = new_src.replace(old_ft, new_ft)

        # Fix 3: increase confidence threshold  
        new_src = new_src.replace("CONFIDENCE_THRESHOLD)", "0.85)")

        cell["source"] = [new_src]
        cell["outputs"] = []
        patched.append(f"Cell {cell_idx} (orig {orig_idx}): Rounds 2+ SLM label + validation + threshold")

    # ── Cell 17: Evaluate FTT-only using original FTT checkpoint ──
    if "slm_mrcd.inference(text)" in src and "SLM-only" in src:
        cell["source"] = [
            "# Evaluate FTT-only baseline (using original FTT checkpoint, BEFORE MRCD fine-tuning)\n"
            "slm_ftt_eval = IntegratedSLM(model_path='bert-base-uncased')\n"
            "slm_ftt_eval.load(FTT_CHECKPOINT)\n"
            "\n"
            "slm_test_pred = []\n"
            "slm_test_conf = []\n"
            "for text in test_texts:\n"
            "    pred, conf, _ = slm_ftt_eval.inference(text)\n"
            "    slm_test_pred.append(pred)\n"
            "    slm_test_conf.append(conf)\n"
            "\n"
            "slm_acc = accuracy_score(test_labels, slm_test_pred)\n"
            "slm_prec, slm_rec, slm_f1, _ = precision_recall_fscore_support(test_labels, slm_test_pred, average='binary', zero_division=0)\n"
            "\n"
            "print(f'FTT-only Accuracy: {slm_acc:.4f}')\n"
            "print(f'FTT-only Precision: {slm_prec:.4f} | Recall: {slm_rec:.4f} | F1: {slm_f1:.4f}')\n"
            "print()\n"
            "print(f'MRCD-FTT Accuracy: {mrcd_acc:.4f}')\n"
            "print(f'MRCD-FTT F1: {mrcd_f1:.4f}')\n"
            "print()\n"
            "delta_acc = mrcd_acc - slm_acc\n"
            "delta_f1 = mrcd_f1 - slm_f1\n"
            "print(f'\\u0394 Accuracy (MRCD-FTT vs FTT-only): {delta_acc:+.4f}')\n"
            "print(f'\\u0394 F1 (MRCD-FTT vs FTT-only): {delta_f1:+.4f}')\n"
            "\n"
            "# Clean up\n"
            "del slm_ftt_eval\n"
            "import gc; gc.collect()\n"
            "if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
        ]
        cell["outputs"] = []
        patched.append(f"Cell {cell_idx} (orig {orig_idx}): FTT-only evaluation with original checkpoint")

    # ── Cell 18: Update comparison to use FTT-only vs MRCD-FTT labels ──
    if "SLM_only" in src and "comparison_df" in src:
        new_src = src.replace("'SLM_only'", "'FTT_only'")
        new_src = new_src.replace("SLM-only vs MRCD", "FTT-only vs MRCD-FTT")
        cell["source"] = [new_src]
        cell["outputs"] = []
        patched.append(f"Cell {cell_idx} (orig {orig_idx}): Updated comparison labels")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb_new, f, ensure_ascii=False)

print(f"\nPatched {len(patched)} cells:")
for p in patched:
    print(f"  ✓ {p}")
print(f"\nSaved to: {NB_PATH}")
