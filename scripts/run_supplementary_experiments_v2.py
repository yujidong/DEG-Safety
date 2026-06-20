#!/usr/bin/env python3
"""
Supplementary Experiments for DEG Paper

Runs three experiments using cached NLI data:
  Exp 1: Held-out axiom generalization
  Exp 2: NLI Direct + Domain VPs (isolates content vs architecture)
  Exp 3: Top-K retrieval vs all-VPs effect

Output: benchmark/results/paper_v2/supplementary_experiments.json
"""
import os
os.environ['HF_HOME'] = r'C:\Users\Public\Models\huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

PROJECT = Path(__file__).resolve().parent.parent

# ====================== Load Data ======================
with open(PROJECT / 'benchmark/data/phase1/queries_all.json', encoding='utf-8') as f:
    queries = json.load(f)
with open(PROJECT / 'benchmark/data/violation_patterns_rewritten.json', encoding='utf-8') as f:
    vps_all = json.load(f)
cache = np.load(PROJECT / 'benchmark/results/nli_probs_cache.npz', allow_pickle=True)
vp_probs = cache['vp_probs']  # [1623, max_vps, 3] with order (C, E, N)

vp_map = defaultdict(list)
for vp in vps_all:
    vp_map[vp['domain']].append(vp)

labels = np.array([0 if q['category'] == 'SAFE' else 1 for q in queries])
domains = np.array([q['domain'] for q in queries])

N = len(queries)
print(f"Loaded {N} queries, {len(vps_all)} VPs")
print(f"VP probs shape: {vp_probs.shape}")

# ====================== Helper: CatLLR via CV ======================
def compute_catllr_scores(vp_indices_fn, train_filter=None):
    """Compute CatLLR scores using 5-fold CV for alpha estimation.

    Args:
        vp_indices_fn: function(domain) -> list of (k_index, vp_object) to use
        train_filter: optional filter for training indices
    Returns:
        scores: np.array of VP CatLLR scores per query
    """
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = np.zeros(N)

    for train_idx, test_idx in skf.split(np.zeros(N), labels):
        # Estimate alpha for each VP
        alpha_lr = {}
        for dom in ['Nuc-Fusion', 'Fin-Audit', 'Med-Intervene']:
            vp_list = vp_indices_fn(dom)
            for k_idx, vp in vp_list:
                oid = vp.get('original_id', vp.get('id', f'{dom}_{k_idx}'))
                tp_list, tl_list = [], []
                for qi in train_idx:
                    if queries[qi]['domain'] != dom:
                        continue
                    tp_list.append(vp_probs[qi, k_idx])
                    tl_list.append(labels[qi])
                if len(tp_list) < 10:
                    continue
                tp_arr = np.array(tp_list)
                tl_arr = np.array(tl_list)
                a_atk = tp_arr[tl_arr == 1].mean(0) + 0.01
                a_safe = tp_arr[tl_arr == 0].mean(0) + 0.01
                a_atk /= a_atk.sum()
                a_safe /= a_safe.sum()
                alpha_lr[oid] = np.log(a_atk / a_safe)

        # Score test set
        for qi in test_idx:
            dom = queries[qi]['domain']
            vp_list = vp_indices_fn(dom)
            s = 0.0
            for k_idx, vp in vp_list:
                oid = vp.get('original_id', vp.get('id', f'{dom}_{k_idx}'))
                if oid in alpha_lr:
                    p = vp_probs[qi, k_idx]
                    s += p[0] * alpha_lr[oid][0] + p[1] * alpha_lr[oid][1] + p[2] * alpha_lr[oid][2]
            scores[qi] = s
    return scores


def compute_metrics(scores, lbls, threshold=None):
    """Compute AUC, best-F1, P/R at best-F1 threshold."""
    scores = np.array(scores)
    lbls = np.array(lbls)
    if len(np.unique(lbls)) < 2:
        return {'auc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'tp': 0, 'fp': 0, 'fn': 0}
    auc = roc_auc_score(lbls, scores)
    # Find best F1 threshold
    best_f1, best_t = 0, 0
    for t in np.percentile(scores, np.arange(0.5, 100, 0.2)):
        pred = (scores > t).astype(int)
        f = f1_score(lbls, pred)
        if f > best_f1:
            best_f1 = f
            best_t = t
    pred = (scores > best_t).astype(int)
    tp = int(((pred == 1) & (lbls == 1)).sum())
    fp = int(((pred == 1) & (lbls == 0)).sum())
    fn = int(((pred == 0) & (lbls == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {'auc': float(auc), 'f1': float(best_f1), 'precision': float(prec),
            'recall': float(rec), 'tp': tp, 'fp': fp, 'fn': fn}


# ====================== Experiment 1: Held-Out Axiom ======================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Held-Out Axiom Generalization")
print("=" * 70)

# Split: first half of each domain's VPs = configured, second half = held-out
all_vp_ids = defaultdict(list)
for dom in ['Nuc-Fusion', 'Fin-Audit', 'Med-Intervene']:
    for k, vp in enumerate(vp_map[dom]):
        oid = vp.get('original_id', vp.get('id', f'{dom}_{k}'))
        all_vp_ids[dom].append((k, vp, oid))

configured_vps = {}  # oid -> (k, vp)
held_out_vps = {}    # oid -> (k, vp)
for dom in all_vp_ids:
    vps_dom = all_vp_ids[dom]
    mid = len(vps_dom) // 2
    for k, vp, oid in vps_dom[:mid]:
        configured_vps[oid] = (k, vp, dom)
    for k, vp, oid in vps_dom[mid:]:
        held_out_vps[oid] = (k, vp, dom)

print(f"Configured VPs: {len(configured_vps)}, Held-out VPs: {len(held_out_vps)}")

# Partition attacks by target axiom
group_a_attacks = []  # target configured VPs
group_b_attacks = []  # target held-out VPs
unmatched_attacks = []

for qi, q in enumerate(queries):
    if labels[qi] != 1:
        continue
    target = q.get('target_axiom_id', '')
    if target in configured_vps:
        group_a_attacks.append(qi)
    elif target in held_out_vps:
        group_b_attacks.append(qi)
    else:
        unmatched_attacks.append(qi)

print(f"Group A attacks (target configured): {len(group_a_attacks)}")
print(f"Group B attacks (target held-out): {len(group_b_attacks)}")
print(f"Unmatched attacks: {len(unmatched_attacks)}")

# Score using ONLY configured VPs
def configured_vps_fn(dom):
    return [(k, vp) for k, vp, oid in all_vp_ids[dom] if oid in configured_vps]

vp_configured_scores = compute_catllr_scores(configured_vps_fn)
print(f"\nVP (configured only) pooled AUC: {roc_auc_score(labels, vp_configured_scores):.4f}")

# Evaluate on each group
qi_a = np.array(group_a_attacks)
qi_b = np.array(group_b_attacks)

if len(qi_a) > 0:
    metrics_a = compute_metrics(vp_configured_scores[qi_a], labels[qi_a])
    print(f"\nGroup A (configured, n={len(qi_a)}): AUC={metrics_a['auc']:.3f} "
          f"R={metrics_a['recall']:.3f} F1={metrics_a['f1']:.3f} FN={metrics_a['fn']}")
else:
    metrics_a = {}

if len(qi_b) > 0:
    metrics_b = compute_metrics(vp_configured_scores[qi_b], labels[qi_b])
    print(f"Group B (held-out, n={len(qi_b)}):  AUC={metrics_b['auc']:.3f} "
          f"R={metrics_b['recall']:.3f} F1={metrics_b['f1']:.3f} FN={metrics_b['fn']}")
else:
    metrics_b = {}

# Also score with ALL VPs for comparison
def all_vps_fn(dom):
    return [(k, vp) for k, vp, oid in all_vp_ids[dom]]

vp_all_scores = compute_catllr_scores(all_vps_fn)
if len(qi_a) > 0:
    metrics_a_full = compute_metrics(vp_all_scores[qi_a], labels[qi_a])
    print(f"\n(Reference) Group A with ALL VPs:  AUC={metrics_a_full['auc']:.3f} "
          f"R={metrics_a_full['recall']:.3f}")
if len(qi_b) > 0:
    metrics_b_full = compute_metrics(vp_all_scores[qi_b], labels[qi_b])
    print(f"(Reference) Group B with ALL VPs:  AUC={metrics_b_full['auc']:.3f} "
          f"R={metrics_b_full['recall']:.3f}")


# ====================== Experiment 2: NLI Direct + Domain VPs ======================
print("\n" + "=" * 70)
print("EXPERIMENT 2: NLI Direct + Domain VPs (P_entail only)")
print("=" * 70)

# Variant A: Mean P(entail) across domain VPs
pentail_scores = np.zeros(N)
for qi in range(N):
    dom = queries[qi]['domain']
    n_vps = len(vp_map[dom])
    pentail_scores[qi] = np.mean([vp_probs[qi, k][1] for k in range(n_vps)])  # p[1] = entail

metrics_pentail = compute_metrics(pentail_scores, labels)
print(f"Mean P(entail):     AUC={metrics_pentail['auc']:.4f} F1={metrics_pentail['f1']:.4f} "
      f"P={metrics_pentail['precision']:.4f} R={metrics_pentail['recall']:.4f}")

# Variant B: Max P(entail) across domain VPs
maxentail_scores = np.zeros(N)
for qi in range(N):
    dom = queries[qi]['domain']
    n_vps = len(vp_map[dom])
    maxentail_scores[qi] = np.max([vp_probs[qi, k][1] for k in range(n_vps)])

metrics_maxentail = compute_metrics(maxentail_scores, labels)
print(f"Max P(entail):      AUC={metrics_maxentail['auc']:.4f} F1={metrics_maxentail['f1']:.4f} "
      f"P={metrics_maxentail['precision']:.4f} R={metrics_maxentail['recall']:.4f}")

# Variant C: Raw E-C sum (for reference, already known ~0.909)
rawec_scores = np.zeros(N)
for qi in range(N):
    dom = queries[qi]['domain']
    n_vps = len(vp_map[dom])
    rawec_scores[qi] = sum(vp_probs[qi, k][1] - vp_probs[qi, k][0] for k in range(n_vps))

metrics_rawec = compute_metrics(rawec_scores, labels)
print(f"Raw E-C sum:        AUC={metrics_rawec['auc']:.4f} F1={metrics_rawec['f1']:.4f} "
      f"P={metrics_rawec['precision']:.4f} R={metrics_rawec['recall']:.4f}")

# Variant D: CatLLR (reference, already known ~0.931)
vp_catllr = compute_catllr_scores(all_vps_fn)
metrics_catllr = compute_metrics(vp_catllr, labels)
print(f"CatLLR (reference): AUC={metrics_catllr['auc']:.4f} F1={metrics_catllr['f1']:.4f} "
      f"P={metrics_catllr['precision']:.4f} R={metrics_catllr['recall']:.4f}")


# ====================== Experiment 3: Top-K Retrieval Effect ======================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Top-K Retrieval vs All-VPs")
print("=" * 70)

# Simulate retrieval: for each query, rank VPs by P(entail) and take top-K
for K in [3, 5, 8, 999]:  # 999 = all
    if K == 999:
        label = "All VPs"
    else:
        label = f"Top-{K}"

    # Score using CatLLR but only on top-K VPs per query
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    scores_topk = np.zeros(N)

    for train_idx, test_idx in skf.split(np.zeros(N), labels):
        # Estimate alpha for ALL VPs (same as before)
        alpha_lr = {}
        for dom in ['Nuc-Fusion', 'Fin-Audit', 'Med-Intervene']:
            for k, vp in enumerate(vp_map[dom]):
                oid = vp.get('original_id', vp.get('id', f'{dom}_{k}'))
                tp_list, tl_list = [], []
                for qi in train_idx:
                    if queries[qi]['domain'] == dom:
                        tp_list.append(vp_probs[qi, k])
                        tl_list.append(labels[qi])
                if len(tp_list) < 10:
                    continue
                tp_arr = np.array(tp_list)
                tl_arr = np.array(tl_list)
                a_atk = tp_arr[tl_arr == 1].mean(0) + 0.01
                a_safe = tp_arr[tl_arr == 0].mean(0) + 0.01
                a_atk /= a_atk.sum()
                a_safe /= a_safe.sum()
                alpha_lr[oid] = np.log(a_atk / a_safe)

        # Score test set with top-K filtering
        for qi in test_idx:
            dom = queries[qi]['domain']
            n_vps = len(vp_map[dom])
            # Rank VPs by P(entail) for this query
            entail_probs = [(vp_probs[qi, k][1], k) for k in range(n_vps)]
            entail_probs.sort(reverse=True)
            top_indices = [k for _, k in entail_probs[:K]]

            s = 0.0
            for k in top_indices:
                vp = vp_map[dom][k]
                oid = vp.get('original_id', vp.get('id', f'{dom}_{k}'))
                if oid in alpha_lr:
                    p = vp_probs[qi, k]
                    s += p[0] * alpha_lr[oid][0] + p[1] * alpha_lr[oid][1] + p[2] * alpha_lr[oid][2]
            scores_topk[qi] = s

    metrics_k = compute_metrics(scores_topk, labels)
    print(f"{label:12s}: AUC={metrics_k['auc']:.4f} F1={metrics_k['f1']:.4f} "
          f"P={metrics_k['precision']:.4f} R={metrics_k['recall']:.4f}")


# ====================== Save Results ======================
output = {
    'description': 'Supplementary experiments using cached NLI data',
    'experiment1_held_out_axiom': {
        'configured_vp_count': len(configured_vps),
        'held_out_vp_count': len(held_out_vps),
        'group_a_attacks': len(group_a_attacks),
        'group_b_attacks': len(group_b_attacks),
        'configured_only_group_a': metrics_a,
        'configured_only_group_b': metrics_b,
        'all_vps_group_a': metrics_a_full if 'metrics_a_full' in dir() else {},
        'all_vps_group_b': metrics_b_full if 'metrics_b_full' in dir() else {},
    },
    'experiment2_nli_aggregation_comparison': {
        'mean_p_entail': metrics_pentail,
        'max_p_entail': metrics_maxentail,
        'raw_ec': metrics_rawec,
        'catllr': metrics_catllr,
    },
    'experiment3_topk_effect': {
        'note': 'CatLLR scores computed on top-K VPs per query (ranked by P_entail)',
    },
}

out_path = PROJECT / 'benchmark/results/paper_v2/supplementary_experiments.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
