"""
Density Ratio Estimator for Embedding-space Energy.

THEORETICAL BASIS:
In the Free Energy Principle, the pragmatic value is:
    F_pragmatic = D_KL[Q(s|z,a_k) || P(s|a_k)]

For queries with "No state change" consequence, NLI cannot provide Q(s|z,a_k).
Instead, we estimate the density ratio from the embedding space:

    r(z) = P(attack | embedding(z)) / P(safe | embedding(z))
         = P(y=1 | x) / (1 - P(y=1 | x))

This is equivalent to KL divergence estimation via density ratio:
    D_KL[P(attack) || P(safe)] ≈ E[r(z) log r(z)]

IMPLEMENTATION:
- Logistic regression on top of all-MiniLM-L6-v2 embeddings
- Trained on labeled (safe/attack) queries
- Output: continuous energy signal in [0, +inf)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
import pickle
import os


class DensityRatioEstimator:
    """
    Estimate P(attack|embedding) / P(safe|embedding) using logistic regression.

    This provides a continuous energy signal from the embedding space,
    replacing the binary "No state change" → 0.0 mapping.
    """

    def __init__(self, C=1.0, calibration_samples=500):
        """
        Args:
            C: Logistic regression regularization parameter
            calibration_samples: Number of samples to use for calibration
        """
        self.C = C
        self.calibration_samples = calibration_samples
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self._embedding_model = None
        self._embedding_model_name = "all-MiniLM-L6-v2"
        self._is_fitted = False

    def _load_embedding_model(self):
        if self._embedding_model is None:
            # Use transformers directly to avoid sentence_transformers compatibility issues
            import torch
            from transformers import AutoTokenizer, AutoModel
            import os

            model_name = self._embedding_model_name
            # Try local cache first
            cache_dir = os.path.expanduser(
                "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
            )
            local_path = None
            if os.path.isdir(cache_dir):
                import glob
                snapshots = glob.glob(os.path.join(cache_dir, "*"))
                for snap in snapshots:
                    if os.path.exists(os.path.join(snap, "model.safetensors")):
                        local_path = snap
                        break

            if local_path:
                self._embedding_tokenizer = AutoTokenizer.from_pretrained(local_path)
                self._embedding_model = AutoModel.from_pretrained(
                    local_path,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                    f"sentence-transformers/{model_name}"
                )
                self._embedding_model = AutoModel.from_pretrained(
                    f"sentence-transformers/{model_name}",
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            self._embedding_model.eval()

    def _encode(self, texts):
        """Encode texts to embeddings using mean pooling."""
        import torch
        inputs = self._embedding_tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        device = next(self._embedding_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._embedding_model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return embeddings.cpu().numpy()

    def preload(self):
        self._load_embedding_model()

    def fit(self, queries: List[str], labels: List[int]):
        """
        Fit the density ratio estimator on labeled data.

        Args:
            queries: List of query strings
            labels: List of labels (0=safe, 1=attack)
        """
        self._load_embedding_model()

        # Encode all queries
        embeddings = self._encode(queries)

        # Fit scaler and model
        X = self.scaler.fit_transform(embeddings)
        self.model.fit(X, labels)
        self._is_fitted = True

        # Compute training accuracy for diagnostics
        train_preds = self.model.predict(X)
        train_acc = np.mean(train_preds == labels)
        train_probs = self.model.predict_proba(X)[:, 1]

        # Compute class separation
        safe_probs = train_probs[np.array(labels) == 0]
        attack_probs = train_probs[np.array(labels) == 1]
        print(f"[DensityRatio] Training: acc={train_acc:.3f}")
        print(f"  Safe P(attack):  mean={safe_probs.mean():.3f} std={safe_probs.std():.3f}")
        print(f"  Attack P(attack): mean={attack_probs.mean():.3f} std={attack_probs.std():.3f}")

        # Cohen's d
        pooled_std = np.sqrt((safe_probs.std()**2 + attack_probs.std()**2) / 2)
        cohens_d = (attack_probs.mean() - safe_probs.mean()) / (pooled_std + 1e-9)
        print(f"  Cohen's d = {cohens_d:.3f}")

        return self

    def predict_energy(self, query: str) -> float:
        """
        Predict continuous energy from density ratio.

        Energy = -log(P(safe|emb)) = -log(1 - P(attack|emb))
        Higher energy = more likely attack

        This formulation:
        - Energy ≈ 0 when P(attack) ≈ 0 (clearly safe)
        - Energy ≈ 1 when P(attack) ≈ 0.5 (ambiguous)
        - Energy → +inf when P(attack) → 1 (clearly attack)
        """
        if not self._is_fitted:
            return 0.0

        self._load_embedding_model()

        emb = self._encode([query])
        X = self.scaler.transform(emb)

        p_attack = self.model.predict_proba(X)[0, 1]
        p_safe = 1.0 - p_attack

        # Energy = -log(P(safe)) — this is the surprisal of the safe hypothesis
        # Clamped to avoid log(0)
        energy = -np.log(max(p_safe, 1e-9))

        return float(energy)

    def predict_attack_proba(self, query: str) -> float:
        """
        Predict P(attack|embedding) directly.

        Used as the recognition model in the FEP dual-channel precision:
        the DR acts as a learned recognition density that infers the latent
        safety state from raw observations (sentence embeddings).

        Args:
            query: Query string

        Returns:
            P(attack|embedding) in [0, 1]
        """
        if not self._is_fitted:
            return 0.5  # Uniform prior when untrained

        self._load_embedding_model()

        emb = self._encode([query])
        X = self.scaler.transform(emb)

        p_attack = float(self.model.predict_proba(X)[0, 1])
        return p_attack

    def predict_energies_batch(self, queries: List[str]) -> np.ndarray:
        """Batch predict energies for multiple queries."""
        if not self._is_fitted:
            return np.zeros(len(queries))

        self._load_embedding_model()

        embeddings = self._encode(queries)
        X = self.scaler.transform(embeddings)

        probs = self.model.predict_proba(X)
        p_safe = probs[:, 0]

        energies = -np.log(np.maximum(p_safe, 1e-9))
        return energies

    def save(self, path: str):
        """Save fitted model to disk."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'C': self.C,
            'is_fitted': self._is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[DensityRatio] Saved to {path}")

    def load(self, path: str):
        """Load fitted model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.C = data['C']
        self._is_fitted = data['is_fitted']
        print(f"[DensityRatio] Loaded from {path}")
        return self


def train_density_ratio(
    queries_file: str = "benchmark/data/phase1/queries_CLEAN_FINAL_20260202_023554.json",
    output_path: str = "benchmark/models/density_ratio_estimator.pkl",
    test_split: float = 0.2,
    seed: int = 42
):
    """Train density ratio estimator on benchmark data."""
    import json

    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Prepare data
    texts = [q['query'] for q in queries]
    labels = [0 if q['category'] == 'SAFE' else 1 for q in queries]

    print(f"Training data: {len(texts)} queries, {sum(labels)} attack, {len(labels)-sum(labels)} safe")

    # Train/test split
    np.random.seed(seed)
    indices = np.random.permutation(len(texts))
    split = int(len(texts) * (1 - test_split))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_queries = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_queries = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Train: {len(train_queries)} ({sum(train_labels)} attack)")
    print(f"Test:  {len(test_queries)} ({sum(test_labels)} attack)")

    # Fit model
    estimator = DensityRatioEstimator(C=1.0)
    estimator.fit(train_queries, train_labels)

    # Evaluate on test set
    test_energies = estimator.predict_energies_batch(test_queries)
    test_labels_arr = np.array(test_labels)

    safe_e = test_energies[test_labels_arr == 0]
    attack_e = test_energies[test_labels_arr == 1]

    print(f"\nTest Set Evaluation:")
    print(f"  Safe energies:  mean={safe_e.mean():.3f} std={safe_e.std():.3f} median={np.median(safe_e):.3f}")
    print(f"  Attack energies: mean={attack_e.mean():.3f} std={attack_e.std():.3f} median={np.median(attack_e):.3f}")

    # Best threshold
    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0.1, 5.0, 0.05):
        tp = sum(1 for e in attack_e if e > thresh)
        fp = sum(1 for e in safe_e if e > thresh)
        fn = sum(1 for e in attack_e if e <= thresh)
        tn = sum(1 for e in safe_e if e <= thresh)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    tp = sum(1 for e in attack_e if e > best_thresh)
    fp = sum(1 for e in safe_e if e > best_thresh)
    fn = sum(1 for e in attack_e if e <= best_thresh)
    tn = sum(1 for e in safe_e if e <= best_thresh)
    acc = (tp + tn) / (tp + fp + fn + tn)

    print(f"\n  Best threshold: {best_thresh:.2f}")
    print(f"  F1={best_f1:.3f} Acc={acc:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    estimator.save(output_path)

    return estimator


if __name__ == "__main__":
    train_density_ratio()
