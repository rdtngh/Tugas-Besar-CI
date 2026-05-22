"""
Simplified Neuro-Fuzzy / ANFIS model untuk prediksi stress mahasiswa.

Model ini menggunakan membership function Gaussian adaptif sebagai lapisan fuzzy,
rule layer sederhana, normalisasi firing strength, dan consequent layer linear.
Meskipun bukan implementasi ANFIS penuh, model ini valid secara konsep sebagai
Neuro-Fuzzy dengan parameter membership yang dapat dilatih dari data.
"""

import numpy as np
import pandas as pd
from joblib import dump, load
from pathlib import Path

FEATURES = [
    'anxiety_level',
    'sleep_quality',
    'study_load',
    'academic_performance',
    'social_support',
    'future_career_concerns',
]
CLASS_LABELS = ['Rendah', 'Sedang', 'Tinggi']


def gaussian_membership(x, center, sigma):
    sigma = np.maximum(sigma, 1e-3)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


class SimplifiedANFIS:
    """Simplified ANFIS model dengan membership function adaptif."""

    def __init__(self, input_dim=6, num_terms=3, num_classes=3, random_state=42):
        self.input_dim = input_dim
        self.num_terms = num_terms
        self.num_classes = num_classes
        self.random_state = random_state
        rng = np.random.default_rng(random_state)

        # Premise parameters: center dan sigma untuk setiap fitur dan fuzzy term
        self.centers = np.tile(np.linspace(0.1, 0.9, num_terms), (input_dim, 1))
        self.sigmas = np.full((input_dim, num_terms), 0.15)

        # Consequent parameters: linear weights untuk setiap class
        self.weights = rng.normal(0, 0.1, size=(num_classes, input_dim * num_terms))
        self.bias = np.zeros(num_classes)

    def _fuzzify(self, X):
        """Layer 1: fuzzifikasi dengan Gaussian membership functions."""
        # X shape: (batch_size, input_dim)
        x_expanded = X[:, :, np.newaxis]
        centers = self.centers[np.newaxis, :, :]
        sigmas = self.sigmas[np.newaxis, :, :]
        membership = gaussian_membership(x_expanded, centers, sigmas)
        return membership

    def _normalize(self, rule_strength):
        """Layer 3: normalisasi firing strength."""
        total = np.sum(rule_strength, axis=1, keepdims=True) + 1e-9
        return rule_strength / total

    def _softmax(self, logits):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass untuk menghasilkan probabilitas kelas."""
        membership = self._fuzzify(X)
        rule_strength = membership.reshape(X.shape[0], -1)
        normalized = self._normalize(rule_strength)
        logits = normalized.dot(self.weights.T) + self.bias
        probs = self._softmax(logits)
        return {
            'membership': membership,
            'rule_strength': rule_strength,
            'normalized_strength': normalized,
            'logits': logits,
            'probs': probs,
        }

    def predict(self, X):
        """Prediksi kelas untuk input X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        result = self.forward(X)
        idx = np.argmax(result['probs'], axis=1)
        labels = [CLASS_LABELS[i] for i in idx]
        return labels, result['probs']

    def train(self, X, y, epochs=250, lr=0.03, batch_size=32, verbose=True):
        """Melatih model menggunakan gradient descent sederhana."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        num_samples = X.shape[0]
        y_onehot = np.eye(self.num_classes)[y]

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                forward = self.forward(X_batch)
                probs = forward['probs']
                normalized = forward['normalized_strength']
                rule_strength = forward['rule_strength']

                # Cross entropy loss
                loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-9), axis=1))

                # Backprop untuk consequent layer
                d_logits = (probs - y_batch) / X_batch.shape[0]
                grad_w = d_logits.T.dot(normalized)
                grad_b = np.sum(d_logits, axis=0)

                # Backprop untuk normalisasi
                w_dot = d_logits.dot(self.weights)
                sum_rule = np.sum(rule_strength, axis=1, keepdims=True)
                numerator = w_dot * sum_rule - np.sum(w_dot * rule_strength, axis=1, keepdims=True)
                d_rule = numerator / (sum_rule ** 2 + 1e-9)
                d_rule = d_rule.reshape(X_batch.shape[0], self.input_dim, self.num_terms)

                # Backprop untuk membership parameters
                x_expanded = X_batch[:, :, np.newaxis]
                mu = forward['membership']
                diff = x_expanded - self.centers[np.newaxis, :, :]
                sigma = np.maximum(self.sigmas, 1e-3)[np.newaxis, :, :]

                d_mu_d_center = mu * diff / (sigma ** 2)
                d_mu_d_sigma = mu * (diff ** 2) / (sigma ** 3)

                grad_centers = np.sum(d_rule * d_mu_d_center, axis=0)
                grad_sigmas = np.sum(d_rule * d_mu_d_sigma, axis=0)

                # Update parameters
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
                self.centers -= lr * grad_centers
                self.sigmas -= lr * grad_sigmas
                self.sigmas = np.clip(self.sigmas, 0.03, 1.0)

            if verbose and epoch % 50 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == np.argmax(y_batch, axis=1))
                print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - batch acc: {acc:.4f}")

        return self

    def save(self, model_path):
        """Simpan parameter model ke file."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        dump({
            'centers': self.centers,
            'sigmas': self.sigmas,
            'weights': self.weights,
            'bias': self.bias,
            'input_dim': self.input_dim,
            'num_terms': self.num_terms,
            'num_classes': self.num_classes,
        }, model_path)

    @classmethod
    def load(cls, model_path):
        """Muat parameter model dari file."""
        data = load(model_path)
        model = cls(input_dim=data['input_dim'], num_terms=data['num_terms'], num_classes=data['num_classes'])
        model.centers = data['centers']
        model.sigmas = data['sigmas']
        model.weights = data['weights']
        model.bias = data['bias']
        return model


def load_anfis_model(model_path='models/anfis_model.pkl'):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ANFIS model tidak ditemukan di {model_path}")
    return SimplifiedANFIS.load(model_path)


def load_scaler(scaler_path='models/anfis_scaler.pkl'):
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler ANFIS tidak ditemukan di {scaler_path}")
    return load(scaler_path)


def predict_anfis(input_data, model, scaler):
    """Prediksi single sample raw input melalui ANFIS."""
    x = np.array([
        input_data['anxiety_level'],
        input_data['sleep_quality'],
        input_data['study_load'],
        input_data['academic_performance'],
        input_data['social_support'],
        input_data['future_career_concerns'],
    ], dtype=float).reshape(1, -1)
    x_df = pd.DataFrame(x, columns=FEATURES)
    x_scaled = scaler.transform(x_df)
    labels, probs = model.predict(x_scaled)
    label = labels[0]
    probs = probs[0]
    return {
        'label': label,
        'score': float(np.max(probs)),
        'class_scores': {
            'Rendah': float(probs[0]),
            'Sedang': float(probs[1]),
            'Tinggi': float(probs[2]),
        },
    }
