"""
Contrastive Learning Module for Hydrocephalus Postoperative Infection Prediction.

This module implements a contrastive learning framework with hard sample mining
for predicting postoperative infections in hydrocephalus patients.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class HydroDataset(Dataset):
    """
    Dataset class for hydrocephalus data with hard sample mining.

    This dataset implements hard positive and negative sample mining based on
    distance matrices to improve contrastive learning performance.

    Args:
        features: Input features array
        labels: Target labels array
        dist_matrix: Precomputed distance matrix between samples
        k: Number of hard samples to mine (default: 5)
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dist_matrix: np.ndarray,
        k: int = 5,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels).view(-1)
        self.dist_matrix = dist_matrix
        self.num_samples = len(labels)
        self.k = k

        self.hard_pos_indices = []
        self.hard_neg_indices = []
        self._mine_hard_samples()

    def _mine_hard_samples(self):
        """Mine hard positive and negative samples for each anchor."""
        labels_np = self.labels.numpy()

        for i in range(self.num_samples):
            current_label = labels_np[i]
            distances = self.dist_matrix[i]

            # --- Mine Top-K Hard Positive Samples ---
            # Select top K samples with largest distance from same class
            pos_mask = labels_np == current_label
            pos_mask[i] = False  # Exclude self
            pos_candidates = np.where(pos_mask)[0]

            if len(pos_candidates) >= self.k:
                sorted_indices = np.argsort(distances[pos_candidates])[::-1]
                hard_pos_idx = pos_candidates[sorted_indices[: self.k]]
            elif len(pos_candidates) > 0:
                indices = np.argsort(distances[pos_candidates])[::-1]
                hard_pos_idx = np.random.choice(
                    pos_candidates[indices], self.k, replace=True
                )
            else:
                hard_pos_idx = np.full(self.k, i)

            # --- Mine Top-K Hard Negative Samples ---
            # Select top K samples with smallest distance from different class
            neg_mask = labels_np != current_label
            neg_candidates = np.where(neg_mask)[0]

            if len(neg_candidates) >= self.k:
                sorted_indices = np.argsort(distances[neg_candidates])
                hard_neg_idx = neg_candidates[sorted_indices[: self.k]]
            elif len(neg_candidates) > 0:
                indices = np.argsort(distances[neg_candidates])
                hard_neg_idx = np.random.choice(
                    neg_candidates[indices], self.k, replace=True
                )
            else:
                hard_neg_idx = np.full(self.k, i)

            self.hard_pos_indices.append(hard_pos_idx)
            self.hard_neg_indices.append(hard_neg_idx)

        self.hard_pos_indices = np.array(self.hard_pos_indices)
        self.hard_neg_indices = np.array(self.hard_neg_indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Get a sample with its hard positive and negative samples.

        Returns:
            Tuple of (anchor_features, label, hard_pos_features, hard_neg_features)
        """
        return (
            self.features[idx],
            self.labels[idx],
            self.features[self.hard_pos_indices[idx]],
            self.features[self.hard_neg_indices[idx]],
        )


class ContrastiveNetwork(nn.Module):
    """
    Contrastive neural network with encoder, projector, and classifier.

    Architecture consists of:
    - Encoder: Multi-layer backbone with BatchNorm for feature extraction
    - Projector: Projection head for contrastive learning
    - Classifier: Classification head with sigmoid output

    Args:
        input_dim: Input feature dimension (default: 46)
        hidden_dim: Hidden layer dimension (default: 256)
        proj_dim: Projection dimension (default: 64)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dim: int = 256,
        proj_dim: int = 64,
        dropout: float = 0.3,
    ):
        super(ContrastiveNetwork, self).__init__()

        # Encoder backbone with BatchNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        # Projection head for InfoNCE loss
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim // 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Classification head for Focal loss
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Tuple of (probabilities, embeddings)
        """
        features = self.encoder(x)
        embeddings = self.projector(features)
        probs = self.classifier(features)
        return probs, embeddings


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Focal Loss and Modified InfoNCE Loss.

    This loss function combines classification loss with contrastive learning
    loss for improved performance on imbalanced medical data.

    Args:
        alpha: Focal loss balancing factor (default: 0.25)
        gamma: Focal loss focusing factor (default: 2.0)
        temperature: Temperature coefficient for contrastive loss (default: 0.5)
        lambda_val: Weight for contrastive loss (default: 0.5)
        pos_weight: Positive sample weight for handling imbalance (default: None)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        temperature: float = 0.5,
        lambda_val: float = 0.5,
        pos_weight: float = None,
    ):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.t = temperature
        self.lambda_val = lambda_val
        self.pos_weight = pos_weight

    def focal_loss(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        Compute focal loss for classification.

        Args:
            probs: Predicted probabilities
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        probs = probs.view(-1)
        targets = targets.float()

        # Avoid log(0)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        # Positive class loss
        pos_term = (
            -self.alpha
            * ((1 - probs) ** self.gamma)
            * torch.log(probs)
            * targets
        )

        # Negative class loss
        neg_term = (
            -(1 - self.alpha)
            * (probs ** self.gamma)
            * torch.log(1 - probs)
            * (1 - targets)
        )

        # Apply positive sample weight
        if self.pos_weight is not None:
            pos_term = pos_term * self.pos_weight

        return (pos_term + neg_term).mean()

    def modified_infonce_loss(
        self,
        z_anchor: torch.Tensor,
        z_hard_pos: torch.Tensor,
        z_hard_neg: torch.Tensor,
    ):
        """
        Compute modified InfoNCE loss for contrastive learning.

        Args:
            z_anchor: Anchor embeddings
            z_hard_pos: Hard positive sample embeddings
            z_hard_neg: Hard negative sample embeddings

        Returns:
            Contrastive loss value
        """
        B, K, D = z_hard_pos.shape

        # Normalize embeddings
        z_anchor = F.normalize(z_anchor, dim=1).unsqueeze(1)
        z_hard_pos = F.normalize(z_hard_pos, dim=2)
        z_hard_neg = F.normalize(z_hard_neg, dim=2)

        # Compute similarities
        sim_pos = torch.sum(z_anchor * z_hard_pos, dim=2) / self.t
        sim_neg = torch.sum(z_anchor * z_hard_neg, dim=2) / self.t

        # Concatenate logits
        logits = torch.cat([sim_pos, sim_neg], dim=1)

        # LogSumExp of all logits
        log_sum_exp_all = torch.logsumexp(logits, dim=1, keepdim=True)

        # Mean log probability of positives
        mean_log_prob_pos = sim_pos.mean(dim=1, keepdim=True)
        loss = log_sum_exp_all - mean_log_prob_pos

        return loss.mean()

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        z_anchor: torch.Tensor,
        z_hard_pos: torch.Tensor,
        z_hard_neg: torch.Tensor,
    ):
        """
        Compute combined hybrid loss.

        Returns:
            Combined loss value
        """
        l_fl = self.focal_loss(probs, targets)
        l_infonce = self.modified_infonce_loss(z_anchor, z_hard_pos, z_hard_neg)
        return l_fl + self.lambda_val * l_infonce


class HydrocephalusContrastiveModel(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for contrastive learning model.

    This class provides a scikit-learn compatible interface for training and
    inference with the contrastive learning model, including data preprocessing,
    training with early stopping, and prediction.

    Args:
        max_epochs: Maximum training epochs (default: 150)
        batch_size: Training batch size (default: 32)
        lr: Learning rate (default: 1e-3)
        device: Computation device (default: 'cuda')
        k: Number of hard samples (default: 5)
        early_stopping_patience: Patience for early stopping (default: 20)
        validation_split: Validation set ratio (default: 0.15)
        class_weight_scale: Class weight scaling factor (default: 2.0)
        pos_threshold: Positive class threshold (default: 0.4)
    """

    def __init__(
        self,
        max_epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cuda",
        k: int = 5,
        early_stopping_patience: int = 20,
        validation_split: float = 0.15,
        class_weight_scale: float = 2.0,
        pos_threshold: float = 0.4,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if torch.cuda.is_available() else "cpu"
        self.k = k
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.class_weight_scale = class_weight_scale
        self.pos_threshold = pos_threshold

        # Initialize imputer and scaler
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_jobs=-1, random_state=42),
            max_iter=10,
            random_state=42,
        )
        self.scaler = MinMaxScaler()
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given data.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            self: Trained model instance
        """
        # Data preprocessing: MICE imputation -> MinMax scaling
        X_imputed = self.imputer.fit_transform(X)
        X_normalized = self.scaler.fit_transform(X_imputed)

        # Calculate class weights
        class_counts = np.bincount(y.astype(int))
        pos_weight = (
            self.class_weight_scale * (class_counts[0] / class_counts[1])
            if len(class_counts) > 1
            else 1.0
        )
        print(f"Class distribution: {class_counts}, Positive weight: {pos_weight:.2f}")

        # Split training and validation sets
        if self.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_normalized, y, test_size=self.validation_split, stratify=y, random_state=42
            )
            dist_matrix_val = cdist(X_val, X_val, metric="euclidean")
            val_dataset = HydroDataset(X_val, y_val, dist_matrix_val, k=self.k)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            X_train, y_train = X_normalized, y

        # Compute distance matrix for hard sample mining
        dist_matrix = cdist(X_train, X_train, metric="euclidean")

        # Create dataset and dataloader
        dataset = HydroDataset(X_train, y_train, dist_matrix, k=self.k)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model, optimizer, and scheduler
        input_dim = X.shape[1]
        self.model = ContrastiveNetwork(input_dim=input_dim).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        criterion = HybridLoss(pos_weight=pos_weight).to(self.device)
        self.model.train()

        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                feats_anchor, labels, feats_pos, feats_neg = [
                    b.to(self.device) for b in batch
                ]

                optimizer.zero_grad()

                probs, z_anchor = self.model(feats_anchor)

                B, K, D = feats_pos.shape

                # Encode hard samples
                _, z_pos = self.model(feats_pos.view(B * K, D))
                z_pos = z_pos.view(B, K, -1)

                _, z_neg = self.model(feats_neg.view(B * K, D))
                z_neg = z_neg.view(B, K, -1)

                loss = criterion(probs, labels, z_anchor, z_pos, z_neg)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches

            # Validation and early stopping
            if self.validation_split > 0:
                val_loss = self._validate(val_loader, criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                scheduler.step(avg_train_loss)

        return self

    def _validate(self, val_loader: DataLoader, criterion: HybridLoss):
        """
        Validate the model on validation set.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                feats_anchor, labels, feats_pos, feats_neg = [
                    b.to(self.device) for b in batch
                ]

                probs, z_anchor = self.model(feats_anchor)

                B, K, D = feats_pos.shape
                _, z_pos = self.model(feats_pos.view(B * K, D))
                z_pos = z_pos.view(B, K, -1)
                _, z_neg = self.model(feats_neg.view(B * K, D))
                z_neg = z_neg.view(B, K, -1)

                loss = criterion(probs, labels, z_anchor, z_pos, z_neg)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def predict_proba(self, X: np.ndarray):
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Class probabilities array of shape (N, 2)
        """
        self.model.eval()
        X_imputed = self.imputer.transform(X)
        X_normalized = self.scaler.transform(X_imputed)

        X_tensor = torch.FloatTensor(X_normalized).to(self.device)

        with torch.no_grad():
            probs, _ = self.model(X_tensor)
            probs = probs.cpu().numpy()

        # Format as (N, 2) for sklearn compatibility
        return np.hstack([1 - probs, probs])

    def predict(self, X: np.ndarray):
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > self.pos_threshold).astype(int)
