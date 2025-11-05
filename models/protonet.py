"""
Prototypical Network for Few-Shot Learning
Implements the Prototypical Network architecture for rare cancer classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict


class Encoder(nn.Module):
    """Feature encoder using pre-trained CNN backbone."""
    
    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 128):
        """
        Args:
            backbone: Pre-trained backbone model ('resnet50' or 'efficientnet')
            embedding_dim: Dimension of the embedding space
        """
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            # Get the feature dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                dummy_features = self.feature_extractor(dummy_input)
                feature_dim = dummy_features.view(1, -1).shape[1]
            
            # Projection layer to embedding space
            self.projection = nn.Linear(feature_dim, embedding_dim)
            
        elif backbone == "efficientnet":
            # For EfficientNet, we'll use a simplified version
            # In practice, you'd use timm library or torchvision
            resnet = models.resnet50(pretrained=True)  # Fallback
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                dummy_features = self.feature_extractor(dummy_input)
                feature_dim = dummy_features.view(1, -1).shape[1]
            self.projection = nn.Linear(feature_dim, embedding_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone = backbone
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input images to embedding space.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ClinicalEncoder(nn.Module):
    """Encoder for clinical data (age, gender, stage, mutations)."""
    
    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Dimension of clinical embedding
        """
        super(ClinicalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding layers for categorical features
        self.gender_embed = nn.Embedding(2, 8)  # M/F
        self.stage_embed = nn.Embedding(4, 16)  # I/II/III/IV
        self.mutation_embed = nn.Embedding(5, 16)  # BRAF/KRAS/TP53/None/Other
        
        # Linear layers for continuous features
        self.age_fc = nn.Linear(1, 16)
        self.tumor_size_fc = nn.Linear(1, 16)
        
        # Combine all features
        self.fc = nn.Sequential(
            nn.Linear(8 + 16 + 16 + 16 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, clinical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode clinical data to embedding space.
        
        Args:
            clinical_data: Dictionary with keys:
                - gender: [batch_size] (0 or 1)
                - stage: [batch_size] (0-3)
                - mutation: [batch_size] (0-4)
                - age: [batch_size, 1]
                - tumor_size: [batch_size, 1]
                
        Returns:
            Clinical embeddings [batch_size, embedding_dim]
        """
        gender_emb = self.gender_embed(clinical_data['gender'])
        stage_emb = self.stage_embed(clinical_data['stage'])
        mutation_emb = self.mutation_embed(clinical_data['mutation'])
        age_emb = self.age_fc(clinical_data['age'])
        tumor_size_emb = self.tumor_size_fc(clinical_data['tumor_size'])
        
        # Concatenate all embeddings
        combined = torch.cat([gender_emb, stage_emb, mutation_emb, age_emb, tumor_size_emb], dim=1)
        clinical_embedding = self.fc(combined)
        
        return clinical_embedding


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot learning.
    Supports both image-only and multi-modal (image + clinical) modes.
    """
    
    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 128,
                 use_clinical: bool = False, clinical_dim: int = 64):
        """
        Args:
            backbone: Pre-trained backbone for image encoder
            embedding_dim: Dimension of image embeddings
            use_clinical: Whether to use clinical data
            clinical_dim: Dimension of clinical embeddings
        """
        super(PrototypicalNetwork, self).__init__()
        
        self.use_clinical = use_clinical
        self.embedding_dim = embedding_dim
        
        # Image encoder
        self.image_encoder = Encoder(backbone, embedding_dim)
        
        # Clinical encoder (optional)
        if use_clinical:
            self.clinical_encoder = ClinicalEncoder(clinical_dim)
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim + clinical_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            final_dim = embedding_dim
        else:
            self.clinical_encoder = None
            final_dim = embedding_dim
        
        # Final normalization
        self.final_norm = nn.LayerNorm(final_dim)
    
    def forward(self, images: torch.Tensor, 
                clinical_data: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            clinical_data: Optional clinical data dictionary
            
        Returns:
            Combined embeddings [batch_size, embedding_dim]
        """
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        if self.use_clinical and clinical_data is not None:
            # Encode clinical data
            clinical_embeddings = self.clinical_encoder(clinical_data)
            
            # Fuse modalities
            combined = torch.cat([image_embeddings, clinical_embeddings], dim=1)
            embeddings = self.fusion(combined)
        else:
            embeddings = image_embeddings
        
        # Final normalization
        embeddings = self.final_norm(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_prototypes(self, support_embeddings: torch.Tensor, 
                          support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set embeddings.
        
        Args:
            support_embeddings: Support set embeddings [n_support, embedding_dim]
            support_labels: Support set labels [n_support]
            
        Returns:
            Prototypes [n_classes, embedding_dim]
        """
        n_classes = support_labels.unique().shape[0]
        prototypes = []
        
        for class_idx in range(n_classes):
            # Get embeddings for this class
            class_mask = (support_labels == class_idx)
            class_embeddings = support_embeddings[class_mask]
            
            # Compute mean prototype
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        return prototypes
    
    def compute_distances(self, query_embeddings: torch.Tensor,
                         prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from query embeddings to prototypes.
        
        Args:
            query_embeddings: Query embeddings [n_query, embedding_dim]
            prototypes: Class prototypes [n_classes, embedding_dim]
            
        Returns:
            Negative distances (logits) [n_query, n_classes]
        """
        # Euclidean distance
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        # Convert to logits (negative distance)
        logits = -distances
        return logits

