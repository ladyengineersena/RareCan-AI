"""
Episode-based Dataset for Few-Shot Learning
Implements episode sampling for N-way K-shot learning tasks.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class RareCancerDataset(Dataset):
    """Dataset for rare cancer few-shot learning."""
    
    def __init__(self, metadata_path: str, image_dir: str, 
                 transform=None, use_clinical: bool = True):
        """
        Args:
            metadata_path: Path to metadata JSON file
            image_dir: Directory containing images
            transform: Image transformations
            use_clinical: Whether to include clinical data
        """
        self.image_dir = image_dir
        self.transform = transform
        self.use_clinical = use_clinical
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Flatten metadata and create class mapping
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        for class_idx, (cancer_type, samples) in enumerate(self.metadata.items()):
            self.class_to_idx[cancer_type] = class_idx
            self.idx_to_class[class_idx] = cancer_type
            
            for sample in samples:
                sample['class_idx'] = class_idx
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_dir, os.path.basename(image_path))
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Prepare clinical data
        clinical_data = None
        if self.use_clinical:
            clinical_data = self._prepare_clinical_data(sample)
        
        return {
            'image': image,
            'label': sample['class_idx'],
            'cancer_type': sample['cancer_type'],
            'clinical_data': clinical_data,
            'image_id': sample['image_id']
        }
    
    def _prepare_clinical_data(self, sample: Dict) -> Dict[str, int]:
        """Prepare clinical data for model input."""
        # Map gender to index
        gender_map = {'M': 0, 'F': 1}
        
        # Map stage to index
        stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        
        # Map mutation to index
        mutation_map = {'BRAF': 0, 'KRAS': 1, 'TP53': 2, 'None': 3, 'Other': 4}
        
        return {
            'gender': gender_map.get(sample.get('gender', 'M'), 0),
            'stage': stage_map.get(sample.get('tumor_stage', 'I'), 0),
            'mutation': mutation_map.get(sample.get('genetic_mutation', 'None'), 3),
            'age': float(sample.get('age', 50)),
            'tumor_size': float(sample.get('tumor_size_mm', 30.0))
        }


class EpisodeSampler:
    """Samples N-way K-shot episodes for few-shot learning."""
    
    def __init__(self, dataset: RareCancerDataset, n_way: int = 5, 
                 k_shot: int = 5, n_query: int = 15):
        """
        Args:
            dataset: RareCancerDataset instance
            n_way: Number of classes per episode
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        
        # Group samples by class
        self.class_samples = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            class_idx = sample['class_idx']
            self.class_samples[class_idx].append(idx)
        
        # Filter classes with enough samples
        self.valid_classes = [
            class_idx for class_idx, samples in self.class_samples.items()
            if len(samples) >= (k_shot + n_query)
        ]
        
        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Not enough classes with sufficient samples. "
                f"Need {n_way} classes, but only {len(self.valid_classes)} available."
            )
        
        print(f"Episode sampler: {n_way}-way {k_shot}-shot with {n_query} queries")
        print(f"Valid classes: {len(self.valid_classes)}")
    
    def sample_episode(self) -> Tuple[List[int], List[int]]:
        """
        Sample a single episode.
        
        Returns:
            Tuple of (support_indices, query_indices)
        """
        # Randomly select n_way classes
        selected_classes = np.random.choice(
            self.valid_classes, size=self.n_way, replace=False
        )
        
        support_indices = []
        query_indices = []
        
        for class_idx in selected_classes:
            # Get all samples for this class
            class_sample_indices = self.class_samples[class_idx]
            
            # Randomly sample support and query
            np.random.shuffle(class_sample_indices)
            support_indices.extend(class_sample_indices[:self.k_shot])
            query_indices.extend(class_sample_indices[self.k_shot:self.k_shot + self.n_query])
        
        return support_indices, query_indices


class EpisodeDataset(Dataset):
    """Dataset that yields episodes for few-shot learning."""
    
    def __init__(self, base_dataset: RareCancerDataset, episode_sampler: EpisodeSampler,
                 n_episodes: int = 1000):
        """
        Args:
            base_dataset: Base dataset with all samples
            episode_sampler: Episode sampler
            n_episodes: Number of episodes to generate
        """
        self.base_dataset = base_dataset
        self.episode_sampler = episode_sampler
        self.n_episodes = n_episodes
    
    def __len__(self):
        return self.n_episodes
    
    def __getitem__(self, idx):
        # Sample a new episode
        support_indices, query_indices = self.episode_sampler.sample_episode()
        
        # Get support samples
        support_samples = [self.base_dataset[i] for i in support_indices]
        query_samples = [self.base_dataset[i] for i in query_indices]
        
        return {
            'support': support_samples,
            'query': query_samples
        }


def collate_episode(batch):
    """Collate function for episode-based training."""
    support_batch = batch[0]['support']
    query_batch = batch[0]['query']
    
    # Stack support images and labels
    support_images = torch.stack([s['image'] for s in support_batch])
    support_labels = torch.tensor([s['label'] for s in support_batch])
    
    # Stack query images and labels
    query_images = torch.stack([q['image'] for q in query_batch])
    query_labels = torch.tensor([q['label'] for q in query_batch])
    
    # Prepare clinical data
    support_clinical = None
    query_clinical = None
    
    if support_batch[0]['clinical_data'] is not None:
        support_clinical = {
            'gender': torch.tensor([s['clinical_data']['gender'] for s in support_batch]),
            'stage': torch.tensor([s['clinical_data']['stage'] for s in support_batch]),
            'mutation': torch.tensor([s['clinical_data']['mutation'] for s in support_batch]),
            'age': torch.tensor([[s['clinical_data']['age']] for s in support_batch], dtype=torch.float32),
            'tumor_size': torch.tensor([[s['clinical_data']['tumor_size']] for s in support_batch], dtype=torch.float32)
        }
        
        query_clinical = {
            'gender': torch.tensor([q['clinical_data']['gender'] for q in query_batch]),
            'stage': torch.tensor([q['clinical_data']['stage'] for q in query_batch]),
            'mutation': torch.tensor([q['clinical_data']['mutation'] for q in query_batch]),
            'age': torch.tensor([[q['clinical_data']['age']] for q in query_batch], dtype=torch.float32),
            'tumor_size': torch.tensor([[q['clinical_data']['tumor_size']] for q in query_batch], dtype=torch.float32)
        }
    
    return {
        'support_images': support_images,
        'support_labels': support_labels,
        'query_images': query_images,
        'query_labels': query_labels,
        'support_clinical': support_clinical,
        'query_clinical': query_clinical
    }

