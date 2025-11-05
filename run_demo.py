"""
Demo script to quickly test the RareCan-AI project.
This script generates synthetic data and runs a quick training test.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_generator import SyntheticHistopathologyGenerator
from src.dataset import RareCancerDataset, EpisodeSampler
from models.protonet import PrototypicalNetwork
from src.utils import set_seed
import torch
from torchvision import transforms

def main():
    print("=" * 60)
    print("RareCan-AI: Demo Script")
    print("=" * 60)
    
    # Set seed
    set_seed(42)
    
    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic data...")
    generator = SyntheticHistopathologyGenerator(output_dir='data/sample', seed=42)
    metadata = generator.generate_dataset(samples_per_class=50)
    print("✓ Synthetic data generated successfully!")
    
    # Step 2: Create dataset
    print("\n[2/4] Creating dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = RareCancerDataset(
        metadata_path='data/sample/metadata.json',
        image_dir='data/sample',
        transform=transform,
        use_clinical=True
    )
    print(f"✓ Dataset created with {len(dataset)} samples from {len(dataset.class_to_idx)} classes")
    
    # Step 3: Test episode sampler
    print("\n[3/4] Testing episode sampler...")
    sampler = EpisodeSampler(dataset, n_way=5, k_shot=5, n_query=15)
    support_indices, query_indices = sampler.sample_episode()
    print(f"✓ Episode sampled: {len(support_indices)} support, {len(query_indices)} query samples")
    
    # Step 4: Create and test model
    print("\n[4/4] Creating and testing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PrototypicalNetwork(
        backbone='resnet50',
        embedding_dim=128,
        use_clinical=True
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count:,} parameters")
    
    # Test forward pass
    model.eval()
    support_samples = [dataset[i] for i in support_indices[:10]]  # Use fewer for quick test
    query_samples = [dataset[i] for i in query_indices[:10]]
    
    support_images = torch.stack([s['image'] for s in support_samples]).to(device)
    support_labels = torch.tensor([s['label'] for s in support_samples]).to(device)
    query_images = torch.stack([q['image'] for q in query_samples]).to(device)
    
    support_clinical = {
        'gender': torch.tensor([s['clinical_data']['gender'] for s in support_samples]).to(device),
        'stage': torch.tensor([s['clinical_data']['stage'] for s in support_samples]).to(device),
        'mutation': torch.tensor([s['clinical_data']['mutation'] for s in support_samples]).to(device),
        'age': torch.tensor([[s['clinical_data']['age']] for s in support_samples], dtype=torch.float32).to(device),
        'tumor_size': torch.tensor([[s['clinical_data']['tumor_size']] for s in support_samples], dtype=torch.float32).to(device)
    }
    
    query_clinical = {
        'gender': torch.tensor([q['clinical_data']['gender'] for q in query_samples]).to(device),
        'stage': torch.tensor([q['clinical_data']['stage'] for q in query_samples]).to(device),
        'mutation': torch.tensor([q['clinical_data']['mutation'] for q in query_samples]).to(device),
        'age': torch.tensor([[q['clinical_data']['age']] for q in query_samples], dtype=torch.float32).to(device),
        'tumor_size': torch.tensor([[q['clinical_data']['tumor_size']] for q in query_samples], dtype=torch.float32).to(device)
    }
    
    with torch.no_grad():
        support_embeddings = model(support_images, support_clinical)
        query_embeddings = model(query_images, query_clinical)
        prototypes = model.compute_prototypes(support_embeddings, support_labels)
        logits = model.compute_distances(query_embeddings, prototypes)
    
    print(f"✓ Forward pass successful! Output shape: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train the model: python src/train.py --data_dir data/sample --metadata_path data/sample/metadata.json --use_clinical")
    print("2. Evaluate the model: python src/evaluate.py --checkpoint_path checkpoints/best_model.pt --data_dir data/sample --metadata_path data/sample/metadata.json")
    print("3. Explore the notebook: notebooks/01_experiments.ipynb")

if __name__ == '__main__':
    main()

