"""
Evaluation script for Prototypical Network.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.dataset import RareCancerDataset, EpisodeDataset, EpisodeSampler, collate_episode
from models.protonet import PrototypicalNetwork
from src.utils import set_seed, compute_metrics, print_metrics, plot_confusion_matrix, load_checkpoint


def evaluate_model(model, dataloader, device, use_clinical=False, class_names=None):
    """Evaluate model on episodes."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for episode in dataloader:
            # Move to device
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)
            
            support_clinical = None
            query_clinical = None
            if use_clinical:
                support_clinical = {
                    k: v.to(device) for k, v in episode['support_clinical'].items()
                }
                query_clinical = {
                    k: v.to(device) for k, v in episode['query_clinical'].items()
                }
            
            # Forward pass
            support_embeddings = model(support_images, support_clinical)
            query_embeddings = model(query_images, query_clinical)
            
            # Compute prototypes
            prototypes = model.compute_prototypes(support_embeddings, support_labels)
            
            # Compute predictions
            logits = model.compute_distances(query_embeddings, prototypes)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Remap labels
            unique_labels = support_labels.unique().sort()[0]
            label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
            query_labels_mapped = torch.tensor([label_map[label.item()] for label in query_labels], 
                                              device=device)
            
            # Get predictions
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = query_labels_mapped.cpu().numpy()
            prob_np = probs.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(prob_np)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_labels, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description='Evaluate Prototypical Network')
    parser.add_argument('--data_dir', type=str, default='data/sample',
                       help='Directory containing data')
    parser.add_argument('--metadata_path', type=str, default='data/sample/metadata.json',
                       help='Path to metadata JSON file')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--n_episodes', type=int, default=200, help='Number of evaluation episodes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50'],
                       help='Backbone model')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--use_clinical', action='store_true', help='Use clinical data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Loading dataset...")
    dataset = RareCancerDataset(
        args.metadata_path, args.data_dir,
        transform=transform, use_clinical=args.use_clinical
    )
    
    # Create episode sampler
    sampler = EpisodeSampler(dataset, args.n_way, args.k_shot, args.n_query)
    
    # Create episode dataset
    episode_dataset = EpisodeDataset(dataset, sampler, args.n_episodes)
    
    # Create dataloader
    dataloader = DataLoader(
        episode_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_episode, num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = PrototypicalNetwork(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        use_clinical=args.use_clinical
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        load_checkpoint(args.checkpoint_path, model)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}. Using untrained model.")
    
    # Get class names
    class_names = [dataset.idx_to_class[i] for i in range(len(dataset.idx_to_class))]
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, all_labels, all_preds, all_probs = evaluate_model(
        model, dataloader, device, args.use_clinical, class_names
    )
    
    # Print metrics
    print_metrics(metrics, "Final Evaluation")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, 
        class_names=[f"Class {i}" for i in range(args.n_way)],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {results_path}")
    print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")


if __name__ == '__main__':
    main()

