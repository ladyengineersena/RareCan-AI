"""
Training script for Prototypical Network few-shot learning.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.dataset import RareCancerDataset, EpisodeDataset, EpisodeSampler, collate_episode
from models.protonet import PrototypicalNetwork
from src.utils import set_seed, save_checkpoint, print_metrics, compute_metrics
import numpy as np


def train_epoch(model, dataloader, optimizer, criterion, device, use_clinical=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_episodes = 0
    
    for batch_idx, episode in enumerate(dataloader):
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
        optimizer.zero_grad()
        
        # Encode support set
        support_embeddings = model(support_images, support_clinical)
        
        # Encode query set
        query_embeddings = model(query_images, query_clinical)
        
        # Compute prototypes
        prototypes = model.compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances and predictions
        logits = model.compute_distances(query_embeddings, prototypes)
        
        # Remap labels to 0..n_way-1 for loss computation
        unique_labels = support_labels.unique().sort()[0]
        label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
        query_labels_mapped = torch.tensor([label_map[label.item()] for label in query_labels], 
                                          device=device)
        
        # Compute loss
        loss = criterion(logits, query_labels_mapped)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_episodes += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Episode {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / n_episodes
    return avg_loss


def evaluate(model, dataloader, device, use_clinical=False):
    """Evaluate model on episodes."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_episodes = 0
    
    criterion = nn.CrossEntropyLoss()
    
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
            
            # Remap labels
            unique_labels = support_labels.unique().sort()[0]
            label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
            query_labels_mapped = torch.tensor([label_map[label.item()] for label in query_labels], 
                                              device=device)
            
            # Compute loss
            loss = criterion(logits, query_labels_mapped)
            total_loss += loss.item()
            
            # Get predictions
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = query_labels_mapped.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            n_episodes += 1
    
    avg_loss = total_loss / n_episodes
    
    # Compute accuracy
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = avg_loss
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Prototypical Network for Rare Cancer Classification')
    parser.add_argument('--data_dir', type=str, default='data/sample',
                       help='Directory containing data')
    parser.add_argument('--metadata_path', type=str, default='data/sample/metadata.json',
                       help='Path to metadata JSON file')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--n_episodes_train', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--n_episodes_val', type=int, default=100, help='Number of validation episodes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (episodes)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50'],
                       help='Backbone model')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--use_clinical', action='store_true', help='Use clinical data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = RareCancerDataset(
        args.metadata_path, args.data_dir, 
        transform=train_transform, use_clinical=args.use_clinical
    )
    
    val_dataset = RareCancerDataset(
        args.metadata_path, args.data_dir,
        transform=val_transform, use_clinical=args.use_clinical
    )
    
    # Create episode samplers
    train_sampler = EpisodeSampler(train_dataset, args.n_way, args.k_shot, args.n_query)
    val_sampler = EpisodeSampler(val_dataset, args.n_way, args.k_shot, args.n_query)
    
    # Create episode datasets
    train_episode_dataset = EpisodeDataset(train_dataset, train_sampler, args.n_episodes_train)
    val_episode_dataset = EpisodeDataset(val_dataset, val_sampler, args.n_episodes_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_episode_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_episode, num_workers=0
    )
    
    val_loader = DataLoader(
        val_episode_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_episode, num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = PrototypicalNetwork(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        use_clinical=args.use_clinical
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.use_clinical)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, args.use_clinical)
        print_metrics(val_metrics, "Validation")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
            print(f"New best model saved! Validation Accuracy: {best_val_acc:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()

