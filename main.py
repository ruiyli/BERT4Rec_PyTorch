import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from model import BERT4Rec
from data_loader import BERT4RecDataProcessor, BERT4RecDataset
from utils import evaluate


def train_model(args):
    """Train BERT4Rec model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    print('Loading data...')
    data_processor = BERT4RecDataProcessor(args.data_path, max_len=args.max_len)
    train_sequences, test_sequences = data_processor.get_train_test_split(test_ratio=0.2)
    
    # Create datasets
    train_dataset = BERT4RecDataset(train_sequences, data_processor.num_items, 
                                   max_len=args.max_len, mask_prob=args.mask_prob)
    test_dataset = BERT4RecDataset(test_sequences, data_processor.num_items, 
                                  max_len=args.max_len, mask_prob=0.0)  # No masking for test
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = BERT4Rec(
        bert_max_len=args.max_len,
        num_items=data_processor.num_items,
        bert_num_blocks=args.num_blocks,
        bert_num_heads=args.num_heads,
        bert_hidden_units=args.hidden_units,
        bert_dropout=args.dropout
    ).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Training loop
    best_hr = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (tokens, labels) in enumerate(train_loader):
            tokens, labels = tokens.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(tokens)
            
            # Reshape for cross entropy loss
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Evaluate
        if epoch % args.eval_every == 0:
            hr, ndcg = evaluate(model, test_loader, device, k=args.top_k)
            print(f'Evaluation - HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg:.4f}')
            
            # Save best model
            if hr > best_hr:
                best_hr = hr
                torch.save(model.state_dict(), f'best_model_bert4rec_{args.dataset}.pth')
                print(f'New best model saved with HR@{args.top_k}: {hr:.4f}')


def test_model(args):
    """Test trained BERT4Rec model"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Load data
    data_processor = BERT4RecDataProcessor(args.data_path, max_len=args.max_len)
    _, test_sequences = data_processor.get_train_test_split(test_ratio=0.2)
    
    test_dataset = BERT4RecDataset(test_sequences, data_processor.num_items, 
                                  max_len=args.max_len, mask_prob=0.0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    model = BERT4Rec(
        bert_max_len=args.max_len,
        num_items=data_processor.num_items,
        bert_num_blocks=args.num_blocks,
        bert_num_heads=args.num_heads,
        bert_hidden_units=args.hidden_units,
        bert_dropout=args.dropout
    ).to(device)
    
    model_path = f'best_model_bert4rec_{args.dataset}.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Model loaded from {model_path}')
    else:
        print(f'No trained model found at {model_path}')
        return
    
    # Evaluate
    hr, ndcg = evaluate(model, test_loader, device, k=args.top_k)
    print(f'Final Test Results:')
    print(f'HR@{args.top_k}: {hr:.4f}')
    print(f'NDCG@{args.top_k}: {ndcg:.4f}')


def main():
    parser = argparse.ArgumentParser(description='BERT4Rec PyTorch Implementation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data', 
                       help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='ml-1m', 
                       choices=['ml-1m', 'ml-20m', 'beauty', 'steam'],
                       help='Dataset name')
    
    # Model parameters
    parser.add_argument('--max_len', type=int, default=50, 
                       help='Maximum sequence length')
    parser.add_argument('--num_blocks', type=int, default=2, 
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=2, 
                       help='Number of attention heads')
    parser.add_argument('--hidden_units', type=int, default=256, 
                       help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate')
    parser.add_argument('--mask_prob', type=float, default=0.15, 
                       help='Mask probability for BERT training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='Number of epochs')
    parser.add_argument('--eval_every', type=int, default=10, 
                       help='Evaluate every N epochs')
    parser.add_argument('--top_k', type=int, default=10, 
                       help='Top K for evaluation metrics')
    parser.add_argument('--no_cuda', action='store_true', 
                       help='Disable CUDA')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'],
                       help='Mode: train or test')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    else:
        test_model(args)


if __name__ == '__main__':
    main()