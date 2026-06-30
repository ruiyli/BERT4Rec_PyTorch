import torch
import numpy as np
from collections import defaultdict


def evaluate(model, data_loader, device, k=10):
    """Evaluate model using HR@K and NDCG@K metrics"""
    model.eval()
    
    all_hr = []
    all_ndcg = []
    
    with torch.no_grad():
        for tokens, labels in data_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(tokens)
            
            # Get the last position prediction (next item prediction)
            batch_size, seq_len, num_items = outputs.shape
            
            # Focus on the last non-padded position
            for i in range(batch_size):
                # Find the last non-zero label
                non_zero_indices = torch.nonzero(labels[i], as_tuple=False)
                if len(non_zero_indices) == 0:
                    continue
                    
                last_idx = non_zero_indices[-1].item()
                true_item = labels[i, last_idx].item()
                
                if true_item == 0:  # Skip padding
                    continue
                
                # Get predictions for the last position
                pred_scores = outputs[i, last_idx]
                
                # Get top-k predictions
                _, topk_indices = torch.topk(pred_scores, k)
                topk_items = topk_indices.cpu().numpy()
                
                # Calculate HR@K
                hr = 1 if true_item in topk_items else 0
                all_hr.append(hr)
                
                # Calculate NDCG@K
                ndcg = 0
                for rank, item in enumerate(topk_items, 1):
                    if item == true_item:
                        ndcg = 1 / np.log2(rank + 1)
                        break
                all_ndcg.append(ndcg)
    
    avg_hr = np.mean(all_hr) if all_hr else 0
    avg_ndcg = np.mean(all_ndcg) if all_ndcg else 0
    
    return avg_hr, avg_ndcg


def get_metrics(predicted, actual, k=10):
    """Calculate HR@K and NDCG@K for a single prediction"""
    # Get top-k predicted items
    topk_pred = predicted.argsort()[-k:][::-1]
    
    # Hit Ratio
    hr = 1 if actual in topk_pred else 0
    
    # NDCG
    ndcg = 0
    for i, item in enumerate(topk_pred):
        if item == actual:
            ndcg = 1 / np.log2(i + 2)  # i+2 because rank starts from 1
            break
    
    return hr, ndcg


def calculate_metrics_all_users(model, test_sequences, num_items, max_len, device, k=10):
    """Calculate metrics for all users in test set"""
    model.eval()
    
    all_hr = []
    all_ndcg = []
    
    with torch.no_grad():
        for user, sequence in test_sequences.items():
            if len(sequence) <= 1:
                continue
            
            # Use all but the last item as input
            input_sequence = sequence[:-1]
            true_item = sequence[-1]
            
            # Pad sequence
            if len(input_sequence) < max_len:
                input_sequence = [0] * (max_len - len(input_sequence)) + input_sequence
            else:
                input_sequence = input_sequence[-max_len:]
            
            # Convert to tensor
            input_tensor = torch.LongTensor([input_sequence]).to(device)
            
            # Get prediction
            output = model(input_tensor)
            
            # Get prediction for the last position
            pred_scores = output[0, -1].cpu().numpy()
            
            # Calculate metrics
            hr, ndcg = get_metrics(pred_scores, true_item, k)
            all_hr.append(hr)
            all_ndcg.append(ndcg)
    
    avg_hr = np.mean(all_hr) if all_hr else 0
    avg_ndcg = np.mean(all_ndcg) if all_ndcg else 0
    
    return avg_hr, avg_ndcg


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False