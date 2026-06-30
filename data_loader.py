import random
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset


class BERT4RecDataProcessor:
    """Data processor for BERT4Rec"""
    def __init__(self, data_path, max_len=50):
        self.max_len = max_len
        self.data_path = data_path
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """Load and preprocess the dataset"""
        print('Loading data...')
        
        # Check for ml-1m.txt format (user item pairs)
        ml1m_file = os.path.join(self.data_path, 'ml-1m.txt')
        if os.path.exists(ml1m_file):
            # Load simple user-item pairs format
            data = []
            with open(ml1m_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        item_id = int(parts[1])
                        data.append({'userId': user_id, 'movieId': item_id, 'rating': 5, 'timestamp': 0})
            
            self.df = pd.DataFrame(data)
            print(f"Loaded {len(self.df)} interactions from ml-1m.txt")
        else:
            # Try to load MovieLens format data
            try:
                self.df = pd.read_csv(os.path.join(self.data_path, 'ratings.dat'), 
                                     sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], 
                                     engine='python')
            except:
                try:
                    self.df = pd.read_csv(os.path.join(self.data_path, 'ratings.csv'))
                except:
                    # If no specific file, try to find any rating file
                    files = os.listdir(self.data_path)
                    rating_files = [f for f in files if 'rating' in f.lower()]
                    if rating_files:
                        self.df = pd.read_csv(os.path.join(self.data_path, rating_files[0]))
                    else:
                        raise FileNotFoundError("No rating file found in data directory")
    
    def _preprocess_data(self):
        """Preprocess the data for BERT4Rec"""
        print('Preprocessing data...')
        
        # Generate item and user encoders
        self.item_encoder, self.item_decoder = self._generate_encoder_decoder(self.df['movieId'])
        self.user_encoder, self.user_decoder = self._generate_encoder_decoder(self.df['userId'])
        
        self.num_items = len(self.item_encoder)
        self.num_users = len(self.user_encoder)
        
        # Map to indices
        self.df['item_idx'] = self.df['movieId'].apply(lambda x: self.item_encoder[x] + 1)  # +1 for padding
        self.df['user_idx'] = self.df['userId'].apply(lambda x: self.user_encoder[x])
        
        # Sort by user and timestamp
        self.df = self.df.sort_values(['user_idx', 'timestamp'])
        
        # Generate sequences
        self.user_sequences = self._generate_sequences()
        
        print(f'Data preprocessing complete. Users: {self.num_users}, Items: {self.num_items}')
    
    def _generate_encoder_decoder(self, col):
        """Generate encoder and decoder dictionaries"""
        encoder = {}
        decoder = {}
        ids = col.unique()
        
        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id
        
        return encoder, decoder
    
    def _generate_sequences(self):
        """Generate user sequences for training"""
        sequences = defaultdict(list)
        group_df = self.df.groupby('user_idx')
        
        for user, group in group_df:
            sequences[user] = group['item_idx'].tolist()
        
        return sequences
    
    def get_train_test_split(self, test_ratio=0.2):
        """Split data into train and test sets"""
        train_sequences = {}
        test_sequences = {}
        
        for user, sequence in self.user_sequences.items():
            if len(sequence) <= 1:
                continue
                
            split_idx = max(1, int(len(sequence) * (1 - test_ratio)))
            train_sequences[user] = sequence[:split_idx]
            test_sequences[user] = sequence[split_idx:]
        
        return train_sequences, test_sequences


class BERT4RecDataset(Dataset):
    """Dataset for BERT4Rec training"""
    def __init__(self, user_sequences, num_items, max_len=50, mask_prob=0.15):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.users = list(user_sequences.keys())
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        sequence = self.user_sequences[user]
        
        # Truncate or pad sequence
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
        
        # Apply BERT-style masking
        tokens = []
        labels = []
        
        for item in sequence:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # Mask token (num_items + 1)
                    tokens.append(self.num_items + 1)
                elif prob < 0.9:
                    # Random token
                    tokens.append(random.randint(1, self.num_items))
                else:
                    # Original token
                    tokens.append(item)
            else:
                tokens.append(item)
            labels.append(item)
        
        # Pad sequence
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens  # 0 is padding token
        labels = [0] * mask_len + labels
        
        return torch.LongTensor(tokens), torch.LongTensor(labels)