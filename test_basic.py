#!/usr/bin/env python3
"""Basic test script for BERT4Rec model"""

import torch
import numpy as np
from model import BERT4Rec
from data_loader import BERT4RecDataProcessor
from utils import set_seed


def test_model_creation():
    """Test model creation and forward pass"""
    print("Testing model creation...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create a small model
    model = BERT4Rec(
        bert_max_len=10,
        num_items=100,
        bert_num_blocks=2,
        bert_num_heads=2,
        bert_hidden_units=64,
        bert_dropout=0.1
    )
    
    print(f"Model created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    
    # Create dummy input (batch_size, seq_len)
    # 0: padding, 1-100: items, 101: mask token
    dummy_input = torch.randint(0, 102, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Model forward pass successful")
    
    return model


def test_data_processor():
    """Test data processor functionality"""
    print("\nTesting data processor...")
    
    # Create a simple test DataFrame
    import pandas as pd
    from io import StringIO
    
    # Create test data
    test_data = """userId,movieId,rating,timestamp
1,101,5,964982703
1,102,3,964982713
2,101,4,964982723
2,103,5,964982733
1,104,4,964982743
2,102,3,964982753"""
    
    df = pd.read_csv(StringIO(test_data))
    
    # Test basic functionality
    processor = BERT4RecDataProcessor.__new__(BERT4RecDataProcessor)
    processor.df = df
    processor.max_len = 5
    
    # Test encoder generation
    encoder, decoder = processor._generate_encoder_decoder(df['movieId'])
    print(f"Item encoder: {encoder}")
    print(f"Item decoder: {decoder}")
    print("✓ Data processor basic functionality successful")


def test_evaluation():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    
    from utils import get_metrics
    
    # Create dummy predictions and actual values
    predicted_scores = np.random.rand(100)  # 100 items
    actual_item = 42
    
    hr, ndcg = get_metrics(predicted_scores, actual_item, k=10)
    
    print(f"HR@10: {hr}")
    print(f"NDCG@10: {ndcg}")
    print("✓ Evaluation metrics calculation successful")


def main():
    """Run all tests"""
    print("Running BERT4Rec basic tests...\n")
    
    try:
        # Test model creation
        model = test_model_creation()
        
        # Test data processor
        test_data_processor()
        
        # Test evaluation
        test_evaluation()
        
        print("\n🎉 All basic tests passed successfully!")
        print("\nNext steps:")
        print("1. Place your dataset in the 'data/' directory")
        print("2. Run: python3 main.py --mode train --dataset your_dataset")
        print("3. Run: python3 main.py --mode test --dataset your_dataset")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()