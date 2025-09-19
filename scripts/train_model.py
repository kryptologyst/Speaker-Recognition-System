#!/usr/bin/env python3
"""
Training script for speaker recognition models
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import SpeakerTrainer, create_training_config

def main():
    parser = argparse.ArgumentParser(description="Train speaker recognition model")
    parser.add_argument("--model-type", choices=["cnn", "transformer", "wav2vec"], 
                       default="cnn", help="Model architecture")
    parser.add_argument("--feature-type", choices=["mfcc", "mel_spectrogram"], 
                       default="mfcc", help="Feature type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_training_config()
    config.update({
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    # Initialize trainer
    trainer = SpeakerTrainer(config)
    
    # Train model
    print(f"Training {args.model_type} model with {args.feature_type} features...")
    result = trainer.train_model(args.model_type, args.feature_type)
    
    print(f"Training completed!")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Model saved to: {result['model_path']}")

if __name__ == "__main__":
    main()
