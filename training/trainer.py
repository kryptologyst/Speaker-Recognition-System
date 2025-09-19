"""
Advanced Training Pipeline for Speaker Recognition Models
Implements modern training techniques with PyTorch Lightning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime

from models.deep_speaker_model import get_model
from data.audio_processor import AudioProcessor
from data.database import SpeakerDatabase


class SpeakerDataset(Dataset):
    """PyTorch Dataset for speaker recognition"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 feature_type: str = "mfcc", transform=None):
        self.features = features
        self.labels = labels
        self.feature_type = feature_type
        self.transform = transform
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.encoded_labels[idx]
        
        # Convert to tensor
        if self.feature_type == "mfcc":
            # MFCC features: (n_mfcc, time_steps)
            feature = torch.FloatTensor(feature)
        elif self.feature_type == "mel_spectrogram":
            # Mel spectrogram: (n_mels, time_steps)
            feature = torch.FloatTensor(feature)
        
        # Apply transforms if any
        if self.transform:
            feature = self.transform(feature)
        
        return feature, torch.LongTensor([label])[0]


class SpeakerRecognitionModule(pl.LightningModule):
    """PyTorch Lightning module for speaker recognition"""
    
    def __init__(self, model_type: str, num_classes: int, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, **model_kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = get_model(model_type, num_classes, **model_kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Metrics tracking
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'preds': preds, 'targets': y}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        return {'test_loss': loss, 'test_acc': acc, 'preds': preds, 'targets': y}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class SpeakerTrainer:
    """High-level trainer for speaker recognition models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db = SpeakerDatabase(config.get('db_path', 'speaker_recognition.db'))
        self.audio_processor = AudioProcessor()
        
        # Create directories
        os.makedirs(config.get('model_dir', 'checkpoints'), exist_ok=True)
        os.makedirs(config.get('log_dir', 'logs'), exist_ok=True)
    
    def prepare_data(self, feature_type: str = "mfcc") -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data loaders"""
        
        # Get data from database
        features, labels, file_ids = self.db.get_training_data(feature_type)
        
        if len(features) == 0:
            raise ValueError("No training data found in database")
        
        # Create dataset
        dataset = SpeakerDataset(features, labels, feature_type)
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=True
        )
        
        return train_loader, val_loader, test_loader, dataset.num_classes, dataset.label_encoder
    
    def train_model(self, model_type: str, feature_type: str = "mfcc") -> Dict:
        """Train a speaker recognition model"""
        
        print(f"Training {model_type} model with {feature_type} features...")
        
        # Prepare data
        train_loader, val_loader, test_loader, num_classes, label_encoder = self.prepare_data(feature_type)
        
        # Create model
        model_kwargs = self.config.get('model_params', {})
        model = SpeakerRecognitionModule(
            model_type=model_type,
            num_classes=num_classes,
            learning_rate=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4),
            **model_kwargs
        )
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=self.config.get('model_dir', 'checkpoints'),
                filename=f'{model_type}_{feature_type}_{{epoch:02d}}_{{val_acc:.3f}}',
                monitor='val_acc',
                mode='max',
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min',
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=self.config.get('log_dir', 'logs'),
            name=f'{model_type}_{feature_type}',
            version=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.get('max_epochs', 100),
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices='auto',
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            deterministic=True
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Test model
        test_results = trainer.test(model, test_loader)
        
        # Save model metadata to database
        model_path = trainer.checkpoint_callback.best_model_path
        accuracy = test_results[0]['test_acc']
        
        model_id = self.db.save_model(
            name=f"{model_type}_{feature_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=model_type,
            architecture=str(model.model),
            parameters=self.config,
            accuracy=accuracy,
            model_path=model_path
        )
        
        # Save label encoder
        label_encoder_path = os.path.join(
            self.config.get('model_dir', 'checkpoints'),
            f'label_encoder_{model_type}_{feature_type}.pkl'
        )
        
        import pickle
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        return {
            'model_id': model_id,
            'model_path': model_path,
            'label_encoder_path': label_encoder_path,
            'accuracy': accuracy,
            'num_classes': num_classes,
            'test_results': test_results[0]
        }
    
    def evaluate_model(self, model_path: str, label_encoder_path: str, 
                      feature_type: str = "mfcc") -> Dict:
        """Evaluate a trained model"""
        
        # Load model
        model = SpeakerRecognitionModule.load_from_checkpoint(model_path)
        model.eval()
        
        # Load label encoder
        import pickle
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Prepare test data
        _, _, test_loader, _, _ = self.prepare_data(feature_type)
        
        # Evaluate
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Convert back to original labels
        pred_labels = label_encoder.inverse_transform(all_preds)
        true_labels = label_encoder.inverse_transform(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   cmap='Blues')
        plt.title(f'Confusion Matrix - Accuracy: {accuracy:.3f}')
        plt.xlabel('Predicted Speaker')
        plt.ylabel('True Speaker')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config.get('model_dir', 'checkpoints'),
            f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'plot_path': plot_path,
            'predictions': pred_labels,
            'true_labels': true_labels
        }
    
    def compare_models(self, model_configs: List[Dict]) -> pd.DataFrame:
        """Compare multiple model configurations"""
        import pandas as pd
        
        results = []
        
        for config in model_configs:
            model_type = config['model_type']
            feature_type = config.get('feature_type', 'mfcc')
            
            print(f"\nTraining {model_type} with {feature_type}...")
            
            # Update trainer config
            self.config.update(config.get('training_params', {}))
            
            try:
                result = self.train_model(model_type, feature_type)
                
                results.append({
                    'model_type': model_type,
                    'feature_type': feature_type,
                    'accuracy': result['accuracy'],
                    'model_path': result['model_path'],
                    'num_classes': result['num_classes']
                })
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
                results.append({
                    'model_type': model_type,
                    'feature_type': feature_type,
                    'accuracy': 0.0,
                    'model_path': None,
                    'num_classes': 0,
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(
            self.config.get('model_dir', 'checkpoints'),
            f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        results_df.to_csv(results_path, index=False)
        
        print(f"\nModel comparison results saved to: {results_path}")
        print(results_df)
        
        return results_df


def create_training_config():
    """Create default training configuration"""
    return {
        'db_path': 'speaker_recognition.db',
        'model_dir': 'checkpoints',
        'log_dir': 'logs',
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'max_epochs': 50,
        'model_params': {
            'input_dim': 80  # For mel spectrogram
        }
    }


if __name__ == "__main__":
    # Create training configuration
    config = create_training_config()
    
    # Initialize trainer
    trainer = SpeakerTrainer(config)
    
    # Define model configurations to compare
    model_configs = [
        {
            'model_type': 'cnn',
            'feature_type': 'mfcc',
            'training_params': {'learning_rate': 1e-3}
        },
        {
            'model_type': 'cnn',
            'feature_type': 'mel_spectrogram',
            'training_params': {'learning_rate': 1e-3}
        },
        {
            'model_type': 'transformer',
            'feature_type': 'mfcc',
            'training_params': {'learning_rate': 5e-4}
        }
    ]
    
    # Compare models
    results = trainer.compare_models(model_configs)
    
    print("\nTraining completed! Check the results above.")
