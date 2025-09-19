"""
Modern Deep Learning Speaker Recognition Models
Implements state-of-the-art architectures for speaker identification and verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Config
from speechbrain.pretrained import SpeakerRecognition


class ResNetBlock(nn.Module):
    """Residual block for deep speaker embeddings"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DeepSpeakerCNN(nn.Module):
    """Deep CNN for speaker recognition with residual connections"""
    def __init__(self, num_speakers, input_dim=80):
        super(DeepSpeakerCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_speakers)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class TransformerSpeakerModel(nn.Module):
    """Transformer-based speaker recognition model"""
    def __init__(self, num_speakers, d_model=512, nhead=8, num_layers=6):
        super(TransformerSpeakerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(80, d_model)  # Project MFCC to model dimension
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_speakers)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class Wav2VecSpeakerModel(nn.Module):
    """Wav2Vec2-based speaker recognition model"""
    def __init__(self, num_speakers, pretrained_model="facebook/wav2vec2-base"):
        super(Wav2VecSpeakerModel, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        # Freeze wav2vec2 parameters (optional)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # Speaker classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),  # wav2vec2-base has 768 hidden size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_speakers)
        )
    
    def forward(self, input_values):
        # Extract features from wav2vec2
        outputs = self.wav2vec2(input_values)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits


class SpeakerEmbeddingModel(nn.Module):
    """Model for generating speaker embeddings"""
    def __init__(self, embedding_dim=256):
        super(SpeakerEmbeddingModel, self).__init__()
        
        self.backbone = DeepSpeakerCNN(num_speakers=embedding_dim, input_dim=80)
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def get_model(model_type, num_speakers, **kwargs):
    """Factory function to get different model types"""
    if model_type == "cnn":
        return DeepSpeakerCNN(num_speakers, **kwargs)
    elif model_type == "transformer":
        return TransformerSpeakerModel(num_speakers, **kwargs)
    elif model_type == "wav2vec":
        return Wav2VecSpeakerModel(num_speakers, **kwargs)
    elif model_type == "embedding":
        return SpeakerEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
