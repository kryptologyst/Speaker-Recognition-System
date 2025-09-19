"""
Tests for audio processing functionality
"""

import pytest
import numpy as np
from data.audio_processor import AudioProcessor

def test_audio_processor_initialization():
    """Test AudioProcessor initialization"""
    processor = AudioProcessor()
    
    assert processor.sample_rate == 16000
    assert processor.n_mfcc == 13
    assert processor.n_mels == 80

def test_preemphasis():
    """Test pre-emphasis filter"""
    processor = AudioProcessor()
    
    # Create test signal
    audio = np.random.randn(1000)
    
    # Apply pre-emphasis
    processed = processor.preemphasis(audio)
    
    assert len(processed) == len(audio)
    assert not np.array_equal(audio, processed)

def test_feature_extraction():
    """Test MFCC feature extraction"""
    processor = AudioProcessor()
    
    # Create test audio (1 second at 16kHz)
    audio = np.random.randn(16000)
    
    # Extract MFCC features
    mfcc = processor.extract_mfcc(audio)
    
    assert mfcc.shape[0] == processor.n_mfcc
    assert mfcc.shape[1] > 0  # Should have time frames
