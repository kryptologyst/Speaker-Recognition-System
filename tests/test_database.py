"""
Tests for database functionality
"""

import pytest
import tempfile
import os
from data.database import SpeakerDatabase

def test_speaker_database_creation():
    """Test database creation and speaker addition"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = SpeakerDatabase(db_path)
        
        # Add a test speaker
        speaker_id = db.add_speaker("Test Speaker", gender="male", age=30)
        assert speaker_id > 0
        
        # Verify speaker was added
        speaker_data = db.get_speaker_data(speaker_id=speaker_id)
        assert speaker_data['name'] == "Test Speaker"
        assert speaker_data['gender'] == "male"
        assert speaker_data['age'] == 30
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_audio_file_management():
    """Test audio file addition and retrieval"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = SpeakerDatabase(db_path)
        
        # Add speaker first
        speaker_id = db.add_speaker("Test Speaker")
        
        # Add audio file (mock)
        audio_id = db.add_audio_file(
            speaker_id=speaker_id,
            file_path="test_audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        assert audio_id > 0
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
