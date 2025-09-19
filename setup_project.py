#!/usr/bin/env python3
"""
Project Setup Script for Speaker Recognition System
Initializes the project with sample data, creates directories, and sets up the environment
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_directories():
    """Create necessary project directories"""
    directories = [
        'checkpoints',
        'logs',
        'audio_data',
        'audio_data/samples',
        'tests',
        'docs',
        'notebooks',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies(dev=False):
    """Install project dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Install main requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Main dependencies installed")
        
        if dev:
            # Install development dependencies
            dev_packages = [
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0", 
                "black>=22.0.0",
                "flake8>=5.0.0",
                "pre-commit>=2.20.0",
                "jupyter>=1.0.0",
                "ipykernel>=6.0.0"
            ]
            
            for package in dev_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            print("✅ Development dependencies installed")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True

def setup_database():
    """Initialize the database with sample data"""
    print("🗄️ Setting up database...")
    
    try:
        from data.database import create_sample_database
        db = create_sample_database()
        print("✅ Database initialized with sample data")
        return True
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def create_config_files():
    """Create configuration files"""
    
    # Create .streamlit config
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    config_content = """
[general]
dataFrameSerialization = "legacy"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    with open(streamlit_dir / "config.toml", "w") as f:
        f.write(config_content.strip())
    
    print("✅ Created Streamlit configuration")
    
    # Create environment template
    env_template = """
# Speaker Recognition System Environment Variables

# Database Configuration
DATABASE_PATH=speaker_recognition.db

# Model Configuration
DEFAULT_MODEL_TYPE=cnn
DEFAULT_FEATURE_TYPE=mfcc

# Audio Processing
SAMPLE_RATE=16000
N_MFCC=13
N_MELS=80

# Training Configuration
BATCH_SIZE=32
LEARNING_RATE=0.001
MAX_EPOCHS=50

# Paths
MODEL_DIR=checkpoints
LOG_DIR=logs
AUDIO_DIR=audio_data

# API Configuration (if using FastAPI)
API_HOST=localhost
API_PORT=8000
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template.strip())
    
    print("✅ Created environment template")

def create_test_files():
    """Create basic test structure"""
    
    test_init = """# Test package initialization"""
    
    with open("tests/__init__.py", "w") as f:
        f.write(test_init)
    
    test_database = '''"""
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
'''
    
    with open("tests/test_database.py", "w") as f:
        f.write(test_database)
    
    test_audio = '''"""
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
'''
    
    with open("tests/test_audio_processor.py", "w") as f:
        f.write(test_audio)
    
    print("✅ Created test files")

def create_example_notebook():
    """Create an example Jupyter notebook"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Speaker Recognition System - Getting Started\n",
                    "\n",
                    "This notebook demonstrates the basic usage of the Speaker Recognition System."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import necessary libraries\n",
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "from data.database import SpeakerDatabase\n",
                    "from data.audio_processor import AudioProcessor\n",
                    "from training.trainer import SpeakerTrainer, create_training_config\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Initialize Components"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize database and audio processor\n",
                    "db = SpeakerDatabase('../speaker_recognition.db')\n",
                    "processor = AudioProcessor()\n",
                    "\n",
                    "print(\"Components initialized successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Explore Database"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Get all speakers\n",
                    "speakers_df = db.get_all_speakers()\n",
                    "print(f\"Total speakers: {len(speakers_df)}\")\n",
                    "speakers_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Audio Processing Example"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate sample audio for demonstration\n",
                    "sample_rate = 16000\n",
                    "duration = 3.0\n",
                    "t = np.linspace(0, duration, int(sample_rate * duration))\n",
                    "\n",
                    "# Create a simple sine wave (mock voice)\n",
                    "frequency = 440  # A4 note\n",
                    "audio = np.sin(2 * np.pi * frequency * t) * 0.3\n",
                    "\n",
                    "# Add some noise to make it more realistic\n",
                    "noise = np.random.normal(0, 0.05, len(audio))\n",
                    "audio += noise\n",
                    "\n",
                    "print(f\"Generated audio: {len(audio)} samples, {duration} seconds\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract features\n",
                    "mfcc_features = processor.extract_mfcc(audio)\n",
                    "mel_features = processor.extract_mel_spectrogram(audio)\n",
                    "\n",
                    "print(f\"MFCC shape: {mfcc_features.shape}\")\n",
                    "print(f\"Mel spectrogram shape: {mel_features.shape}\")\n",
                    "\n",
                    "# Visualize features\n",
                    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))\n",
                    "\n",
                    "# Plot MFCC\n",
                    "im1 = ax1.imshow(mfcc_features, aspect='auto', origin='lower')\n",
                    "ax1.set_title('MFCC Features')\n",
                    "ax1.set_ylabel('MFCC Coefficient')\n",
                    "plt.colorbar(im1, ax=ax1)\n",
                    "\n",
                    "# Plot Mel Spectrogram\n",
                    "im2 = ax2.imshow(mel_features, aspect='auto', origin='lower')\n",
                    "ax2.set_title('Mel Spectrogram')\n",
                    "ax2.set_ylabel('Mel Band')\n",
                    "ax2.set_xlabel('Time Frame')\n",
                    "plt.colorbar(im2, ax=ax2)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open("notebooks/getting_started.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Created example notebook")

def create_scripts():
    """Create utility scripts"""
    
    # Training script
    train_script = '''#!/usr/bin/env python3
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
'''
    
    with open("scripts/train_model.py", "w") as f:
        f.write(train_script)
    
    # Make executable
    os.chmod("scripts/train_model.py", 0o755)
    
    # Data preparation script
    data_script = '''#!/usr/bin/env python3
"""
Data preparation script for speaker recognition
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import SpeakerDatabase
from data.audio_processor import AudioProcessor

def process_directory(audio_dir, db_path="speaker_recognition.db"):
    """Process audio files in directory structure"""
    
    db = SpeakerDatabase(db_path)
    processor = AudioProcessor()
    
    audio_dir = Path(audio_dir)
    
    if not audio_dir.exists():
        print(f"Directory {audio_dir} does not exist")
        return
    
    # Assume directory structure: audio_dir/speaker_name/*.wav
    for speaker_dir in audio_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_name = speaker_dir.name
            
            # Add speaker to database
            speaker_id = db.add_speaker(speaker_name)
            print(f"Processing speaker: {speaker_name} (ID: {speaker_id})")
            
            # Process audio files
            audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.mp3"))
            
            for audio_file in audio_files:
                try:
                    # Process audio
                    processed_data = processor.process_for_training(
                        str(audio_file), segment_length=3.0, augment=True
                    )
                    
                    if processed_data:
                        # Add to database
                        audio_id = db.add_audio_file(
                            speaker_id=speaker_id,
                            file_path=str(audio_file),
                            duration=len(processed_data[0]['audio']) / processor.sample_rate
                        )
                        
                        # Store features
                        for data in processed_data:
                            features = data['features']
                            db.store_features(audio_id, "mfcc", features['mfcc'])
                            db.store_features(audio_id, "mel_spectrogram", features['mel_spectrogram'])
                        
                        print(f"  ✅ {audio_file.name}: {len(processed_data)} segments")
                    else:
                        print(f"  ❌ {audio_file.name}: processing failed")
                
                except Exception as e:
                    print(f"  ❌ {audio_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare audio data for training")
    parser.add_argument("audio_dir", help="Directory containing speaker subdirectories")
    parser.add_argument("--db-path", default="speaker_recognition.db", 
                       help="Database path")
    
    args = parser.parse_args()
    
    process_directory(args.audio_dir, args.db_path)
    print("Data preparation completed!")

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/prepare_data.py", "w") as f:
        f.write(data_script)
    
    os.chmod("scripts/prepare_data.py", 0o755)
    
    print("✅ Created utility scripts")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Speaker Recognition System")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-db", action="store_true", help="Skip database setup")
    
    args = parser.parse_args()
    
    print("🎤 Setting up Speaker Recognition System...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(dev=args.dev):
            print("❌ Setup failed during dependency installation")
            return 1
    
    # Setup database
    if not args.skip_db:
        if not setup_database():
            print("❌ Setup failed during database initialization")
            return 1
    
    # Create configuration files
    create_config_files()
    
    # Create test files
    create_test_files()
    
    # Create example notebook
    create_example_notebook()
    
    # Create utility scripts
    create_scripts()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Run: streamlit run ui/streamlit_app.py")
    print("3. Open your browser to http://localhost:8501")
    print("4. Check out the example notebook: notebooks/getting_started.ipynb")
    print("\nFor more information, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
