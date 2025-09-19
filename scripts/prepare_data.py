#!/usr/bin/env python3
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
