"""
Database management for speaker recognition system
Handles speaker data, audio files, and model metadata
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
import hashlib
import pickle


class SpeakerDatabase:
    """SQLite database for managing speaker data and audio files"""
    
    def __init__(self, db_path: str = "speaker_recognition.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Speakers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                gender TEXT,
                age INTEGER,
                accent TEXT,
                language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Audio files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id INTEGER,
                file_path TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                duration REAL,
                sample_rate INTEGER,
                channels INTEGER,
                file_size INTEGER,
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (speaker_id) REFERENCES speakers (id)
            )
        ''')
        
        # Features table for storing extracted features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                feature_type TEXT NOT NULL,
                features BLOB,
                feature_shape TEXT,
                extraction_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files (id)
            )
        ''')
        
        # Models table for tracking trained models
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                architecture TEXT,
                parameters TEXT,
                accuracy REAL,
                model_path TEXT,
                training_data_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Recognition sessions for tracking predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                audio_file_id INTEGER,
                predicted_speaker_id INTEGER,
                confidence REAL,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id),
                FOREIGN KEY (audio_file_id) REFERENCES audio_files (id),
                FOREIGN KEY (predicted_speaker_id) REFERENCES speakers (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_speaker(self, name: str, gender: str = None, age: int = None, 
                   accent: str = None, language: str = 'en') -> int:
        """Add a new speaker to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO speakers (name, gender, age, accent, language)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, gender, age, accent, language))
            
            speaker_id = cursor.lastrowid
            conn.commit()
            return speaker_id
        
        except sqlite3.IntegrityError:
            # Speaker already exists
            cursor.execute('SELECT id FROM speakers WHERE name = ?', (name,))
            return cursor.fetchone()[0]
        
        finally:
            conn.close()
    
    def add_audio_file(self, speaker_id: int, file_path: str, 
                      duration: float = None, sample_rate: int = None,
                      channels: int = None, quality_score: float = None) -> int:
        """Add audio file record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        try:
            cursor.execute('''
                INSERT INTO audio_files 
                (speaker_id, file_path, file_hash, duration, sample_rate, 
                 channels, file_size, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (speaker_id, file_path, file_hash, duration, sample_rate, 
                  channels, file_size, quality_score))
            
            audio_id = cursor.lastrowid
            conn.commit()
            return audio_id
        
        except sqlite3.IntegrityError:
            # File already exists
            cursor.execute('SELECT id FROM audio_files WHERE file_hash = ?', (file_hash,))
            return cursor.fetchone()[0]
        
        finally:
            conn.close()
    
    def store_features(self, audio_file_id: int, feature_type: str, 
                      features: np.ndarray, extraction_method: str = "default"):
        """Store extracted features in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize features
        features_blob = pickle.dumps(features)
        feature_shape = str(features.shape)
        
        cursor.execute('''
            INSERT INTO audio_features 
            (audio_file_id, feature_type, features, feature_shape, extraction_method)
            VALUES (?, ?, ?, ?, ?)
        ''', (audio_file_id, feature_type, features_blob, feature_shape, extraction_method))
        
        conn.commit()
        conn.close()
    
    def get_features(self, audio_file_id: int, feature_type: str) -> Optional[np.ndarray]:
        """Retrieve features from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT features FROM audio_features 
            WHERE audio_file_id = ? AND feature_type = ?
            ORDER BY created_at DESC LIMIT 1
        ''', (audio_file_id, feature_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def get_speaker_data(self, speaker_id: int = None, speaker_name: str = None) -> Dict:
        """Get speaker information and associated audio files"""
        conn = sqlite3.connect(self.db_path)
        
        if speaker_id:
            query = '''
                SELECT s.*, COUNT(af.id) as audio_count
                FROM speakers s
                LEFT JOIN audio_files af ON s.id = af.speaker_id
                WHERE s.id = ?
                GROUP BY s.id
            '''
            params = (speaker_id,)
        else:
            query = '''
                SELECT s.*, COUNT(af.id) as audio_count
                FROM speakers s
                LEFT JOIN audio_files af ON s.id = af.speaker_id
                WHERE s.name = ?
                GROUP BY s.id
            '''
            params = (speaker_name,)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df.to_dict('records')[0] if not df.empty else {}
    
    def get_all_speakers(self) -> pd.DataFrame:
        """Get all speakers with their audio file counts"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT s.*, COUNT(af.id) as audio_count,
                   AVG(af.duration) as avg_duration,
                   SUM(af.file_size) as total_size
            FROM speakers s
            LEFT JOIN audio_files af ON s.id = af.speaker_id
            GROUP BY s.id
            ORDER BY s.name
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_training_data(self, feature_type: str = "mfcc") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get training data for model training"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT af.features, s.name, af.audio_file_id
            FROM audio_features af
            JOIN audio_files a ON af.audio_file_id = a.id
            JOIN speakers s ON a.speaker_id = s.id
            WHERE af.feature_type = ?
            ORDER BY s.name, af.created_at
        '''
        
        cursor = conn.cursor()
        cursor.execute(query, (feature_type,))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return np.array([]), np.array([]), []
        
        features = []
        labels = []
        file_ids = []
        
        for feature_blob, speaker_name, file_id in results:
            feature_array = pickle.loads(feature_blob)
            features.append(feature_array)
            labels.append(speaker_name)
            file_ids.append(file_id)
        
        return np.array(features), np.array(labels), file_ids
    
    def save_model(self, name: str, model_type: str, architecture: str,
                  parameters: Dict, accuracy: float, model_path: str,
                  training_data_hash: str = None) -> int:
        """Save model metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        parameters_json = json.dumps(parameters)
        
        cursor.execute('''
            INSERT OR REPLACE INTO models 
            (name, model_type, architecture, parameters, accuracy, model_path, training_data_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, model_type, architecture, parameters_json, accuracy, model_path, training_data_hash))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return model_id
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get model information"""
        conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1'
        df = pd.read_sql_query(query, conn, params=(model_name,))
        conn.close()
        
        if not df.empty:
            model_info = df.to_dict('records')[0]
            model_info['parameters'] = json.loads(model_info['parameters'])
            return model_info
        
        return {}
    
    def log_recognition(self, model_id: int, audio_file_id: int,
                       predicted_speaker_id: int, confidence: float,
                       processing_time: float):
        """Log recognition session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recognition_sessions 
            (model_id, audio_file_id, predicted_speaker_id, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_id, audio_file_id, predicted_speaker_id, confidence, processing_time))
        
        conn.commit()
        conn.close()
    
    def get_recognition_stats(self, model_id: int = None) -> pd.DataFrame:
        """Get recognition statistics"""
        conn = sqlite3.connect(self.db_path)
        
        if model_id:
            query = '''
                SELECT rs.*, s.name as predicted_speaker, m.name as model_name
                FROM recognition_sessions rs
                JOIN speakers s ON rs.predicted_speaker_id = s.id
                JOIN models m ON rs.model_id = m.id
                WHERE rs.model_id = ?
                ORDER BY rs.created_at DESC
            '''
            params = (model_id,)
        else:
            query = '''
                SELECT rs.*, s.name as predicted_speaker, m.name as model_name
                FROM recognition_sessions rs
                JOIN speakers s ON rs.predicted_speaker_id = s.id
                JOIN models m ON rs.model_id = m.id
                ORDER BY rs.created_at DESC
            '''
            params = ()
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except FileNotFoundError:
            return ""
    
    def create_mock_data(self):
        """Create mock data for testing"""
        # Add sample speakers
        speakers_data = [
            ("Alice Johnson", "female", 28, "American", "en"),
            ("Bob Smith", "male", 35, "British", "en"),
            ("Carol Davis", "female", 42, "Australian", "en"),
            ("David Wilson", "male", 31, "Canadian", "en"),
            ("Emma Brown", "female", 26, "Irish", "en"),
            ("Frank Miller", "male", 45, "American", "en"),
            ("Grace Lee", "female", 33, "Korean-American", "en"),
            ("Henry Taylor", "male", 39, "Scottish", "en")
        ]
        
        for name, gender, age, accent, language in speakers_data:
            self.add_speaker(name, gender, age, accent, language)
        
        print("Mock speaker data created successfully!")
        
        # Create sample audio file entries (without actual files)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get speaker IDs
        cursor.execute('SELECT id, name FROM speakers')
        speakers = cursor.fetchall()
        
        import random
        
        for speaker_id, speaker_name in speakers:
            # Add 3-5 mock audio files per speaker
            num_files = random.randint(3, 5)
            
            for i in range(num_files):
                file_path = f"mock_data/{speaker_name.lower().replace(' ', '_')}/audio_{i+1}.wav"
                duration = random.uniform(2.0, 10.0)
                sample_rate = 16000
                channels = 1
                quality_score = random.uniform(0.7, 1.0)
                
                audio_id = self.add_audio_file(
                    speaker_id, file_path, duration, sample_rate, channels, quality_score
                )
                
                # Add mock features
                mfcc_features = np.random.randn(13, int(duration * 50))  # Mock MFCC
                mel_features = np.random.randn(80, int(duration * 50))   # Mock Mel spectrogram
                
                self.store_features(audio_id, "mfcc", mfcc_features, "librosa")
                self.store_features(audio_id, "mel_spectrogram", mel_features, "torchaudio")
        
        conn.close()
        print("Mock audio files and features created successfully!")


def create_sample_database():
    """Create a sample database with mock data"""
    db = SpeakerDatabase("sample_speaker_recognition.db")
    db.create_mock_data()
    return db


if __name__ == "__main__":
    # Create sample database
    db = create_sample_database()
    
    # Display statistics
    speakers_df = db.get_all_speakers()
    print("\nSpeaker Database Statistics:")
    print(speakers_df)
    
    print(f"\nTotal speakers: {len(speakers_df)}")
    print(f"Total audio files: {speakers_df['audio_count'].sum()}")
    print(f"Average files per speaker: {speakers_df['audio_count'].mean():.1f}")
