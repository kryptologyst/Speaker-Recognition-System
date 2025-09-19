"""
Advanced Audio Processing for Speaker Recognition
Implements modern feature extraction and data augmentation techniques
"""

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import butter, filtfilt
import webrtcvad
import soundfile as sf
from typing import Tuple, Optional, List
import random


class AudioProcessor:
    """Advanced audio processing with modern techniques"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_mels: int = 80,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = 2048):
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Initialize transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0
        )
        
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'win_length': win_length,
                'n_mels': n_mels
            }
        )
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
    
    def load_audio(self, file_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load audio file with proper preprocessing"""
        try:
            # Load with librosa for better audio handling
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            return None, None
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing to audio signal"""
        # Remove silence using librosa
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # Apply pre-emphasis filter
        audio_preemphasized = self.preemphasis(audio_trimmed)
        
        # Normalize
        audio_normalized = librosa.util.normalize(audio_preemphasized)
        
        return audio_normalized
    
    def preemphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter"""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def voice_activity_detection(self, audio: np.ndarray) -> np.ndarray:
        """Remove non-speech segments using VAD"""
        # Convert to 16-bit PCM for webrtcvad
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Frame size for VAD (10ms, 20ms, or 30ms)
        frame_duration = 30  # ms
        frame_size = int(self.sample_rate * frame_duration / 1000)
        
        voiced_frames = []
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            if len(frame) == frame_size:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                if is_speech:
                    voiced_frames.extend(frame)
        
        return np.array(voiced_frames, dtype=np.float32) / 32767.0
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        mfcc = self.mfcc_transform(audio_tensor)
        return mfcc.squeeze(0).numpy()
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram features"""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        mel_spec = self.mel_spectrogram(audio_tensor)
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        return log_mel_spec.squeeze(0).numpy()
    
    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """Extract comprehensive spectral features"""
        features = {}
        
        # MFCC
        features['mfcc'] = self.extract_mfcc(audio)
        
        # Mel spectrogram
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        
        # Spectral features using librosa
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        features['pitch'] = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        return features
    
    def augment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation techniques"""
        augmented_samples = [audio]  # Original
        
        # Time stretching
        stretch_factors = [0.9, 1.1]
        for factor in stretch_factors:
            stretched = librosa.effects.time_stretch(audio, rate=factor)
            augmented_samples.append(stretched)
        
        # Pitch shifting
        pitch_shifts = [-2, 2]  # semitones
        for shift in pitch_shifts:
            pitched = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=shift)
            augmented_samples.append(pitched)
        
        # Add noise
        noise_levels = [0.005, 0.01]
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, len(audio))
            noisy_audio = audio + noise
            augmented_samples.append(noisy_audio)
        
        # Speed perturbation
        speed_factors = [0.95, 1.05]
        for factor in speed_factors:
            # Resample to change speed
            new_length = int(len(audio) / factor)
            speed_audio = librosa.resample(audio, orig_sr=self.sample_rate, 
                                         target_sr=int(self.sample_rate * factor))
            # Pad or trim to original length
            if len(speed_audio) > len(audio):
                speed_audio = speed_audio[:len(audio)]
            else:
                speed_audio = np.pad(speed_audio, (0, len(audio) - len(speed_audio)))
            augmented_samples.append(speed_audio)
        
        return augmented_samples
    
    def segment_audio(self, audio: np.ndarray, segment_length: float = 3.0, 
                     overlap: float = 0.5) -> List[np.ndarray]:
        """Segment long audio into smaller chunks"""
        segment_samples = int(segment_length * self.sample_rate)
        overlap_samples = int(overlap * segment_samples)
        step_size = segment_samples - overlap_samples
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, step_size):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
        
        return segments
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization"""
        mean = np.mean(features, axis=-1, keepdims=True)
        std = np.std(features, axis=-1, keepdims=True)
        return (features - mean) / (std + 1e-8)
    
    def process_for_training(self, file_path: str, segment_length: float = 3.0,
                           augment: bool = True) -> List[dict]:
        """Complete processing pipeline for training data"""
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return []
        
        # Preprocess
        audio = self.preprocess_audio(audio)
        
        # Apply VAD
        audio = self.voice_activity_detection(audio)
        
        if len(audio) < self.sample_rate:  # Less than 1 second
            return []
        
        # Segment audio
        segments = self.segment_audio(audio, segment_length)
        
        processed_data = []
        for segment in segments:
            if len(segment) < self.sample_rate * 0.5:  # Skip very short segments
                continue
            
            # Extract features
            features = self.extract_spectral_features(segment)
            
            # Normalize MFCC and mel spectrogram
            features['mfcc'] = self.normalize_features(features['mfcc'])
            features['mel_spectrogram'] = self.normalize_features(features['mel_spectrogram'])
            
            processed_data.append({
                'audio': segment,
                'features': features,
                'file_path': file_path
            })
            
            # Data augmentation
            if augment:
                augmented_segments = self.augment_audio(segment)
                for aug_segment in augmented_segments[1:]:  # Skip original
                    aug_features = self.extract_spectral_features(aug_segment)
                    aug_features['mfcc'] = self.normalize_features(aug_features['mfcc'])
                    aug_features['mel_spectrogram'] = self.normalize_features(
                        aug_features['mel_spectrogram'])
                    
                    processed_data.append({
                        'audio': aug_segment,
                        'features': aug_features,
                        'file_path': file_path,
                        'augmented': True
                    })
        
        return processed_data


class RealTimeAudioProcessor:
    """Real-time audio processing for live speaker recognition"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.buffer = np.array([])
        self.buffer_size = sample_rate * 3  # 3 seconds buffer
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[dict]:
        """Process incoming audio chunk"""
        # Add to buffer
        self.buffer = np.append(self.buffer, audio_chunk)
        
        # Keep buffer size manageable
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        # Process if we have enough data
        if len(self.buffer) >= self.sample_rate * 2:  # 2 seconds minimum
            features = self.audio_processor.extract_spectral_features(self.buffer)
            features['mfcc'] = self.audio_processor.normalize_features(features['mfcc'])
            features['mel_spectrogram'] = self.audio_processor.normalize_features(
                features['mel_spectrogram'])
            
            return {
                'audio': self.buffer.copy(),
                'features': features,
                'timestamp': np.datetime64('now')
            }
        
        return None
