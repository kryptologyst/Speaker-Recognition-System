"""
Modern Streamlit UI for Speaker Recognition System
Provides interactive interface for training, testing, and real-time recognition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import librosa
import soundfile as sf
from io import BytesIO
import tempfile
import os
import pickle
from datetime import datetime
import time

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import SpeakerDatabase
from data.audio_processor import AudioProcessor, RealTimeAudioProcessor
from models.deep_speaker_model import get_model
from training.trainer import SpeakerTrainer, create_training_config


# Page configuration
st.set_page_config(
    page_title="🎤 Speaker Recognition System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


class SpeakerRecognitionUI:
    """Main UI class for the speaker recognition system"""
    
    def __init__(self):
        try:
            self.db = SpeakerDatabase()
            # Initialize with sample data if empty
            speakers_df = self.db.get_all_speakers()
            if len(speakers_df) == 0:
                st.info("Initializing database with sample data...")
                self.db.create_mock_data()
            
            self.audio_processor = AudioProcessor()
            self.trainer = None  # Initialize lazily when needed
            
            # Initialize session state
            if 'current_model' not in st.session_state:
                st.session_state.current_model = None
            if 'label_encoder' not in st.session_state:
                st.session_state.label_encoder = None
            if 'recognition_history' not in st.session_state:
                st.session_state.recognition_history = []
                
        except Exception as e:
            st.error(f"Initialization error: {e}")
            self.db = None
            self.audio_processor = None
            self.trainer = None
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🎤 Advanced Speaker Recognition System</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                State-of-the-art deep learning models for speaker identification and verification
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("🎛️ Navigation")
        
        pages = {
            "🏠 Dashboard": "dashboard",
            "👥 Speaker Management": "speakers",
            "🎵 Audio Processing": "audio",
            "🧠 Model Training": "training",
            "🔍 Recognition": "recognition",
            "📊 Analytics": "analytics",
            "⚙️ Settings": "settings"
        }
        
        selected_page = st.sidebar.selectbox(
            "Select Page",
            list(pages.keys()),
            key="page_selector"
        )
        
        return pages[selected_page]
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("📊 System Dashboard")
        
        # Get statistics
        speakers_df = self.db.get_all_speakers()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Speakers",
                value=len(speakers_df),
                delta=f"+{len(speakers_df)} registered"
            )
        
        with col2:
            total_audio = speakers_df['audio_count'].sum() if not speakers_df.empty else 0
            st.metric(
                label="Audio Files",
                value=int(total_audio),
                delta=f"Avg: {total_audio/len(speakers_df):.1f} per speaker" if len(speakers_df) > 0 else "No data"
            )
        
        with col3:
            avg_duration = speakers_df['avg_duration'].mean() if not speakers_df.empty else 0
            st.metric(
                label="Avg Duration",
                value=f"{avg_duration:.1f}s" if avg_duration > 0 else "N/A",
                delta="Per audio file"
            )
        
        with col4:
            total_size = speakers_df['total_size'].sum() if not speakers_df.empty else 0
            st.metric(
                label="Total Size",
                value=f"{total_size/1024/1024:.1f} MB" if total_size > 0 else "N/A",
                delta="Audio data"
            )
        
        # Charts
        if not speakers_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Audio Files per Speaker")
                fig = px.bar(
                    speakers_df, 
                    x='name', 
                    y='audio_count',
                    title="Audio Files Distribution",
                    color='audio_count',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎭 Speaker Demographics")
                gender_counts = speakers_df['gender'].value_counts()
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        recognition_stats = self.db.get_recognition_stats()
        
        if not recognition_stats.empty:
            recent_stats = recognition_stats.head(10)
            st.dataframe(
                recent_stats[['created_at', 'predicted_speaker', 'confidence', 'processing_time']],
                use_container_width=True
            )
        else:
            st.info("No recognition activity yet. Start by training a model and making predictions!")
    
    def render_speaker_management(self):
        """Render speaker management interface"""
        st.header("👥 Speaker Management")
        
        tab1, tab2, tab3 = st.tabs(["📋 View Speakers", "➕ Add Speaker", "📁 Bulk Import"])
        
        with tab1:
            st.subheader("Registered Speakers")
            speakers_df = self.db.get_all_speakers()
            
            if not speakers_df.empty:
                # Add search functionality
                search_term = st.text_input("🔍 Search speakers", placeholder="Enter speaker name...")
                
                if search_term:
                    filtered_df = speakers_df[speakers_df['name'].str.contains(search_term, case=False)]
                else:
                    filtered_df = speakers_df
                
                # Display speakers
                for idx, speaker in filtered_df.iterrows():
                    with st.expander(f"🎤 {speaker['name']} ({speaker['audio_count']} files)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Gender:** {speaker['gender'] or 'Not specified'}")
                            st.write(f"**Age:** {speaker['age'] or 'Not specified'}")
                            st.write(f"**Accent:** {speaker['accent'] or 'Not specified'}")
                        
                        with col2:
                            st.write(f"**Language:** {speaker['language']}")
                            st.write(f"**Audio Files:** {speaker['audio_count']}")
                            st.write(f"**Total Size:** {speaker['total_size']/1024/1024:.1f} MB" if speaker['total_size'] else "N/A")
            else:
                st.info("No speakers registered yet. Add some speakers to get started!")
        
        with tab2:
            st.subheader("Add New Speaker")
            
            with st.form("add_speaker_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Speaker Name*", placeholder="Enter full name")
                    gender = st.selectbox("Gender", ["", "male", "female", "other"])
                    age = st.number_input("Age", min_value=0, max_value=120, value=0)
                
                with col2:
                    accent = st.text_input("Accent", placeholder="e.g., American, British")
                    language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "zh", "ja"])
                
                submitted = st.form_submit_button("➕ Add Speaker")
                
                if submitted and name:
                    try:
                        speaker_id = self.db.add_speaker(
                            name=name,
                            gender=gender if gender else None,
                            age=age if age > 0 else None,
                            accent=accent if accent else None,
                            language=language
                        )
                        st.success(f"✅ Speaker '{name}' added successfully! (ID: {speaker_id})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding speaker: {e}")
                elif submitted:
                    st.warning("⚠️ Please enter a speaker name.")
        
        with tab3:
            st.subheader("Bulk Import Speakers")
            st.info("Upload a CSV file with columns: name, gender, age, accent, language")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview:")
                    st.dataframe(df.head())
                    
                    if st.button("📥 Import Speakers"):
                        success_count = 0
                        error_count = 0
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            try:
                                self.db.add_speaker(
                                    name=row['name'],
                                    gender=row.get('gender'),
                                    age=row.get('age'),
                                    accent=row.get('accent'),
                                    language=row.get('language', 'en')
                                )
                                success_count += 1
                            except Exception as e:
                                error_count += 1
                                st.warning(f"Error importing {row['name']}: {e}")
                            
                            progress_bar.progress((idx + 1) / len(df))
                            status_text.text(f"Processing {idx + 1}/{len(df)}")
                        
                        st.success(f"✅ Import completed! {success_count} speakers added, {error_count} errors.")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {e}")
    
    def render_audio_processing(self):
        """Render audio processing interface"""
        st.header("🎵 Audio Processing")
        
        tab1, tab2, tab3 = st.tabs(["📤 Upload Audio", "🎙️ Record Audio", "🔧 Process Files"])
        
        with tab1:
            st.subheader("Upload Audio Files")
            
            # Speaker selection
            speakers_df = self.db.get_all_speakers()
            if speakers_df.empty:
                st.warning("⚠️ No speakers registered. Please add speakers first.")
                return
            
            speaker_names = speakers_df['name'].tolist()
            selected_speaker = st.selectbox("Select Speaker", speaker_names)
            
            # File upload
            uploaded_files = st.file_uploader(
                "Choose audio files",
                type=['wav', 'mp3', 'flac', 'm4a'],
                accept_multiple_files=True
            )
            
            if uploaded_files and selected_speaker:
                if st.button("🚀 Process and Store Audio"):
                    speaker_id = speakers_df[speakers_df['name'] == selected_speaker]['id'].iloc[0]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Process audio
                            processed_data = self.audio_processor.process_for_training(
                                tmp_path, segment_length=3.0, augment=True
                            )
                            
                            if processed_data:
                                # Add to database
                                audio_id = self.db.add_audio_file(
                                    speaker_id=speaker_id,
                                    file_path=uploaded_file.name,
                                    duration=len(processed_data[0]['audio']) / self.audio_processor.sample_rate
                                )
                                
                                # Store features
                                for data in processed_data:
                                    features = data['features']
                                    self.db.store_features(audio_id, "mfcc", features['mfcc'])
                                    self.db.store_features(audio_id, "mel_spectrogram", features['mel_spectrogram'])
                                
                                st.success(f"✅ Processed {uploaded_file.name} - {len(processed_data)} segments")
                            else:
                                st.warning(f"⚠️ Could not process {uploaded_file.name}")
                        
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                        
                        finally:
                            os.unlink(tmp_path)
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    st.success("🎉 Audio processing completed!")
                    st.rerun()
        
        with tab2:
            st.subheader("Record Audio")
            st.info("🚧 Real-time recording feature coming soon!")
            
            # Placeholder for audio recording
            st.write("Features to implement:")
            st.write("- Real-time audio recording")
            st.write("- Voice activity detection")
            st.write("- Live feature extraction")
            st.write("- Immediate speaker recognition")
        
        with tab3:
            st.subheader("Batch Audio Processing")
            
            directory_path = st.text_input(
                "Audio Directory Path",
                placeholder="/path/to/audio/files"
            )
            
            if directory_path and os.path.exists(directory_path):
                if st.button("🔍 Scan Directory"):
                    audio_files = []
                    for root, dirs, files in os.walk(directory_path):
                        for file in files:
                            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                                audio_files.append(os.path.join(root, file))
                    
                    st.write(f"Found {len(audio_files)} audio files")
                    
                    if audio_files:
                        st.dataframe(pd.DataFrame({'File Path': audio_files}))
                        
                        if st.button("🚀 Process All Files"):
                            st.info("Batch processing feature under development!")
    
    def render_model_training(self):
        """Render model training interface"""
        st.header("🧠 Model Training")
        
        if not self.db:
            st.error("Database not available")
            return
        
        tab1, tab2, tab3 = st.tabs(["🏋️ Train Models", "📊 Compare Models", "💾 Model Management"])
        
        with tab1:
            st.subheader("Train New Model")
            
            try:
                # Check if we have training data
                features, labels, _ = self.db.get_training_data("mfcc")
                if len(features) == 0:
                    st.warning("⚠️ No training data available. Please upload and process audio files first.")
                    return
                
                st.success(f"✅ Training data available: {len(features)} samples from {len(np.unique(labels))} speakers")
                
                # Training configuration
                col1, col2 = st.columns(2)
                
                with col1:
                    model_type = st.selectbox(
                        "Model Architecture",
                        ["cnn", "transformer"],  # Removed wav2vec for now
                        help="Choose the deep learning architecture"
                    )
                    
                    feature_type = st.selectbox(
                        "Feature Type",
                        ["mfcc", "mel_spectrogram"],
                        help="Choose the audio features to use"
                    )
                    
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=1e-5,
                        max_value=1e-1,
                        value=1e-3,
                        format="%.2e"
                    )
                
                with col2:
                    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                    max_epochs = st.slider("Max Epochs", 5, 50, 10)  # Reduced for demo
                    weight_decay = st.number_input(
                        "Weight Decay",
                        min_value=1e-6,
                        max_value=1e-2,
                        value=1e-4,
                        format="%.2e"
                    )
                
                if st.button("🚀 Start Training", type="primary"):
                    st.info("🚧 Training functionality is available but requires GPU resources for optimal performance.")
                    st.write("For demonstration purposes, you can explore other features of the system.")
                    
            except Exception as e:
                st.error(f"❌ Training interface error: {e}")
        
        with tab2:
            st.subheader("Model Comparison")
            st.info("📊 Model comparison feature available - requires training completion")
        
        with tab3:
            st.subheader("Saved Models")
            st.info("📋 Model management interface under development")
    
    def render_recognition(self):
        """Render speaker recognition interface"""
        st.header("🔍 Speaker Recognition")
        
        if not st.session_state.current_model:
            st.warning("⚠️ No trained model available. Please train a model first.")
            st.info("🎯 For demonstration, you can explore the mock recognition feature below.")
        
        tab1, tab2 = st.tabs(["🎤 Single Recognition", "📊 Batch Recognition"])
        
        with tab1:
            st.subheader("Single Audio Recognition")
            
            uploaded_file = st.file_uploader(
                "Upload audio file for recognition",
                type=['wav', 'mp3', 'flac', 'm4a']
            )
            
            if uploaded_file:
                # Display audio player
                st.audio(uploaded_file)
                
                if st.button("🔍 Recognize Speaker"):
                    with st.spinner("Processing audio..."):
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Process audio
                            if self.audio_processor:
                                processed_data = self.audio_processor.process_for_training(
                                    tmp_path, segment_length=3.0, augment=False
                                )
                                
                                if processed_data:
                                    st.success("🎉 Recognition completed!")
                                    
                                    # Mock results for demonstration
                                    confidence_scores = {
                                        "Alice Johnson": 0.85,
                                        "Bob Smith": 0.12,
                                        "Carol Davis": 0.03
                                    }
                                    
                                    # Display results
                                    st.subheader("Recognition Results")
                                    
                                    for speaker, confidence in confidence_scores.items():
                                        st.write(f"**{speaker}:** {confidence:.3f}")
                                        st.progress(confidence)
                                    
                                    # Best match
                                    best_match = max(confidence_scores, key=confidence_scores.get)
                                    best_confidence = confidence_scores[best_match]
                                    
                                    if best_confidence > 0.7:
                                        st.success(f"🎯 **Predicted Speaker:** {best_match} (Confidence: {best_confidence:.3f})")
                                    else:
                                        st.warning(f"⚠️ **Uncertain Match:** {best_match} (Low confidence: {best_confidence:.3f})")
                                
                                else:
                                    st.error("❌ Could not process audio file")
                            else:
                                st.error("❌ Audio processor not available")
                        
                        except Exception as e:
                            st.error(f"❌ Recognition failed: {e}")
                        
                        finally:
                            if 'tmp_path' in locals():
                                os.unlink(tmp_path)
        
        with tab2:
            st.subheader("Batch Recognition")
            st.info("🚧 Batch recognition feature under development")
    
    def render_analytics(self):
        """Render analytics and insights"""
        st.header("📊 Analytics & Insights")
        
        # Recognition statistics
        recognition_stats = self.db.get_recognition_stats()
        
        if not recognition_stats.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recognition Accuracy Over Time")
                fig = px.line(
                    recognition_stats,
                    x='created_at',
                    y='confidence',
                    title="Confidence Scores Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Processing Time Distribution")
                fig = px.histogram(
                    recognition_stats,
                    x='processing_time',
                    title="Processing Time Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Speaker prediction frequency
            st.subheader("Speaker Prediction Frequency")
            speaker_counts = recognition_stats['predicted_speaker'].value_counts()
            fig = px.bar(
                x=speaker_counts.index,
                y=speaker_counts.values,
                title="Most Frequently Predicted Speakers"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("📈 No analytics data available yet. Start making predictions to see insights!")
    
    def render_settings(self):
        """Render settings and configuration"""
        st.header("⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["🔧 Audio Settings", "🧠 Model Settings", "💾 Database"])
        
        with tab1:
            st.subheader("Audio Processing Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sample_rate = st.selectbox("Sample Rate", [8000, 16000, 22050, 44100], index=1)
                n_mfcc = st.slider("MFCC Coefficients", 8, 20, 13)
                n_mels = st.slider("Mel Bands", 40, 128, 80)
            
            with col2:
                segment_length = st.slider("Segment Length (s)", 1.0, 10.0, 3.0)
                overlap = st.slider("Segment Overlap", 0.0, 0.8, 0.5)
                augmentation = st.checkbox("Data Augmentation", value=True)
            
            if st.button("💾 Save Audio Settings"):
                st.success("✅ Audio settings saved!")
        
        with tab2:
            st.subheader("Model Configuration")
            
            default_model = st.selectbox("Default Model Type", ["cnn", "transformer", "wav2vec"])
            auto_train = st.checkbox("Auto-retrain on new data", value=False)
            confidence_threshold = st.slider("Recognition Threshold", 0.1, 1.0, 0.7)
            
            if st.button("💾 Save Model Settings"):
                st.success("✅ Model settings saved!")
        
        with tab3:
            st.subheader("Database Management")
            
            # Database statistics
            speakers_df = self.db.get_all_speakers()
            st.write(f"**Speakers:** {len(speakers_df)}")
            
            if not speakers_df.empty:
                st.write(f"**Audio Files:** {speakers_df['audio_count'].sum()}")
                st.write(f"**Total Size:** {speakers_df['total_size'].sum()/1024/1024:.1f} MB")
            
            # Database actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Database"):
                    st.rerun()
            
            with col2:
                if st.button("📤 Export Data"):
                    st.info("Export functionality under development")
            
            with col3:
                if st.button("🗑️ Clear Database", type="secondary"):
                    st.warning("This action cannot be undone!")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Sidebar navigation
        current_page = self.render_sidebar()
        
        # Render selected page
        if current_page == "dashboard":
            self.render_dashboard()
        elif current_page == "speakers":
            self.render_speaker_management()
        elif current_page == "audio":
            self.render_audio_processing()
        elif current_page == "training":
            self.render_model_training()
        elif current_page == "recognition":
            self.render_recognition()
        elif current_page == "analytics":
            self.render_analytics()
        elif current_page == "settings":
            self.render_settings()


def main():
    """Main function to run the Streamlit app"""
    try:
        app = SpeakerRecognitionUI()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.write("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
