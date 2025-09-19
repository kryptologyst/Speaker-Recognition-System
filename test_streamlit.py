"""
Simple test version of Streamlit app to debug loading issues
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="🎤 Speaker Recognition Test",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Speaker Recognition System - Test Version")

try:
    from data.database import SpeakerDatabase
    st.success("✅ Database module imported successfully")
    
    db = SpeakerDatabase()
    st.success("✅ Database initialized successfully")
    
    # Initialize with sample data if empty
    speakers_df = db.get_all_speakers()
    if len(speakers_df) == 0:
        st.warning("⚠️ Database is empty. Creating sample data...")
        db.create_mock_data()
        speakers_df = db.get_all_speakers()
        st.success(f"✅ Created {len(speakers_df)} sample speakers")
    else:
        st.info(f"📊 Found {len(speakers_df)} speakers in database")
    
    # Display speakers
    if not speakers_df.empty:
        st.subheader("Registered Speakers")
        st.dataframe(speakers_df)
    
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.code(str(e))

try:
    from data.audio_processor import AudioProcessor
    processor = AudioProcessor()
    st.success("✅ Audio processor initialized successfully")
except Exception as e:
    st.error(f"❌ Audio processor error: {e}")

st.write("---")
st.write("If you see this message, the basic Streamlit app is working!")
st.write("You can now proceed to use the full application.")

if st.button("🚀 Launch Full Application"):
    st.write("Run: `streamlit run ui/streamlit_app.py --server.port 8503`")
