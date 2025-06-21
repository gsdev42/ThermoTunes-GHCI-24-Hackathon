import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from pathlib import Path
import io
import os


st.set_page_config(page_title="ThermoTunes",
                   layout="centered",
                   page_icon="🌡️",
                   initial_sidebar_state="expanded")


st.markdown("""
    <style>
    .main {
        background-color: #fff5f0;
        color: #8B0000;
    }
    .stApp {
        background: linear-gradient(135deg, #fff5f0 0%, #ffe8e0 100%);
    }
    .css-1d391kg {
        background-color: #ff6347;
    }
    .stButton > button {
        background-color: #ff4500;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff6347;
        color: white;
    }
    .stRadio > div {
        background-color: rgba(255, 69, 0, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    .stMetric {
        background-color: rgba(255, 99, 71, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ff4500;
    }
    h1 {
        color: #ff4500 !important;
        text-align: center;
    }
    h2, h3 {
        color: #ff6347 !important;
    }
    .stSidebar {
        background-color: #ffe8e0;
    }
    </style>
""",
            unsafe_allow_html=True)


st.title("🌡️ Climate Music Generator")
st.markdown("### Making climate data audible through AI")
st.markdown(
    "This application uses LSTM neural networks to convert climate data patterns into musical compositions."
)


st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section:", ["Visualize", "Listen"])



@st.cache_data
def load_climate_data():
    """Load and process climate dataset"""
    file_paths = [
        "your_climate_dataset.csv", "sample_climate_data.csv",
        "climate_data.csv"
    ]

    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'dt' in df.columns:
                    df['dt'] = pd.to_datetime(df['dt'])
                elif 'Date' in df.columns:
                    df['dt'] = pd.to_datetime(df['Date'])

               
                temp_cols = [
                    'LandAverageTemperature', 'Temperature', 'AvgTemperature',
                    'temp'
                ]
                for col in temp_cols:
                    if col in df.columns:
                        df['LandAverageTemperature'] = df[col]
                        break

                return df.dropna(subset=['LandAverageTemperature'])
            except Exception as e:
                continue

    return None



df = load_climate_data()

if df is None:
    st.error(
        "⚠️ No climate dataset found. Please ensure you have one of the following files:"
    )
    st.markdown("- `your_climate_dataset.csv`")
    st.markdown("- `sample_climate_data.csv`")
    st.markdown("- `climate_data.csv`")
    st.info(
        "💡 You can generate sample data using the 'Generate New Music' section."
    )


if page == "Visualize":
    st.header("Global Temperature Trends")

    if df is not None:
  
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", "20000+")
        with col2:
            if 'LandAverageTemperature' in df.columns:
                avg_temp = df['LandAverageTemperature'].mean()
            if 'dt' in df.columns:
                date_range = f"{df['dt'].min().year} - {df['dt'].max().year}"
                st.metric("Date Range", date_range)

    
        st.subheader("Global Temperature Over Time")

        fig, ax = plt.subplots(figsize=(12, 6))

        if 'dt' in df.columns and 'LandAverageTemperature' in df.columns:
            ax.plot(df['dt'],
                    df['LandAverageTemperature'],
                    color='#ff4500',
                    linewidth=2.5,
                    alpha=0.9)
            ax.set_xlabel("Year", fontsize=12, color='#8B0000')
            ax.set_ylabel("Land Average Temperature (°C)",
                          fontsize=12,
                          color='#8B0000')
            ax.set_title("Global Land Average Temperature",
                         fontsize=16,
                         pad=20,
                         color='#ff4500')
            ax.grid(True, linestyle='--', alpha=0.3, color='#ff6347')

           
            if len(df) > 1:
                z = np.polyfit(df.index, df['LandAverageTemperature'], 1)
                p = np.poly1d(z)
                ax.plot(df['dt'],
                        p(df.index),
                        color='#dc143c',
                        linestyle='--',
                        alpha=0.8,
                        linewidth=2,
                        label='Warming Trend')
                ax.legend()

      
            ax.set_facecolor('#fff8f0')
            fig.patch.set_facecolor('#fff5f0')

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("No climate dataset found. Generating sample data...")
    
        np.random.seed(42)
        dates = pd.date_range(start='1900-01-01', end='2023-12-31', freq='M')

        
        base_temp = 8.5
        trend = np.linspace(0, 1.2, len(dates))  
        seasonal = 3 * np.sin(
            2 * np.pi * np.arange(len(dates)) / 12)  
        noise = np.random.normal(0, 0.3, len(dates)) 

        temperatures = base_temp + trend + seasonal + noise

        sample_df = pd.DataFrame({
            'dt':
            dates,
            'LandAverageTemperature':
            temperatures,
            'LandAverageTemperatureUncertainty':
            np.random.uniform(0.1, 0.5, len(dates))
        })

        sample_df.to_csv('sample_climate_data.csv', index=False)
        st.success("Sample climate data generated successfully!")
        st.rerun()


elif page == "Listen":
    st.header("AI-Generated Climate Music")
    st.markdown(
        "Experience how climate data sounds when transformed into music through machine learning."
    )

    midi_paths = [
        "generated_climate_music.mid", "climate_music.mid", "output.mid"
    ]
    midi_file = None

    for path in midi_paths:
        if Path(path).exists():
            midi_file = path
            break

    if midi_file:
        st.success(f"Found generated music file: {midi_file}")

   
        file_size = Path(midi_file).stat().st_size
        st.info(f"File size: {file_size} bytes")

       
        with open(midi_file, "rb") as f:
            midi_data = f.read()
            st.download_button(label="Download MIDI File",
                               data=midi_data,
                               file_name=midi_file,
                               mime="audio/midi")

        st.markdown("---")
        st.markdown("**About the Generated Music:**")
        st.markdown(
            "- Generated using LSTM neural networks trained on temperature patterns"
        )
        st.markdown("- Chord progressions reflect climate data variations")
        st.markdown("- Each note corresponds to temperature changes over time")

    else:
        st.warning("No generated music file found.")
        st.info(
            "Generate music first by training the LSTM model on the climate data."
        )

      
        if df is not None:
            if st.button("Generate Music Now"):
                with st.spinner("Training LSTM model and generating music..."):
                    try:
                        from climate_lstm_generator import generate_climate_music
                        success = generate_climate_music(df)

                        if success:
                            st.success(
                                "Music generation completed successfully!")
                            st.rerun()
                        else:
                            st.error(
                                "Music generation failed. Please try again.")

                    except Exception as e:
                        st.error(f"Error during music generation: {str(e)}")
        else:
            st.info(
                "Climate data is required to generate music. Please check the Visualize section first."
            )


st.markdown("---")
