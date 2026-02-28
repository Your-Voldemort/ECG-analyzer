import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from model import (
    Config,
    FeatureExtractionModule,
    create_rpm_representations
)

config = Config()

st.set_page_config(
    page_title="ECG Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def resample_signal(signal, original_rate, target_rate=100):
    """
    Resample ECG signal to target sampling rate using linear interpolation.
    """
    if original_rate == target_rate:
        return signal
    original_length = len(signal)
    target_length = int(original_length * target_rate / original_rate)
    indices = np.linspace(0, original_length - 1, target_length)
    return np.interp(indices, np.arange(original_length), signal).astype(np.float32)


def process_single_signal(signal, sampling_rate):
    """ Preprocess a single ECG signal for model inference."""
    if sampling_rate != 100:
        signal = resample_signal(signal, sampling_rate, 100)
    
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    
    if len(signal) > config.SIGNAL_LENGTH:
        signal = signal[:config.SIGNAL_LENGTH]
    elif len(signal) < config.SIGNAL_LENGTH:
        padding = np.zeros(config.SIGNAL_LENGTH - len(signal))
        signal = np.concatenate([signal, padding])
    
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
    
    mean = signal_tensor.mean()
    std = signal_tensor.std()
    if std > 1e-8:
        signal_normalized = (signal_tensor - mean) / std
    else:
        signal_normalized = signal_tensor - mean
    
    rpm = create_rpm_representations(signal_normalized, config.RPM_SIZE)
    return rpm

def parse_csv(content, device_type='generic', sampling_rate=100):
    df = pd.read_csv(content)
    
    if device_type == 'apple':
        possible_cols = ['ecg', 'ecg_mv', 'value', 'microvolts', 'uV']
        signal_col = None
        for col in df.columns:
            if any(pc.lower() in col.lower() for pc in possible_cols):
                signal_col = col
                break
        if signal_col is None:
            signal_col = df.select_dtypes(include=[np.number]).columns[0]
        return df[signal_col].dropna().values.astype(np.float32), 512
    
    elif device_type == 'samsung':
        signal_col = df.select_dtypes(include=[np.number]).columns[0]
        return df[signal_col].dropna().values.astype(np.float32), 500
    
    else:
        if df.shape[1] > 1:
            signal_col = df.select_dtypes(include=[np.number]).columns[0]
        else:
            signal_col = df.columns[0]
        return df[signal_col].dropna().values.astype(np.float32), sampling_rate


def generate_demo_ecg(ecg_type='normal', duration_sec=10, fs=100):
    """
    Generate synthetic ECG signal for demonstration purposes.
    
    Creates a physiologically-inspired ECG waveform with P-wave, QRS complex,
    and T-wave components. Can generate both normal and abnormal patterns.
    """    
    t = np.linspace(0, duration_sec, duration_sec * fs)
    ecg = np.zeros_like(t)
    
    for beat_time in np.arange(0, duration_sec, 0.8):
        beat_idx = (t >= beat_time) & (t < beat_time + 0.7)
        bt = t[beat_idx] - beat_time
        
        p = 0.15 * np.exp(-((bt - 0.1) ** 2) / 0.002)
        qrs = -0.1 * np.exp(-((bt - 0.2) ** 2) / 0.0005) + \
              1.0 * np.exp(-((bt - 0.22) ** 2) / 0.0008) + \
              -0.2 * np.exp(-((bt - 0.24) ** 2) / 0.0005)
        t_wave = 0.3 * np.exp(-((bt - 0.4) ** 2) / 0.008)
        ecg[beat_idx] += p + qrs + t_wave
    
    if ecg_type == 'abnormal':
        for beat_time in np.arange(0, duration_sec, 0.8):
            st_idx = (t >= beat_time + 0.25) & (t < beat_time + 0.35)
            ecg[st_idx] += 0.3
    
    ecg += 0.02 * np.random.randn(len(ecg))
    return ecg.astype(np.float32), fs

@st.cache_resource
def initialize_gemini():
    if not GEMINI_AVAILABLE:
        return None, "google-generativeai not installed"
    
    try:
        api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', None)
        if not api_key:
            return None, "API key not configured"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model, None
    except Exception as e:
        return None, str(e)


def estimate_heart_rate(signal, fs):
    """"
    Estimate heart rate from ECG signal using peak detection.
    
    Detects R-peaks in the QRS complex and calculates heart rate based
    on R-R intervals. Uses scipy's find_peaks for robust detection.
    
    Algorithm:
    1. Detect R-peaks with minimum distance of 0.6s (100 bpm max)
    2. Calculate R-R intervals in seconds
    3. Compute mean heart rate: 60 / mean(RR_interval)   
    """
    try:
        from scipy.signal import find_peaks
        # Detect R-peaks (QRS complexes)
        # distance: minimum samples between peaks (0.6s = 100 bpm limit)
        # prominence: minimum peak height relative to baseline
        peaks, _ = find_peaks(signal, distance=int(0.6 * fs), prominence=0.3)
        if len(peaks) < 2:
            return 75.0
        rr_intervals = np.diff(peaks) / fs
        return 60 / np.mean(rr_intervals) # Convert to BPM
    except:
        return 75.0


def analyze_with_gemini(gemini_model, results, signal, fs):
    """
    Generate AI-powered medical interpretation using Google Gemini.
    
    Creates a structured, patient-friendly explanation of ECG results including:
    - What was found in the ECG
    - Medical significance
    - Recommended next steps
    """
    hr = estimate_heart_rate(signal, fs)
    diagnosis = results['detailed_class']
    confidence = max(results['all_probs'].values()) * 100
    
    class_names = {
        'NORM': 'Normal ECG',
        'MI': 'Myocardial Infarction',
        'STTC': 'ST-T Wave Changes',
        'CD': 'Conduction Disturbance',
        'HYP': 'Cardiac Hypertrophy'
    }
    
    prompt = f"""You are a cardiologist. Explain this ECG result in 3 clear sections.

Result: {diagnosis} ({class_names.get(diagnosis)})
Confidence: {confidence:.1f}%
Heart Rate: {hr:.0f} bpm

Write EXACTLY this format:

**What We Found:**
One clear sentence about what the ECG shows.

**What This Means:**
One clear sentence about the medical significance.

**What You Should Do:**
One clear sentence about next steps.

---

**Confidence:** {confidence:.0f}% | **Heart Rate:** {hr:.0f} bpm

Rules:
- Each section is ONE sentence only
- Use simple language
- Be direct and specific
- For normal: be reassuring
- For abnormal: state urgency clearly
- No extra text, no disclaimers"""
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0,
                'top_p': 0.2,
                'max_output_tokens': 1500,
            }
        )
        return response.text
    except Exception as e:
        return None

@st.cache_resource
def load_model():
    model = FeatureExtractionModule(config)
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model.pth')
        
        # Try loading from script directory first
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, True
        
        # Fallback to current working directory
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
            model.eval()
            return model, True
        
        return model, False
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return model, False


def predict(model, signal, sampling_rate):
    """
    Predict ECG classification using trained model.
    
    Performs end-to-end inference pipeline: Signal preprocessing and RPM generation,
    Model forward pass, Probability calculation via softmax, Class prediction
    """
    rpm = process_single_signal(signal, sampling_rate)
    
    with torch.no_grad():
        _, logits = model(rpm)
        probs = F.softmax(logits, dim=1).squeeze()
    
    pred_idx = probs.argmax().item()
    pred_name = config.CLASS_NAMES[pred_idx]
    sorted_probs = torch.sort(probs, descending=True).values
    top_probability = sorted_probs[0].item()
    second_probability = sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0
    
    return {
        'is_normal': pred_name == "NORM",
        'detailed_class': pred_name,
        'all_probs': {name: probs[i].item() for i, name in enumerate(config.CLASS_NAMES)},
        'top_probability': top_probability,
        'prediction_margin': top_probability - second_probability
    }


def assess_signal_quality(signal, fs):
    signal = np.asarray(signal, dtype=np.float32)
    score = 100
    issues = []
    tips = []

    if len(signal) < fs * 8:
        score -= 20
        issues.append("Recording is short for reliable rhythm interpretation")
        tips.append("Record at least 10 seconds of stable ECG while staying still")

    finite_mask = np.isfinite(signal)
    if not finite_mask.all():
        score -= 35
        issues.append("Signal contains invalid values")
        tips.append("Re-export the file and verify the ECG column has numeric values only")
        signal = signal[finite_mask]

    if len(signal) == 0:
        return {
            'score': 0,
            'grade': 'Poor',
            'issues': ["No valid ECG samples found"],
            'tips': ["Upload a valid CSV with one ECG signal column"]
        }

    dynamic_range = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    if dynamic_range < 0.08:
        score -= 25
        issues.append("Very low signal amplitude (possible weak contact/flat trace)")
        tips.append("Tighten watch contact and keep your hand still during recording")

    signal_std = float(np.std(signal))
    if signal_std < 0.02:
        score -= 20
        issues.append("Low variability detected (possible flatline segment)")

    if len(signal) > 1:
        diff_std = float(np.std(np.diff(signal)))
        noise_ratio = diff_std / (signal_std + 1e-8)
        if noise_ratio > 1.4:
            score -= 20
            issues.append("High noise level detected")
            tips.append("Retake in a quiet posture and avoid movement/talking")

    sig_min, sig_max = float(np.min(signal)), float(np.max(signal))
    if sig_max > sig_min:
        near_clip = ((np.abs(signal - sig_max) < 1e-6) | (np.abs(signal - sig_min) < 1e-6)).mean()
        if near_clip > 0.02:
            score -= 15
            issues.append("Potential clipping at signal limits")
            tips.append("Check device fit and avoid pressing too hard on the sensor")

    edge_len = max(1, int(0.1 * len(signal)))
    edge_shift = abs(float(np.mean(signal[:edge_len]) - np.mean(signal[-edge_len:])))
    if edge_shift > (0.7 * (signal_std + 1e-8)):
        score -= 10
        issues.append("Baseline drift detected")
        tips.append("Keep wrist and finger position stable throughout recording")

    score = int(max(0, min(100, score)))
    grade = 'Good' if score >= 80 else 'Fair' if score >= 60 else 'Poor'

    if not tips:
        tips.append("Current signal quality appears acceptable for screening")

    return {
        'score': score,
        'grade': grade,
        'issues': list(dict.fromkeys(issues)),
        'tips': list(dict.fromkeys(tips))
    }


def build_health_report(results, heart_rate, quality, uncertain=False, uncertain_reasons=None):
    diagnosis = results['detailed_class']
    confidence = max(results['all_probs'].values()) * 100

    class_titles = {
        'NORM': 'Normal ECG Pattern',
        'MI': 'Myocardial Infarction Pattern',
        'STTC': 'ST/T Wave Change Pattern',
        'CD': 'Conduction Disturbance Pattern',
        'HYP': 'Cardiac Hypertrophy Pattern'
    }

    if heart_rate < 50:
        hr_status = "Low heart rate for resting adults"
    elif heart_rate > 100:
        hr_status = "High heart rate for resting adults"
    else:
        hr_status = "Heart rate in common resting range"

    risk_by_class = {
        'NORM': 'Low',
        'MI': 'Urgent',
        'STTC': 'High',
        'CD': 'High',
        'HYP': 'Moderate'
    }
    risk_level = risk_by_class.get(diagnosis, 'Moderate')

    if uncertain:
        risk_level = 'Moderate' if quality['grade'] != 'Poor' else 'High'

    if diagnosis == 'NORM' and (heart_rate < 50 or heart_rate > 100):
        risk_level = 'Moderate'

    assessment_map = {
        'NORM': "The ECG morphology appears normal in this recording.",
        'MI': "The ECG pattern may be consistent with a heart attack-related change.",
        'STTC': "The ECG pattern shows ST/T changes that can indicate cardiac stress or ischemia.",
        'CD': "The ECG pattern suggests delayed electrical conduction in the heart.",
        'HYP': "The ECG pattern suggests possible chamber wall thickening or strain."
    }

    actions_map = {
        'NORM': [
            "Continue healthy lifestyle habits (sleep, hydration, exercise, and stress control).",
            "Repeat ECG if symptoms occur (chest pain, dizziness, palpitations, shortness of breath).",
            "Share this result during routine doctor visits if you have cardiovascular risk factors."
        ],
        'MI': [
            "Seek emergency care immediately, especially if symptoms are present.",
            "Do not drive yourself; contact emergency services or ask someone to assist.",
            "Bring this ECG report and timeline of symptoms to the emergency team."
        ],
        'STTC': [
            "Arrange a same-day or urgent cardiology/medical review.",
            "Avoid intense physical exertion until medically evaluated.",
            "Track symptoms and triggers (activity, stress, medications) to share with your clinician."
        ],
        'CD': [
            "Schedule prompt cardiology review and discuss possible rhythm/conduction testing.",
            "Avoid heavy exertion if you feel faint, dizzy, or unusually fatigued.",
            "Monitor pulse and symptoms daily until reviewed by a clinician."
        ],
        'HYP': [
            "Book a non-emergency medical review to assess blood pressure and cardiac risk.",
            "Track blood pressure at home and limit sodium intake.",
            "Discuss whether echocardiography or further cardiac testing is needed."
        ]
    }

    confidence_note = (
        "High confidence prediction"
        if confidence >= 80
        else "Moderate confidence prediction"
        if confidence >= 60
        else "Lower confidence prediction; consider repeat recording"
    )

    if uncertain:
        diagnosis_title = "Uncertain result - retake recommended"
        assessment = "Model confidence/signal quality is not strong enough for a reliable single-label conclusion."
        actions = [
            "Retake ECG while seated, relaxed, and motionless for at least 10 seconds.",
            "Ensure good skin contact and repeat the recording 1-2 times.",
            "If symptoms are present (chest pain, fainting, breathlessness), seek medical care now."
        ]
        if uncertain_reasons:
            confidence_note = "Uncertain due to: " + "; ".join(uncertain_reasons)
    else:
        diagnosis_title = class_titles.get(diagnosis, diagnosis)
        assessment = assessment_map.get(diagnosis, "Pattern detected; clinical review recommended.")
        actions = actions_map.get(diagnosis, ["Discuss this ECG result with a healthcare professional."])

    return {
        'risk_level': risk_level,
        'diagnosis_title': diagnosis_title,
        'confidence': confidence,
        'heart_rate': heart_rate,
        'heart_rate_status': hr_status,
        'assessment': assessment,
        'actions': actions,
        'confidence_note': confidence_note,
        'quality_score': quality['score'],
        'quality_grade': quality['grade']
    }


def generate_pdf_report(report, results, quality, source, fs):
    buffer = BytesIO()

    lines = [
        "ECG Analyzer Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data source: {source}",
        f"Sampling rate: {fs} Hz",
        "",
        f"Risk level: {report['risk_level']}",
        f"Finding: {report['diagnosis_title']}",
        f"Heart rate: {report['heart_rate']:.0f} bpm ({report['heart_rate_status']})",
        f"Model confidence: {report['confidence']:.1f}%",
        f"Signal quality: {quality['grade']} ({quality['score']}/100)",
        "",
        "Interpretation:",
        report['assessment'],
        "",
        "Actionable steps:"
    ]

    for idx, action in enumerate(report['actions'], start=1):
        lines.append(f"{idx}. {action}")

    lines.append("")
    lines.append("Class probabilities:")
    for class_name, prob in sorted(results['all_probs'].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {class_name}: {prob * 100:.2f}%")

    lines.append("")
    lines.append("Note: This is an AI screening tool and not a medical diagnosis.")

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis('off')
    y = 0.98
    for line in lines:
        ax.text(0.03, y, line, fontsize=10, va='top', ha='left', wrap=True)
        y -= 0.03
        if y < 0.04:
            break

    with PdfPages(buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()

def plot_ecg(signal, fs=100):
    time = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(time, signal, color='#FF4B4B', linewidth=0.8)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title('Your ECG Recording', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig

def main():
    st.title("ECG Analyzer")
    
    model, model_loaded = load_model()
    
    with st.sidebar:
        st.header("⚙ Settings")
        
        if model_loaded:
            st.success("✓ Model loaded")
        else:
            st.error("✗ Model not found")
            st.info("Place `best_model.pth` in project directory")
        
        st.divider()
        
        st.subheader("Data Source")
        source = st.radio(
            "Select input type:",
            ["Apple Watch", "Samsung Watch", "Research Data", "Demo ECG"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        if source != "Demo ECG":
            uploaded_file = st.file_uploader("Upload ECG file", type=['csv'])
        else:
            uploaded_file = None
        
        if source == "Research Data":
            sampling_rate = st.number_input("Sampling Rate (Hz)", min_value=50, max_value=1000, value=100, step=50)
        
        if source == "Demo ECG":
            demo_type = st.selectbox("Demo type:", ["Normal ECG", "Abnormal ECG"])
    
    tabs = st.tabs(["▶ Analyze", "📖 Instructions"])
    
    # ----- ANALYSIS TAB -----
    with tabs[0]:
        if not model_loaded:
            st.warning("⚠ Please place trained model file in the project directory")
            return
        
        signal = None
        fs = 100
        
        # Load ECG data based on source
        if source == "Apple Watch" and uploaded_file:
            try:
                signal, fs = parse_csv(uploaded_file, 'apple')
                st.success(f"✓ Loaded Apple Watch ECG ({len(signal)} samples at {fs} Hz)")
            except Exception as e:
                st.error(f"✗ Error: {str(e)}")
        
        elif source == "Samsung Watch" and uploaded_file:
            try:
                signal, fs = parse_csv(uploaded_file, 'samsung')
                st.success(f"✓ Loaded Samsung Watch ECG ({len(signal)} samples at {fs} Hz)")
            except Exception as e:
                st.error(f"✗ Error: {str(e)}")
        
        elif source == "Research Data" and uploaded_file:
            try:
                signal, fs = parse_csv(uploaded_file, 'generic', sampling_rate)
                st.success(f"✓ Loaded research data ({len(signal)} samples at {fs} Hz)")
            except Exception as e:
                st.error(f"✗ Error: {str(e)}")
        
        elif source == "Demo ECG":
            ecg_type = 'normal' if 'Normal' in demo_type else 'abnormal'
            signal, fs = generate_demo_ecg(ecg_type=ecg_type)
            st.info(f"ℹ Generated {demo_type} ({len(signal)} samples at {fs} Hz)")
        
        # Run analysis if signal is loaded        
        if signal is not None and len(signal) > 0:
            if st.button("▶ Analyze My ECG", type="primary", use_container_width=True):
                with st.spinner("Analyzing your ECG..."):
                    try:
                        quality = assess_signal_quality(signal, fs)
                        results = predict(model, signal, fs)
                        hr = estimate_heart_rate(signal, fs)
                        uncertain_reasons = []

                        if results['top_probability'] < 0.60:
                            uncertain_reasons.append("low prediction confidence")
                        if results['prediction_margin'] < 0.15:
                            uncertain_reasons.append("model classes are closely competing")
                        if quality['grade'] == 'Poor':
                            uncertain_reasons.append("poor ECG signal quality")

                        uncertain = len(uncertain_reasons) > 0
                        
                        st.divider()
                        
                        if uncertain:
                            st.warning("⚠ Analysis Complete - Uncertain result, please retake ECG for better reliability")
                        elif results['is_normal']:
                            st.success("✓ Analysis Complete - Normal ECG")
                        else:
                            st.warning(f"⚠ Analysis Complete - {results['detailed_class']} Detected")

                        report = build_health_report(
                            results,
                            hr,
                            quality,
                            uncertain=uncertain,
                            uncertain_reasons=uncertain_reasons
                        )

                        st.subheader("Personalized ECG Report")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Risk Level", report['risk_level'])
                        col2.metric("Heart Rate", f"{report['heart_rate']:.0f} bpm")
                        col3.metric("Model Confidence", f"{report['confidence']:.1f}%")
                        col4.metric("Signal Quality", f"{report['quality_grade']} ({report['quality_score']}/100)")

                        if quality['issues']:
                            st.markdown("**Signal quality checks:**")
                            for issue in quality['issues']:
                                st.markdown(f"- {issue}")

                        if quality['tips']:
                            st.markdown("**Retake tips:**")
                            for tip in quality['tips']:
                                st.markdown(f"- {tip}")

                        st.markdown(f"**Finding:** {report['diagnosis_title']}")
                        st.markdown(f"**Interpretation:** {report['assessment']}")
                        st.markdown(f"**Heart-rate context:** {report['heart_rate_status']}")
                        st.markdown(f"**Confidence note:** {report['confidence_note']}")

                        st.markdown("**Actionable steps:**")
                        for action in report['actions']:
                            st.markdown(f"- {action}")

                        prob_df = pd.DataFrame(
                            [
                                {
                                    'Class': class_name,
                                    'Probability (%)': round(prob * 100, 2)
                                }
                                for class_name, prob in sorted(
                                    results['all_probs'].items(),
                                    key=lambda item: item[1],
                                    reverse=True
                                )
                            ]
                        )
                        with st.expander("Model class probabilities"):
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)

                        pdf_bytes = generate_pdf_report(report, results, quality, source, fs)
                        st.download_button(
                            "Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"ecg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        gemini_model, gemini_error = initialize_gemini()
                        
                        if not gemini_error and not uncertain:
                            st.subheader("AI Assistant Summary")
                            analysis = analyze_with_gemini(gemini_model, results, signal, fs)
                            if analysis:
                                st.markdown(analysis)
                            else:
                                st.error("AI analysis failed. Please try again.")
                        elif uncertain:
                            st.info("AI Assistant Summary skipped because the result is uncertain. Please retake ECG for a clearer interpretation.")
                        
                        with st.expander(" View ECG Signal"):
                            fig_ecg = plot_ecg(signal[:1000], fs)
                            st.pyplot(fig_ecg)
                            plt.close()
                        
                        st.divider()
                        st.caption("**Important:** This is an AI screening tool for educational purposes only. Not a substitute for professional medical advice. Always consult a healthcare provider.")
                        
                    except Exception as e:
                        st.error(f"✗ Error: {str(e)}")
        
        elif source != "Demo ECG":
            st.info("← Upload an ECG file to get started")

    # ----- INSTRUCTIONS TAB -----    
    with tabs[1]:
        if source == "Apple Watch":
            st.markdown("""
            ### Apple Watch ECG Export
            
            **Record ECG:**
            1. Open ECG app on Apple Watch
            2. Hold finger on Digital Crown for 30 seconds
            
            **Export Data:**
            1. iPhone → Health app → Profile
            2. Export All Health Data
            3. Unzip and find `electrocardiograms/ecg_*.csv`
            
            **Upload:** Use sidebar uploader
            
            • Sampling Rate: 512 Hz
            • Duration: ~30 seconds
            """)
        
        elif source == "Samsung Watch":
            st.markdown("""
            ### Samsung Galaxy Watch ECG Export
            
            **Record ECG:**
            1. Open Samsung Health Monitor
            2. Tap ECG and hold finger
            
            **Export Data:**
            1. Samsung Health → Menu
            2. Download personal data → ECG
            3. Find in Samsung Health folder
            
            **Upload:** Use sidebar uploader
            
            • Sampling Rate: 500 Hz
            • Compatible: Galaxy Watch 4/5/6
            """)
        
        else:
            st.markdown("""
            ### Researcher Guide
            
            **Format:** CSV with single column
            
            **Example:**
            ```
            0.125
            0.132
            0.128
            ```
            
            **Classes:**
            • NORM: Normal ECG
            • MI: Myocardial Infarction
            • STTC: ST/T Change
            • CD: Conduction Disturbance
            • HYP: Hypertrophy
            """)


if __name__ == "__main__":
    main()