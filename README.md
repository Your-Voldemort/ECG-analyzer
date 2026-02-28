# Abnormal-ECG-signal-classification

A deep learning-based ECG analyzer that classifies multi-class ECG signals using Convolutional Deep Neural Networks (CDNNs). This project incorporates advanced explainability techniques including SHAP (SHapley Additive exPlanations) values and Grad-CAM (Gradient-weighted Class Activation Mapping) to provide interpretable results for cardiovascular condition detection.


This application analyzes electrocardiogram (ECG) signals and classifies them into five categories:
- **NORM**: Normal ECG
- **MI**: Myocardial Infarction (Heart Attack)
- **STTC**: ST-T Wave Changes
- **CD**: Conduction Disturbance
- **HYP**: Cardiac Hypertrophy

The model uses Recurrence Plot Matrix (RPM) representations of ECG signals for feature extraction and classification.

## Project Structure

```
Abnormal-ECG-signal-classification/
├── streamlit_app/              # Deployment & Web Interface
│   ├── app.py                  # Main Streamlit application
│   ├── model.py                # Model architecture
│   ├── best_model.pth          # Trained model weights (use git LFS to track it)
│   └── .streamlit/
│       └── secrets.toml        # (add .toml in gitignore)
├── model.py                    # Original Training Code
├── requirements.txt
├── .gitignore               
├── .gitattributes              # Git LFS configuration
└── README.md
```


**Important Note:**
- **`streamlit_app/`**: Contains the web application for ECG analysis and inference
- **`model.py` (root)**: Original model training and development code
- The model architecture in `streamlit_app/model.py` is adapted for deployment and inference only

## Features

- **Multi-class ECG Classification**: Detects 5 different cardiovascular conditions
- **Multiple Input Methods**:
  - Upload ECG data files (CSV format)
  - Generate demo ECG signals for testing
  - Support for Apple Watch and Samsung Health exports
- **Real-time Analysis**: Instant ECG signal processing and classification
- **Visual Feedback**: Interactive ECG waveform visualization
- **AI-Powered Insights**: Optional Gemini AI integration for detailed medical explanations
- **Heart Rate Estimation**: Automatic calculation from ECG signals
- **User-Friendly Interface**: Built with Streamlit for easy interaction

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Git LFS (for model files)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Abnormal-ECG-signal-classification.git
   cd Abnormal-ECG-signal-classification
   ```

2. **Install Git LFS** (if not already installed)
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

#### Optional: Gemini AI Integration

To enable AI-powered medical insights:

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. Set the API key as an environment variable:
   
   **Windows (Command Prompt):**
   ```cmd
   set GEMINI_API_KEY=your_api_key_here
   ```
   
   **Windows (PowerShell):**
   ```powershell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
   
   **macOS/Linux:**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

3. Or create a `.streamlit/secrets.toml` file:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

## Running the Application

### Local Development

1. **Navigate to the streamlit_app directory**
   ```bash
   cd streamlit_app
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

### Deployment on Streamlit Cloud

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Deploy ECG analyzer"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app/app.py`
   - Click "Deploy"

3. **Add secrets** (if using Gemini AI)
   - In Streamlit Cloud dashboard, go to app settings
   - Add `GEMINI_API_KEY` in the Secrets section

## Usage

### 1. Upload ECG Data

**Supported Formats:**
- CSV files with ECG signal data
- Apple Watch ECG exports
- Samsung Health ECG exports

**Steps:**
1. Click "Upload ECG File" in the sidebar
2. Select your device type (Generic, Apple Watch, or Samsung Health)
3. Set the sampling rate (default: 100 Hz)
4. Upload your CSV file

### 2. Generate Demo ECG

For testing purposes:
1. Select "Generate Demo ECG" in the sidebar
2. Choose ECG type (Normal or Abnormal)
3. Set duration (5-60 seconds)
4. Click "Generate"

### 3. View Results

The application displays:
- **ECG Waveform**: Visual representation of your ECG signal
- **Classification Result**: Detected condition with confidence level
- **Probability Distribution**: Likelihood of each condition
- **AI Explanation**: Detailed medical interpretation (if Gemini is enabled)
- **Heart Rate**: Estimated BPM from the signal


## Model Architecture

The model uses:
- **Input**: Recurrence Plot Matrix (RPM) representations of ECG signals
- **Architecture**: Feature Extraction Module with convolutional layers
- **Output**: 5-class classification (NORM, MI, STTC, CD, HYP)
- **Signal Length**: 1000 samples at 100 Hz (10 seconds)

See `requirements.txt` for complete list of requirements.

## Troubleshooting

### Model File Not Found

If you get "best_model.pth not found":

1. **Ensure Git LFS is installed and initialized:**
   ```bash
   git lfs install
   git lfs pull
   ```

2. **Verify the model file exists:**
   ```bash
   ls -lh streamlit_app/best_model.pth
   ```

3. **Check Git LFS tracking:**
   ```bash
   git lfs ls-files
   ```

### Deployment Issues on Streamlit Cloud

1. Ensure `best_model.pth` is pushed with Git LFS
2. Verify `.gitattributes` contains: `*.pth filter=lfs diff=lfs merge=lfs -text`
3. Check model file shows "Stored with Git LFS" badge on GitHub
4. Trigger a redeployment in Streamlit Cloud dashboard

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

## Disclaimer

This application is for **educational and research purposes only**. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## License

GNU GENERAL PUBLIC LICENSE. See <https://www.gnu.org/licenses/>.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- ECG dataset sources: https://physionet.org/content/ptb-xl/1.0.1/
- Research papers : Wei Zeng, Liangmin Shan, Chengzhi Yuan, Shaoyi Du,
Advancing cardiac diagnostics: Exceptional accuracy in abnormal ECG signal classification with cascading deep learning and explainability analysis
https://doi.org/10.1016/j.asoc.2024.112056.
(Some of the strategies used in the project were inspired by this paper)
---

**Note**: Make sure to install Git LFS before cloning to properly download the model files.