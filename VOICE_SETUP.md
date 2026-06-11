# 🎤 Voice Input Setup Guide

## Installation

### 1. Install Python Dependencies
```bash
pip install SpeechRecognition PyAudio pocketsphinx
```

### 2. Install System Dependencies

#### Windows:
```bash
# Install PyAudio (if pip install fails)
pip install pipwin
pipwin install pyaudio
```

#### macOS:
```bash
brew install portaudio
pip install pyaudio
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install python3-pyaudio
sudo apt-get install portaudio19-dev
pip install pyaudio
```

## Usage

### Quick Voice Input:
1. Click "🎤 Click to Speak" button
2. Speak your query clearly
3. Wait for recognition
4. Click "✅ Use This Text" to process

### Advanced Voice Input:
1. Expand "🔧 Advanced Voice Options"
2. Click "🎤 Start Recording"
3. Speak your query
4. Click "⏹️ Stop Recording"
5. Review recognized text
6. Click "✅ Use This Text" to process

## Voice Commands Examples

- "Analyze TCS"
- "Compare HDFC and ICICI"
- "Risk analysis of RELIANCE"
- "Backtest TCS with RSI strategy"
- "SWOT analysis of INFY"
- "Capital needed for 50000 rupees profit from WIPRO"

## Troubleshooting

### Microphone Issues:
- Check microphone permissions
- Ensure microphone is not muted
- Try different microphone if available

### Recognition Issues:
- Speak clearly and slowly
- Reduce background noise
- Try different phrases
- Use manual typing as fallback

### Installation Issues:
- Update pip: `pip install --upgrade pip`
- Install Visual C++ Build Tools (Windows)
- Use conda instead of pip if needed

## Features

- **Multiple Recognition Engines**: Google Speech Recognition + Sphinx
- **Noise Reduction**: Automatic ambient noise adjustment
- **Fallback Options**: Manual typing when voice fails
- **Real-time Feedback**: Visual indicators during recording
- **Error Handling**: Graceful handling of recognition failures
