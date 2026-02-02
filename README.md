# ASL to LLM - Rule-based Sign Language Detection

A simple hobby project that detects hand gestures using MediaPipe and sends text to an n8n chat agent.

## Features

- Real-time hand gesture detection
- 5 basic gestures: Thumbs Up, Thumbs Down, Open Palm, Fist, Pointing
- Automatic text buffering with 3-second timeout
- Sends to n8n webhook for LLM processing
- Scrolling ticker for LLM responses

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run:
```bash
python main.py
```

## Gesture Mappings

| Gesture | Word |
|----------|-------|
| Thumbs up | yes |
| Thumbs down | no |
| Open palm | hello |
| Fist | stop |
| Pointing | what |

## Usage

- Show gestures to camera
- Words buffer automatically
- After 3 seconds of inactivity, sends sentence to LLM
- LLM response scrolls across top of screen
- Press 'q' to quit

## Configuration

Edit `config.py` to change:
- Timeout duration
- Webhook URL
- Gesture mappings
