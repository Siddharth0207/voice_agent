# Voice Agent for Sales and Chat

A Python-based voice agent designed for sales and chat applications. This project leverages advanced speech recognition, language processing, and real-time audio handling to provide an interactive voice assistant experience.

## Features

- Real-time audio processing via WebSockets
- Speech-to-text and text-to-speech capabilities
- Integration with language models for intelligent responses
- Modular architecture for easy extension

## Project Structure

```text
app.py                  # Main application entry point
routers/                # API and WebSocket route handlers
services/               # Core service modules (audio, language, whisper, etc.)
home.html               # Web interface (if applicable)
requirements.txt        # Python dependencies
pyproject.toml          # Project metadata and build configuration
setup.py                # Setup script for packaging
```

## Installation

1. Clone the repository:

   ```powershell
   git clone <repository-url>
   cd voice_agent
   ```

2. (Optional) Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. Install dependencies using [uv](https://github.com/astral-sh/uv):

   ```powershell
   uv pip install -r requirements.txt
   ```

## Usage

Run the main application:

```powershell
python app.py
```

Access the web interface (if available) at `http://localhost:8000` or as specified in the output.

## Configuration

- Adjust settings in `app.py` or the relevant files in `services/` and `routers/` as needed.
- Place additional documents in `services/document/` for reference or processing.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the terms of the LICENSE file in this repository.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [OpenAI Whisper](https://github.com/openai/whisper)
- Other open-source libraries used in this project
