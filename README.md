# DeepBrief

A video analysis application that helps students, educators, and professionals analyze presentations by combining speech transcription, visual analysis, and AI-powered feedback.

## Features

- **Video Processing**: Support for MP4, MOV, AVI, and WebM formats
- **Speech Analysis**: Automatic transcription with speaking rate and filler word detection
- **Visual Analysis**: Scene detection with frame captioning and quality assessment
- **AI Feedback**: Actionable insights and recommendations for improvement
- **Professional Reports**: Interactive HTML and structured JSON outputs

## Installation

### Prerequisites

- Python 3.11 or higher
- ffmpeg (for video processing)

### Install with uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/michael-borck/deep-brief.git
cd deep-brief

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Install Development Dependencies

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the web interface
deep-brief

# Or use Python module
python -m deep_brief.interface.gradio_app
```

## Development

This project uses modern Python tooling:

- **uv** for fast package management
- **ruff** for formatting and linting
- **basedpyright** for type checking
- **pytest** for testing

### Running Tests

```bash
pytest -v
```

### Code Quality

```bash
# Format and lint
ruff format .
ruff check .

# Type checking
basedpyright

# Run all checks
ruff format . && ruff check . && basedpyright && pytest -v
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the development guidelines in CLAUDE.md.