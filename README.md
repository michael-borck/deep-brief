# DeepBrief

[![PyPI version](https://badge.fury.io/py/deep-brief.svg)](https://pypi.org/project/deep-brief/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A video analysis application that helps students, educators, and professionals analyze presentations by combining speech transcription, visual analysis, and AI-powered feedback.

> **Status**: Phase 1 MVP in development. Core infrastructure complete, video processing pipeline in progress.

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

### Option 1: Install from PyPI (recommended for users)

```bash
pip install deep-brief
```

### Option 2: Install from source (for development)

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/michael-borck/deep-brief.git
cd deep-brief

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Installing ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Quick Start

```bash
# Show available commands
deep-brief --help

# Check version
deep-brief version

# Launch web interface (coming soon)
deep-brief analyze

# Analyze a specific video (CLI mode - coming soon)
deep-brief analyze video.mp4 --output ./reports
```

**Current Status**: The CLI framework is complete. Video processing features are in active development.

## Development

This project uses modern Python tooling and follows strict quality standards:

- **uv** for fast package management
- **ruff** for formatting and linting
- **basedpyright** for strict type checking
- **pytest** for testing with coverage
- **pyproject.toml** for all configuration (no setup.py)

### Development Setup

```bash
# Clone and setup
git clone https://github.com/michael-borck/deep-brief.git
cd deep-brief
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify setup
deep-brief --help
pytest -v
```

### Code Quality Standards

```bash
# Format code
ruff format .

# Lint code  
ruff check .

# Type checking (strict mode)
basedpyright

# Run tests with coverage
pytest -v

# Run all quality checks
ruff format . && ruff check . && basedpyright && pytest -v
```

### Project Structure

```
src/deep_brief/          # Main package
├── core/                # Video processing pipeline
├── analysis/            # Speech and visual analysis  
├── reports/             # Report generation
├── interface/           # Gradio web interface
└── utils/               # Configuration and utilities

tests/                   # Test suite (mirrors src structure)
docs/                    # Documentation and specs
tasks/                   # Development task tracking
config/                  # Configuration files
```

### Current Development Phase

- ✅ **Phase 0**: Project setup, packaging, PyPI publication
- 🚧 **Phase 1**: Core video processing pipeline (in progress)
- 📋 **Phase 2**: Enhanced analysis features
- 📋 **Phase 3**: Advanced AI features

See `tasks/tasks-prd-phase1-mvp.md` for detailed task tracking.

## Links

- **PyPI**: https://pypi.org/project/deep-brief/
- **GitHub**: https://github.com/michael-borck/deep-brief
- **Documentation**: Coming soon

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the development guidelines in `CLAUDE.md` for our coding standards and toolchain requirements.