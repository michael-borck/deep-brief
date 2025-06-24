# DeepBrief Development Guidelines

## Modern Python Toolchain

This project uses a modern Python development toolchain with strict type checking and code quality standards:

### Core Tools
- **uv** - Fast Python package installer and resolver
- **ruff** - Fast Python linter and formatter
- **basedpyright** - Static type checker (Pylance/Pyright fork)
- **pytest** - Testing framework
- **twine** - Package publishing tool for PyPI

### Development Standards
- **Type Hints**: All functions must have complete type annotations
- **No Type Checker Workarounds**: Fix type issues properly, don't use `# type: ignore`
- **Modern Patterns**: Use Pydantic for data validation, no legacy patterns
- **Code Quality**: Run ruff for formatting and linting before commits
- **Testing**: Write tests as we develop, not after
- **Testing Framework**: Use pytest exclusively, NOT unittest
- **Mock Data**: When generating test mock data, use pytest fixtures and patterns
- **Modern Packaging**: Use pyproject.toml exclusively, NO setup.py files

### Commands to Run
```bash
# Install dependencies (CPU-only for development)
uv pip install -e ".[dev]"

# Format and lint code
ruff format .
ruff check .

# Type checking
basedpyright

# Run tests
pytest -v

# Run all checks together
ruff format . && ruff check . && basedpyright && pytest -v

# Build package (modern way)
uv build

# Upload to PyPI (use twine, not uv)
# Test first:
twine upload --repository testpypi dist/*
# Then production:
twine upload dist/*

# Package is live at: https://pypi.org/project/deep-brief/
```

### GPU Migration (Future)
When ready to deploy on GPU-enabled hardware:

```bash
# Install GPU-accelerated dependencies
uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu121

# No code changes needed - PyTorch automatically detects and uses GPU
# Models will automatically use CUDA when available
```

**Migration Notes:**
- All APIs remain identical (torch.device() handles CPU/GPU automatically)
- Whisper, Transformers models automatically detect GPU availability
- Performance improvement without code changes
- Test on CPU first, then deploy to GPU for production speed

### Project Structure
```
src/
├── deep_brief/
│   ├── core/          # Video processing pipeline
│   ├── analysis/      # Speech and visual analysis
│   ├── reports/       # Report generation
│   ├── interface/     # Gradio web interface
│   └── utils/         # Utilities and configuration
config/                # Configuration files
tests/                 # Test files (mirror src structure)
docs/                  # Documentation
```

### Dependencies
- **Pydantic** for data validation and settings management
- **FastAPI** patterns for dependency injection where applicable
- **Modern async/await** patterns where beneficial
- **Pathlib** instead of os.path
- **Rich** for better CLI output and progress bars

### Code Quality Requirements
1. All code must pass basedpyright without errors
2. All code must be formatted with ruff
3. All functions must have docstrings
4. All public APIs must have comprehensive type hints
5. Tests must be written for all new functionality
6. No deprecated patterns or workarounds