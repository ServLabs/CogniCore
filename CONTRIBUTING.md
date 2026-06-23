# Contributing to CogniCore

Thank you for your interest in contributing to CogniCore! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search [GitHub Issues](https://github.com/ServLabs/CogniCore/issues) to see if the bug has already been reported.
2. **Create a new issue** - If not found, open a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version)
   - Relevant logs or screenshots

### Suggesting Features

1. Open a [GitHub Issue](https://github.com/ServLabs/CogniCore/issues/new) with the `enhancement` label.
2. Describe the feature and its use case.
3. Explain why it would benefit the project.

### Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Follow the coding style** (see below).
3. **Write tests** for new functionality.
4. **Update documentation** if needed.
5. **Submit a pull request** with a clear description.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CogniCore.git
cd CogniCore

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Run tests
COGNICORE_DATA_DIR=/tmp/cognicore pytest
```

## Coding Style

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all functions
- Maximum line length: 100 characters
- Use `black` for formatting, `isort` for imports

```bash
# Format code
black .
isort .

# Lint
ruff check .
mypy .
```

### Project Conventions

- **Imports**: Follow the layered export pattern (see `.windsurf/workflows/cognicore-style.md`)
- **Dataclasses**: Use `@dataclass(frozen=True)` for immutable data
- **Singletons**: Use module-level `_instance` + `get_*()` function
- **Datetime**: Always use `datetime.now(timezone.utc)`, never `datetime.utcnow()`
- **Config**: Domain-specific values from `config.py`, not hardcoded enums

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add contrastive learning algorithm
fix: resolve memory leak in FAISS wrapper
docs: update API reference for scheduled endpoints
test: add unit tests for recall engine
refactor: consolidate audit into core module
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `style`

## Pull Request Process

1. Ensure all tests pass: `pytest`
2. Update documentation if needed
3. Add entry to CHANGELOG.md (if applicable)
4. Request review from maintainers
5. Address review feedback
6. Maintainer will merge when approved

## Architecture Overview

Before contributing, familiarize yourself with the project structure:

```
CogniCore/
├── config.py       # 3-tier env var config (Tier 3 > 2 > 1)
├── interfaces/     # External APIs (admin, service, chat, scheduled)
├── memory/         # Memory layer (9 types + MML + learning + maintenance)
├── control/        # Brain control (CEN, DMN, salience, governor)
├── response/       # Response pipeline + sub-agents
├── connectors/     # External connections (AI, data, sandbox)
├── observability/  # Telemetry (audit, tracing, analytics/DuckDB, evals)
└── prompts/        # LLM prompt templates
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Testing

- Write tests for all new functionality
- Place tests in `tests/` directory
- Use pytest fixtures from `tests/conftest.py`
- Aim for meaningful coverage, not 100%

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=.
```

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Questions?

- Open a [GitHub Discussion](https://github.com/ServLabs/CogniCore/discussions)
- Check existing documentation
- Review closed issues for similar questions

Thank you for contributing! 🎉
