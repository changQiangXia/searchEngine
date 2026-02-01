# Contributing to NexusMind

Thank you for your interest in contributing to NexusMind! This document provides guidelines for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/nexus-mind.git
cd nexus-mind
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use:
- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking

Run before committing:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/nexus_mind/
```

## Testing

Run the test suite:
```bash
pytest tests/unit -v
```

For GPU tests:
```bash
pytest tests/unit -v -m gpu
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the PROJECT_NOTICE.md with any architecture decisions
3. Ensure all tests pass
4. Update version numbers following [Semantic Versioning](https://semver.org/)
5. Submit PR with clear description of changes

## Commit Message Format

```
type: subject

body (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Build/tools

Example:
```
feat: add concept interpolation CLI command

- Implement SLERP interpolation
- Add CLI command with rich output
- Include unit tests
```

## Questions?

Feel free to open an issue for questions or discussions.

## Code of Conduct

Be respectful and constructive in all interactions.