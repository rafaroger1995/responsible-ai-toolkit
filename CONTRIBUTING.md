# Contributing to Responsible AI Toolkit

Thank you for your interest in contributing. This project aims to democratize responsible AI governance for regulated industries, and community contributions are essential to that mission.

## How to Contribute

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the test suite** to ensure all tests pass: `pytest tests/ -v`
4. **Submit a pull request** with a clear description of the change.

## Development Setup

```bash
git clone https://github.com/rafaroger1995/responsible-ai-toolkit.git
cd responsible-ai-toolkit
pip install -e ".[dev]"
pytest tests/ -v
```

## Guidelines

- Follow existing code style and patterns.
- Add docstrings for all public classes and methods.
- Include regulatory framework alignment tags where applicable.
- Keep the library dependency-light (currently only `numpy`).
- Financial services examples are especially welcome.

## Reporting Issues

Please use GitHub Issues for bug reports and feature requests. Include:
- Python version and OS
- Minimal reproduction steps
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
