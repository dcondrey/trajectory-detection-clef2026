# Contributing

Thank you for your interest in contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/sense-clef2026.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install in development mode: `pip install -e ".[dev]"`
5. Create a feature branch: `git checkout -b feature/your-feature`

## Development

### Running Tests

```bash
make test
```

### Code Style

- Follow PEP 8
- Use type hints for function signatures
- Use the `logging` module instead of `print()`
- Run linting: `make lint`

### Commit Messages

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code restructuring

## Submitting Changes

1. Ensure all tests pass: `make test`
2. Ensure code is formatted: `make lint`
3. Push to your fork and submit a pull request
4. Describe your changes clearly in the PR description

## Reporting Issues

- Use the GitHub issue tracker
- Include steps to reproduce, expected behavior, and actual behavior
- Include Python version and OS information

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
