# Contributing

Pull requests welcome.

## Setup

```bash
pip install -e ".[dev]"
```

## Before submitting

```bash
ruff check .
pytest -q
```

## Guidelines

- Write parsers from spec, don't wrap existing libraries
- Keep pure-Python fallbacks for any native extensions
- Add tests for new features
