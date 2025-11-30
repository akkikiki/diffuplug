# Tests

Unit tests for the dllm_plugin.

## Running Tests

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=dllm_plugin --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_plugin_registration.py
```

## Test Structure

- `test_plugin_registration.py` - Tests for plugin registration and patching
- `test_models.py` - Tests for model imports and basic functionality
- `test_sampler.py` - Tests for LLaDA sampler
- `conftest.py` - Pytest configuration and fixtures
