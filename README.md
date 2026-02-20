# Inspect Sandboxes

Collection of sandbox environments for [Inspect AI](https://inspect.ai-safety-institute.org.uk/).

## Available Providers

| Provider | Registry Name | Description | Requirements |
|----------|---------------|-------------|--------------|
| [Modal](https://modal.com) | `modal` | Serverless container platform with GPU support | Modal account (free tier available) |

## Installation

```bash
# Using pip
pip install git+https://github.com/meridianlabs-ai/inspect_sandboxes.git

# Using uv
uv pip install git+https://github.com/meridianlabs-ai/inspect_sandboxes.git
```

## Provider Documentation

- [Modal](src/inspect_sandboxes/modal/README.md)

## Development

```bash
# Install dependencies
make install

# Run tests (skips integration tests)
make test

# Run all tests including integration tests
make test-all

# Run type checking and linting
make check
```
