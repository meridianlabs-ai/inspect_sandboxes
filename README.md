# Inspect Sandboxes

Collection of sandbox environments for [Inspect AI](https://inspect.ai-safety-institute.org.uk/).

## Available Providers

| Provider | Registry Name | Description | Requirements |
|----------|---------------|-------------|--------------|
| [Daytona](https://www.daytona.io) | `daytona` | Cloud sandbox runtime | Daytona account + API key |
| [Modal](https://modal.com) | `modal` | Serverless container platform | Modal account |

## Installation

```bash
# Using pip
pip install inspect-sandboxes

# Using uv
uv pip install inspect-sandboxes
```

## Provider Documentation

- [Daytona](src/inspect_sandboxes/daytona/README.md)
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
