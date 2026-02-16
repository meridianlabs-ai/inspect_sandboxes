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

## Quick Start

### Modal

#### Setup

First, create a Modal account and authenticate:

```bash
# Install Modal CLI
pip install modal

# Set up authentication
python3 -m modal setup
```

#### Basic Usage

Use Modal sandbox in your Inspect evaluation:

```python
from inspect_ai import Task, eval
from inspect_ai.solver import generate, system_message

task = Task(
    dataset=[{"input": "What is 2+2?", "target": "4"}],
    solver=[
        system_message("You are a helpful assistant."),
        generate(),
    ],
    sandbox="modal",  # Uses Modal's default Debian + Python 3.11 image
)

eval(task)
```

#### Custom Configuration

**Using a Dockerfile:**

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("modal", "path/to/Dockerfile"),
)
```

**Using Docker Compose:**

```yaml
# compose.yaml
services:
  default:
    image: python:3.11
    # Your service configuration...

# Optional Modal-specific settings
x-inspect_modal_sandbox:
  timeout: 3600              # Maximum sandbox lifetime (seconds)
  idle_timeout: 300          # Idle timeout (seconds)
  block_network: false       # Block network access
  allow_cidr: []             # CIDR allowlist for network access
  cloud: "aws"               # Cloud provider (aws, gcp, oci, auto)
```

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("modal", "compose.yaml"),
)
```

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

## Credits

The Modal sandbox implementation is based on [@anthonyduong9](https://github.com/anthonyduong9)'s work.
