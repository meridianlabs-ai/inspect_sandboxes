# Modal Sandbox

Serverless container sandbox for [Inspect AI](https://inspect.ai-safety-institute.org.uk/) using [Modal](https://modal.com).

## Setup

Create a Modal account and authenticate:

```bash
pip install modal
python3 -m modal setup
```

## Usage

### Default image

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

### Dockerfile

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("modal", "path/to/Dockerfile"),
)
```

### Docker Compose

```yaml
# compose.yaml
services:
  default:
    image: python:3.11
    # Your service configuration...

# Optional Modal-specific settings
x-modal:
  timeout: 3600        # Maximum sandbox lifetime (seconds)
  idle_timeout: 300    # Idle timeout (seconds)
  block_network: false # Block network access
  cidr_allowlist: []   # CIDR allowlist for network access
  cloud: "aws"         # Cloud provider (aws, gcp, oci, auto)
  gpu: "A10G"          # GPU type (A10G, A100, T4, H100, ANY, A10G:2, ...)
```

GPU can also be requested via standard Compose deploy resources (defaults to `ANY` type):

```yaml
services:
  default:
    image: pytorch/pytorch:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
```

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("modal", "compose.yaml"),
)
```

## Credits

Based on [@anthonyduong9](https://github.com/anthonyduong9)'s work.
