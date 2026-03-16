# Daytona Sandbox

Cloud development environment sandbox for [Inspect AI](https://inspect.ai-safety-institute.org.uk/) using [Daytona](https://www.daytona.io/).

## Setup

Create a Daytona account and set your API key:

```bash
export DAYTONA_API_KEY=your_api_key
```

## Usage

### Default snapshot

```python
from inspect_ai import Task, eval
from inspect_ai.solver import generate, system_message

task = Task(
    dataset=[{"input": "What is 2+2?", "target": "4"}],
    solver=[
        system_message("You are a helpful assistant."),
        generate(),
    ],
    sandbox="daytona",  # Uses Daytona's default snapshot
)

eval(task)
```

### Dockerfile

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("daytona", "path/to/Dockerfile"),
)
```

### Docker Compose

```yaml
# compose.yaml
services:
  default:
    image: python:3.12
    environment:
      - MY_VAR=hello
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4g
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1

# Optional Daytona-specific settings
x-daytona:
  auto_stop_interval: 10       # Minutes of inactivity before auto-stop (0 = disabled)
  auto_archive_interval: 60    # Minutes before stopped sandbox auto-archives
  auto_delete_interval: 1440   # Minutes before stopped sandbox auto-deletes
  network_block_all: false     # Block all network access
  network_allow_list: "10.0.0.0/8,192.168.0.0/16"  # Comma-separated CIDR allowlist
  language: "python"           # Hint for language-aware features
  os_user: "ubuntu"            # OS user for commands (overrides service-level user)
  public: false                # Publicly accessible sandbox
  ephemeral: true              # Auto-delete when stopped
  timeout: 60.0                # Seconds to wait for sandbox creation
  env_vars:                    # Extra env vars, merged with service-level environment
    EXTRA_VAR: "value"
  labels:                      # Custom labels, merged with inspect's own labels
    team: "research"
```

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("daytona", "compose.yaml"),
)
```

## Notes

- **stderr**: The Daytona API returns a single combined output; `stderr` is always empty and all output appears in `stdout`.
- **stdin**: The `input` parameter to `exec()` is supported by uploading a temp file and piping it into the command.
- **user**: The `user` parameter to `exec()` is not supported; use `os_user` in `x-daytona` to set the OS user.
