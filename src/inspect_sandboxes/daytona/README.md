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

### Single-service Docker Compose

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
```

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("daytona", "compose.yaml"),
)
```

### Multi-service Docker Compose (DinD)

When a compose file defines more than one service, the provider automatically
uses Docker-in-Docker: a single Daytona sandbox runs a Docker daemon, and
services are brought up via `docker compose` inside it. Each service is
exposed as a separate `SandboxEnvironment`.

```yaml
# compose.yaml
services:
  default:
    image: python:3.12
    x-default: true
  helper:
    image: redis:7
```

```python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("daytona", "compose.yaml"),
)

# In a solver, access services by name:
default_env = sandbox()          # the x-default service
helper_env = sandbox("helper")
```

**Default service selection** (priority): `x-default: true` > service named
`"default"` or `"main"` > first service in the file.

**Resources**: Per-service resources are summed across all services plus 1 CPU
and 1 GiB overhead for the Docker daemon. Ensure the total fits within your
Daytona per-sandbox limits.

**DinD image**: The DinD sandbox uses `docker:28.3.3-dind` as the base image.

**DinD snapshots**: The provider auto-creates a Daytona snapshot for the DinD
base image. You can also provide a pre-created snapshot:

```yaml
x-daytona:
  snapshot: "my-dind-snapshot"
```

**Image caching**: Each DinD sandbox gets a fresh Docker daemon with no image
cache. `docker compose build` rebuilds from scratch every sample. For faster
startup, pre-build your images and push them to a registry, then use `image:`
instead of `build:` in your compose file:

```yaml
services:
  default:
    image: myregistry/myapp:latest   # pulled, not built
  verifier:
    image: myregistry/verifier:latest
```

*Daytona does not support snapshotting a running sandbox (to capture Docker's
layer cache) or shared volumes across sandboxes, so registry-based caching is
the recommended approach.*

### x-daytona extensions

```yaml
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
  snapshot: "my-snapshot"      # Pre-created snapshot (skips image build)
  resources:                   # Sandbox-level resource overrides
    cpu: 4                     #   (DinD: overrides per-service aggregation)
    memory: 8
    gpu: 1
  env_vars:                    # Extra env vars (single-service: merged with service
    EXTRA_VAR: "value"         #   environment; DinD: set on the VM, not compose services)
  labels:                      # Custom labels, merged with inspect's own labels
    team: "research"
```

## Notes and Limitations

- **Default user**: The default sandbox user is `daytona` (not root), with passwordless `sudo`.
- **user**: The `user` parameter to `exec()` is supported via `sudo -u` in single-service mode and `docker compose exec --user` in DinD mode. Numeric UIDs are supported. Requires `sudo` with passwordless access (configured by default).
- **stdin**: The `input` parameter to `exec()` is supported via input redirection from a temp file. POSIX-compatible.
- **stderr**: The Daytona API returns combined stdout+stderr; `stderr` is always empty.
- **Architecture**: Daytona runners use `linux/amd64`. arm64-only images are not supported.
- **Network**: Outbound internet depends on your Daytona subscription tier. Tiers 1-2 restrict to essential services (Docker Hub, npm, PyPI, GitHub, AI providers). Tiers 3-4 have full internet. Docker Compose `network_mode` is translated to Daytona's `network_block_all`; `x-daytona.network_block_all` takes precedence. For DinD, the VM always has network enabled (for `docker pull`); service-level isolation is via compose `network_mode`.
- **DinD startup latency**: Docker daemon boot + image pulls + compose up can take 30s+. Use pre-built images and snapshots to reduce this.