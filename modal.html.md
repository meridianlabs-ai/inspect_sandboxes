# Modal – Inspect Sandboxes

Serverless container sandbox for [Inspect AI](https://inspect.aisi.org.uk/) using [Modal](https://modal.com).

## Setup

Create a Modal account and authenticate:

``` bash
python3 -m modal setup
```

## Usage

### Default image

A minimal eval that uses Modal’s default sandbox image — no Dockerfile or compose file provided.

``` python
from inspect_ai import Task, eval
from inspect_ai.solver import generate, system_message

task = Task(
    dataset=[{"input": "What is 2+2?", "target": "4"}],
    solver=[
      system_message("You are a helpful assistant."),
      generate(),
    ],
    sandbox="modal",  # Uses Modal's default image (Debian Linux + local Python version)
)

eval(task)
```

> **NOTE: Note**
>
> The default image is only used when no config is specified AND no `Dockerfile` / `compose.yaml` / `compose.yml` / `docker-compose.yaml` / `docker-compose.yml` is present in the task’s source directory — if one is found there, it is auto-detected and used instead.

### Dockerfile

Build the sandbox image from a local Dockerfile.

``` python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("modal", "path/to/Dockerfile"),
)
```

### Docker Compose

Configure the sandbox via a Docker Compose file. Modal-specific settings go under the top-level `x-modal` key — see [Configuration](#configuration).

``` yaml
# compose.yaml
services:
  default:
    image: python:3.11
    # Your service configuration...
```

``` python
task = Task(
    dataset=[...],
    solver=[...],
    sandbox=("modal", "path/to/compose.yaml"),
)
```

GPU can be requested via standard Compose deploy resources (defaults to `ANY` type). `x-modal.gpu` takes precedence if both are set:

``` yaml
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

## Configuration

The top-level `x-modal` key on a compose file passes settings to [`modal.Sandbox.create`](https://modal.com/docs/reference/modal.Sandbox). Values here override the corresponding compose defaults (e.g. `x-modal.block_network` overrides Compose `network_mode`).

| Setting | Type | Default | Description |
|----|----|----|----|
| `timeout` | int (seconds) | `86400` (24 h) | Maximum sandbox lifetime. Modal’s own default is `300`; `inspect_sandboxes` raises it to 24 hours. |
| `idle_timeout` | int (seconds) | `None` | Seconds of inactivity before the sandbox is terminated. `None` means no idle timeout. |
| `gpu` | str | `None` | GPU type, e.g. `"A10G"`, `"A100"`, `"T4"`, `"H100"`, `"ANY"`, or with a count like `"A10G:2"`. Overrides the GPU type derived from compose `deploy.resources`. |
| `cloud` | str | `None` | Cloud provider: `"aws"`, `"gcp"`, `"oci"`, or `"auto"`. |
| `region` | str or list\[str\] | `None` | Region or regions to run the sandbox on. |
| `block_network` | bool | `False` | Block all network access from the sandbox. |
| `cidr_allowlist` | list\[str\] | `None` | CIDR blocks the sandbox is allowed to reach. `None` allows all. |
| `secrets` | str or list\[str\] | `None` | Name(s) of Modal secrets to inject as environment variables. |
| `pty` | bool | `False` | Enable a pseudo-TTY. |
| `encrypted_ports` | list\[int\] | `[]` | Ports to tunnel into the sandbox over TLS. |
| `h2_ports` | list\[int\] | `[]` | Encrypted ports to tunnel via HTTP/2. |
| `unencrypted_ports` | list\[int\] | `[]` | Ports to tunnel without encryption. |
| `custom_domain` | str | `None` | Custom subdomain parent for sandbox connections. |
| `verbose` | bool | `False` | Enable verbose Modal logging. |

Example:

``` yaml
x-modal:
  timeout: 3600
  idle_timeout: 300
  cloud: "aws"
  gpu: "A10G"
  secrets: ["openai-api-key"]
```

Unsupported Modal parameters (require SDK objects, not compose-representable): `network_file_systems`, `volumes`, `proxy`.

## Finding sandboxes

Every sandbox created by `inspect_sandboxes` is named and tagged so you can locate it in the Modal dashboard for debugging, audit, or manual cleanup.

**Sandbox names** follow `inspect-{task_id}-{sample_id}-{hex}` (e.g. `inspect-my_eval-42-a1b2c3d4`). The 8-character hex suffix guarantees uniqueness — Modal requires sandbox names to be unique within an app, and all `inspect_sandboxes` sandboxes share one app. If `task_id` or `sample_id` is unavailable, that segment is dropped; if both are unavailable, the name is just `inspect-{hex}`. The `sample_id` segment requires `inspect-ai >= 0.3.211` ([PR \#3619](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3619)); on older versions it’s silently omitted.

**Modal app**: All sandboxes are created under the app named `inspect_modal_sandbox` (visible in your Modal dashboard).

**Tags** applied to every sandbox:

- `created_by: inspect-ai` — identifies sandboxes created by this package.

**Bulk cleanup** via the Inspect CLI — finds and terminates every sandbox tagged `created_by: inspect-ai`:

``` bash
inspect sandbox cleanup modal                   # terminate all
inspect sandbox cleanup modal <sandbox-id>      # terminate one
```

## Notes

- **Network mode**: Docker Compose `network_mode` is automatically translated to Modal’s `block_network`. `network_mode: none` blocks all network access; any other value (e.g. `bridge`) allows it. `x-modal.block_network` takes precedence if set.

## Limitations

- **Single service only**: Multi-service compose files raise a `ValueError`. Modal sandboxes run on [gVisor](https://gvisor.dev/), a user-space kernel that restricts the low-level network operations Docker needs to connect containers together — specifically, creating virtual ethernet pairs between containers and setting up NAT rules. Modal’s alpha `experimental_options={"enable_docker": True}` doesn’t currently lift these restrictions, so compose services can’t share a private network or reach each other by service name. For multi-service workloads, use the [Daytona provider](./daytona.html.md), which supports multi-service compose via Docker-in-Docker.
