# Inspect Sandboxes

The `inspect-sandboxes` Python package provides cloud sandbox environments for [Inspect AI](https://inspect.aisi.org.uk/) evaluations. Each sample runs in a managed sandbox on your provider of choice while the Inspect process runs locally on your machine.

This lets you run evaluations without a local Docker daemon or your own Kubernetes cluster, and scale to many concurrent samples beyond what fits on a single machine.

For sandbox concepts and the `SandboxEnvironment` API, see the [Inspect AI sandboxing guide](https://inspect.aisi.org.uk/sandboxing.html).

## Providers

| Provider | Registry Name | Multi-service compose | Requirements |
|----------|---------------|-----------------------|--------------|
| [Daytona](https://www.daytona.io) | `daytona` | Yes (DinD) | Daytona account + API key |
| [Modal](https://modal.com) | `modal` | No ([why?](https://meridianlabs-ai.github.io/inspect_sandboxes/modal.html#modal-limitations)) | Modal account |

## Installation

```bash
# Using pip
pip install inspect-sandboxes

# Using uv
uv pip install inspect-sandboxes
```

## Quick start (Modal)

Authenticate with Modal (one-time):

```bash
python3 -m modal setup
```

Run a minimal evaluation:

```python
from inspect_ai import Task, eval
from inspect_ai.solver import generate

task = Task(
    dataset=[{"input": "What is 2+2?", "target": "4"}],
    solver=[generate()],
    sandbox="modal",
)

eval(task)
```

For each sample in the dataset, a fresh Modal sandbox is provisioned, used to run the solver, and terminated when the sample completes. Since no image is specified here, Modal's default sandbox image is used (Debian Linux with Python matching your local interpreter's minor version) — place a `Dockerfile` or `compose.yaml` alongside the task (auto-detected) or pass one explicitly via `sandbox=("modal", "path/to/Dockerfile")` to provide your own.

See the [Modal](https://meridianlabs-ai.github.io/inspect_sandboxes/modal.html) provider page for full configuration options.

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
