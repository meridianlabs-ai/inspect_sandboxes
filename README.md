# Inspect Sandboxes

The `inspect-sandboxes` Python package provides cloud sandbox environments for [Inspect AI](https://inspect.aisi.org.uk/) evaluations. Each sample runs in a managed sandbox on your provider of choice (Daytona, E2B, Modal, or Runloop) while the Inspect process runs locally on your machine.

This lets you run evaluations without a local Docker daemon or your own Kubernetes cluster, and scale to many concurrent samples beyond what fits on a single machine.

## Documentation

See the [documentation](https://meridianlabs-ai.github.io/inspect_sandboxes/) for installation, a provider quick start, and full configuration options for each provider.

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
