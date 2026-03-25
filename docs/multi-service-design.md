# Multi-Service Compose Support for Daytona and Modal

## Problem

Our Daytona and Modal sandbox providers only support single-service compose files today. The inspect_ai contract supports `dict[str, SandboxEnvironment]` from `sample_init()` — one entry per service — enabling evals that need multiple cooperating services.

## Why DinD (Docker-in-Docker)

We investigated two approaches:

**Option A: One sandbox per service** — create separate Daytona/Modal sandboxes for each compose service and connect them via networking.

**Option B: DinD** — create a single sandbox running a Docker daemon, upload compose files, and run `docker compose up` inside it.

We chose **Option B** because neither Daytona nor Modal support inter-sandbox networking. Sandboxes are fully isolated — no shared networks, DNS, or IP communication between them. This is by design for security. DinD gives us real Docker Compose networking inside a single sandbox, which is also the approach Harbor uses for Daytona.

## How DinD Works (Harbor's Approach)

Harbor's Daytona DinD implementation:

1. Creates a Daytona sandbox from `docker:28.3.3-dind` (Alpine + Docker daemon)
2. Manually starts `dockerd` inside the sandbox and polls `docker info` until ready
3. Uploads compose files to the sandbox filesystem
4. Runs `docker compose build` then `docker compose up -d` inside the sandbox
5. Polls until the "main" container is running
6. Executes all commands via `docker compose exec -T main bash -lc <command>`
7. File transfers are two-hop: local ↔ Daytona SDK ↔ sandbox ↔ `docker compose cp` ↔ container
8. Cleanup: `docker compose down --remove-orphans`, then delete the sandbox

Harbor uses a strategy pattern to auto-detect compose mode — if `docker-compose.yaml` exists, use DinD; otherwise use direct single-container mode.

## Inspect Docker vs Harbor DinD

| | inspect_ai Docker | Harbor DinD |
|---|---|---|
| **Per-service environments** | Yes — one `SandboxEnvironment` per service | No — only "main" is exposed |
| **Service routing** | `docker compose exec <service> <cmd>` | Hardcoded to `docker compose exec main ...` |
| **Service discovery** | `docker compose config` → enumerates all services | Not enumerated |
| **Default selection** | `x-default: true` or service named "default" (error if neither) | Always "main" |
| **File I/O** | Single-hop (host ↔ container) | Two-hop (host ↔ sandbox ↔ container) |
| **Docker daemon** | Uses host Docker | Must start and manage dockerd inside sandbox |
| **Networking** | Compose creates network automatically | Same (compose networking works inside DinD) |
| **Cleanup** | `docker compose down --volumes` | `docker compose down` → delete sandbox |

## Gaps to Bridge

To implement inspect_ai-compliant multi-service support on Daytona via DinD, we need to:

1. **Service verification** — we know expected services from the compose config; after `compose up`, verify they're actually running via `docker compose ps`
2. **Create one `SandboxEnvironment` per service**, each storing a service name and routing exec/read/write to that service via `docker compose exec <service> ...`
3. **Discover working directory per service** at init time (via `exec <service> pwd`, defaulting to `/` on failure)
4. **Default service selection** matching inspect_ai's logic (`x-default: true` / "default" name)
5. **Accept two-hop file transfer overhead** (unavoidable with DinD on a remote sandbox)
6. **Manage Docker daemon lifecycle** (start, poll, cleanup) inside the sandbox

The DinD machinery (daemon management, compose file upload, polling) is proven by Harbor. The new work is the per-service `SandboxEnvironment` routing layer on top.

## Proposed Design (Daytona first, Modal later)

### Architecture

- **Separate class**: `DaytonaSandboxEnvironment` (registered `@sandboxenv`) handles lifecycle and detection. A new `DaytonaDinDServiceEnvironment` implements per-service exec/read/write by routing through `docker compose exec <service>`.
- **Detection**: `parse_compose_yaml(config, multiple_services=True)` — if >1 service, take the DinD path. Single-service compose files continue using the current direct path.
- **New file**: DinD logic in `daytona/_dind.py`.

### Shared state: DaytonaDinDProject

All per-service environments for one sample share a `DaytonaDinDProject` (analogous to inspect_ai's `ComposeProject`):

```python
@dataclass
class DaytonaDinDProject:
    sandbox: AsyncSandbox          # The DinD sandbox
    project_name: str              # docker compose -p <name>
    compose_path: str              # Path to compose file inside sandbox
    services: list[str]            # Running service names
```

### DinD startup sequence

1. Create Daytona sandbox from `docker:28.3.3-dind` with `network_block_all=False` (daemon needs network for image pulls)
2. Start dockerd: `dockerd-entrypoint.sh dockerd &`
3. Poll `docker info` until ready (up to 60s)
4. Upload the full build context directory to sandbox (compose file, Dockerfiles, volumes, and any other files services need)
5. `docker compose build` then `docker compose up -d`
6. Verify running services via `docker compose ps` against expected services from compose config (fail if any service didn't start)
7. Discover working directory per service via `docker compose exec <service> pwd` (default to `/` on failure)
8. Create `DaytonaDinDServiceEnvironment` per service, order dict with default first

### Per-service routing

```
DaytonaDinDServiceEnvironment.exec(["pytest"])
  → docker compose -p <project> -f <compose_path> exec -T <service> pytest
    → runs inside the service's container via DinD
```

- **exec**: `docker compose exec -T [--workdir] [--env K=V] <service> <cmd>`
- **write_file**: two-hop — SDK upload to sandbox temp → `docker compose cp temp <service>:<path>`
- **read_file**: two-hop — `docker compose cp <service>:<path> temp` → SDK download

### Cleanup

- `sample_cleanup()`: Detect DinD environments → `docker compose down --remove-orphans` → delete sandbox
- `task_cleanup()`: DinD sandboxes tracked by ID in `_running_sandboxes` — existing cleanup loop handles them

### Networking

Daytona restricts outgoing network access by subscription tier. Tiers 1–2 only allow traffic to an "Essential Services" allowlist (managed by Daytona), while tiers 3–4 have full internet access. The essential services list includes container registries (Docker Hub, GHCR), package managers (npm, PyPI), and git hosts — so `docker pull` from standard registries works on all tiers. Services that need arbitrary external endpoints (APIs, databases) require tier 3+.

The DinD sandbox itself must always have network access (`network_block_all=False`) since the Docker daemon needs it for image pulls. If the user's compose file requests network isolation (e.g., `network_mode: none`), this is applied at the compose service level — the user's compose config handles this naturally since `network_mode` is per-service.

### Deferred

- **Startup latency**: Docker daemon boot + image pulls + compose up adds overhead. Daytona snapshots could help later (create snapshot after daemon is running and base images are pulled) but adds complexity.
- **Modal support**: Same DinD approach should work for Modal. Start with Daytona since we have Harbor as a reference implementation.
