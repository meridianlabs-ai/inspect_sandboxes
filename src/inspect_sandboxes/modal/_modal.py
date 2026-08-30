from __future__ import annotations

import asyncio
import errno
import os
import shlex
import sys
from contextvars import ContextVar
from logging import getLogger
from pathlib import PurePosixPath
from typing import Any, Literal, cast, overload

import modal
import modal.exception
from inspect_ai.util import (
    ComposeConfig,
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    is_compose_yaml,
    is_dockerfile,
    parse_compose_yaml,
    sandboxenv,
    trace_message,
    warn_once,
)
from inspect_ai.util._sandbox.environment import (
    HostMapping,
    PortMapping,
    SandboxConnection,
)
from rich import box, print
from rich.prompt import Confirm
from rich.table import Table
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import override

from inspect_sandboxes._util.naming import make_sandbox_name

from ._compose import _MODAL_PORT_KEYS, convert_compose_to_modal_params

logger = getLogger(__name__)

MODAL_APP_NAME = "inspect_modal_sandbox"
INSPECT_SANDBOX_TAG = {"created_by": "inspect-ai"}

_running_sandboxes: ContextVar[list[str]] = ContextVar("modal_running_sandboxes")


def sandbox_cleanup_startup() -> None:
    _running_sandboxes.set([])


def running_sandboxes() -> list[str]:
    return _running_sandboxes.get()


# ---------------------------------------------------------------------------
# Underlying SDK retry (for reference):
#   The Modal gRPC layer retries DEADLINE_EXCEEDED, UNAVAILABLE, CANCELLED,
#   INTERNAL, and UNKNOWN with 3 attempts and 0.1–1 s exponential backoff.
#   It does NOT retry RESOURCE_EXHAUSTED (rate limiting). If the SDK's retry
#   is exhausted the error reaches this layer for additional retry.
# ---------------------------------------------------------------------------

# Retry decorator for file I/O and sandbox lifecycle ops.
# RemoteError indicates permanent server-side failures (e.g. image build errors).
_standard_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_not_exception_type(modal.exception.RemoteError),
    reraise=True,
)

# Retry decorator for exec operations
_exec_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_not_exception_type(
        (
            modal.exception.RemoteError,
            UnicodeDecodeError,  # permanent decode failure in command output
            asyncio.TimeoutError,  # handled by exec()'s timeout retry loop
            asyncio.CancelledError,  # handled by exec()'s timeout retry loop
        )
    ),
    reraise=True,
)

# Absolute candidates for `su`, in preference order. Resolved only through
# these paths -- never through $PATH, which exec()'s caller-supplied `env=`
# may replace before this wrapper runs. A PATH-based lookup would then miss
# a `su` that is actually present in the sandbox (observed as a spurious
# "su: not found" even though direct exec of the same image succeeds).
_SU_CANDIDATES = ("/bin/su", "/usr/bin/su")


def _locate_su_snippet(candidates: tuple[str, ...] = _SU_CANDIDATES) -> str:
    """POSIX-sh snippet that sets `$su` to the first executable candidate.

    Exits with a clear error -- rather than falling back to a PATH-based
    `su` lookup -- if none of `candidates` is executable.
    """
    branches = [
        f"if [ -x {path} ]; then su={path};"
        if index == 0
        else f" elif [ -x {path} ]; then su={path};"
        for index, path in enumerate(candidates)
    ]
    checked = ", ".join(candidates)
    return (
        "".join(branches)
        + f" else echo 'su: not found (checked {checked})' >&2; exit 1; fi"
    )


def _resolve_uid_snippet(uid: str) -> str:
    """POSIX-sh snippet resolving `uid` to `$u` via a plain /etc/passwd scan.

    Avoids `getent`, which minimal images (e.g. bare `busybox:1.36`) lack.
    Fails loudly, before `$u` is ever used, when the uid is unmapped.
    """
    quoted_uid = shlex.quote(uid)
    return (
        'u=""; while IFS=: read -r name _ passwd_uid _; do '
        f'if [ "$passwd_uid" = {quoted_uid} ]; then u="$name"; break; fi; '
        "done < /etc/passwd; "
        f"if [ -z \"$u\" ]; then echo 'su: user {uid} does not exist' >&2; exit 1; fi"
    )


def _capture_path_snippet() -> str:
    r"""POSIX-sh snippet that sets `$path_q` to `$PATH`, single-quote-escaped.

    Escapes every `'` in `$PATH` to `'\''` -- the standard substitution for
    safely re-embedding a value between single quotes -- using only `case`
    and parameter expansion, never an external tool such as `sed`: at the
    point this runs, PATH is still whatever the caller's `env=` supplied,
    and a minimal image is not guaranteed to have anything beyond shell
    builtins on it.
    """
    return (
        'path_q=""; rest="$PATH"; '
        "while true; do "
        'case "$rest" in '
        "*\\'*) path_q=\"$path_q${rest%%\\'*}'\\''\"; "
        'rest="${rest#*\\\'}";; '
        '*) path_q="$path_q$rest"; break;; '
        "esac; "
        "done"
    )


def _build_exec_cmd(cmd: list[str], user: str | None) -> list[str]:
    """Build the argv Modal's Sandbox.exec() should run to honour `user`.

    Modal's Sandbox.exec() has no user parameter, so a user switch is
    emulated with `su`. Without `-l`, `su` preserves the caller's cwd and
    (with `-p`) its environment, so this only changes the effective uid --
    matching the Docker and k8s providers, which scope `user=` to uid
    rather than a fresh login environment.

    `su` accepts only usernames, while this method's contract is "username
    or UID", so a numeric user is resolved to its username inside the
    sandbox (the uid->name mapping lives there, not on the host) via
    `_resolve_uid_snippet`. An unmapped uid fails loudly rather than
    silently running as the container default.

    The `--` immediately before `"$u"` is load-bearing: without it, an
    option-like username (a literal "-p", or a malformed passwd entry whose
    name starts with "-") is parsed by `su` as another flag instead of the
    target user, and the command runs as uid 0 on both util-linux and
    BusyBox `su`.

    `su` is located via `_locate_su_snippet` (absolute paths only, never
    $PATH).

    util-linux `su -p` does *not* reliably preserve `PATH`: on Debian and
    Ubuntu, PAM's `pam_env` (driven by `/etc/login.defs`' `ENV_PATH`) resets
    it for the target user regardless of `-p`, so a caller-supplied `env=`
    with a custom PATH silently loses it and a relative-name command that
    only exists on that PATH fails "not found" -- while BusyBox `su`
    preserves PATH unconditionally, so the same wrapper must not regress
    it there. The fix: capture the outer wrapper's PATH (still whatever
    the caller's `env=` supplied, since this runs before `su`) into
    `$path_q` via `_capture_path_snippet` -- single-quote-escaped so it
    round-trips safely -- and re-assert it as the first statement *inside*
    `su`'s `-c` payload, before `exec`ing the wrapped command. That payload
    is built by quote-switching (a double-quoted `PATH='$path_q'; export
    PATH; exec ` fragment, glued with no intervening space to the already
    single-quote-escaped `quoted_cmd`) rather than interpolating
    `quoted_cmd` inside the double-quoted fragment, because `quoted_cmd`
    may itself contain unescaped `"`, `$`, or `` ` `` that would otherwise
    be re-expanded by the outer shell before `su` ever sees them.

    This wrapper resolves its own utilities (`su`, plus the shell builtins
    `read`/`echo`/`case`/`[`/`exit`) without needing PATH at all, so none
    of the above depends on PATH being sane before it is restored.
    """
    if user is None:
        return cmd

    quoted_cmd = shlex.quote(shlex.join(cmd))
    set_user = (
        _resolve_uid_snippet(user) + "; "
        if user.isdigit()
        else f"u={shlex.quote(user)}; "
    )
    script = (
        f"{set_user}{_locate_su_snippet()}; {_capture_path_snippet()}; "
        f'exec "$su" -p -s /bin/sh -- "$u" -c '
        f"\"PATH='$path_q'; export PATH; exec \"{quoted_cmd}"
    )
    return ["/bin/sh", "-c", script]


@sandboxenv(name="modal")
class ModalSandboxEnvironment(SandboxEnvironment):
    def __init__(self, sandbox: modal.Sandbox, has_tunnels: bool = False) -> None:
        super().__init__()
        self.sandbox = sandbox
        # Whether any tunnels were declared at creation (from service.ports or
        # x-modal.*_ports). When none were, connection() must NOT call
        # tunnels(): with no declared tunnels that RPC blocks for ~50s before
        # raising, on every call.
        self._has_tunnels = has_tunnels

    @classmethod
    def config_files(cls) -> list[str]:
        return [
            "compose.yaml",
            "compose.yml",
            "docker-compose.yaml",
            "docker-compose.yml",
            "Dockerfile",
        ]

    @classmethod
    def is_docker_compatible(cls) -> bool:
        return True

    @override
    @classmethod
    async def task_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None
    ) -> None:
        modal.enable_output()
        sandbox_cleanup_startup()

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        app = await cls._lookup_app(MODAL_APP_NAME)

        sandbox_kwargs: dict[str, Any] = {
            "app": app,
            "name": make_sandbox_name(task_name, metadata),
            "timeout": 60 * 60 * 24,
        }
        command: list[str] = []
        modal_params = None

        if config is None:
            trace_message(
                logger, "modal", f"Using default Modal image for task '{task_name}'"
            )
        elif is_dockerfile(config):
            sandbox_kwargs["image"] = modal.Image.from_dockerfile(config)
        elif is_compose_yaml(config):
            compose_config = parse_compose_yaml(config, multiple_services=False)
            modal_params = convert_compose_to_modal_params(compose_config, config)
            command = modal_params.command
            sandbox_kwargs.update(modal_params.kwargs)
        elif isinstance(config, ComposeConfig):
            modal_params = convert_compose_to_modal_params(config, None)
            command = modal_params.command
            sandbox_kwargs.update(modal_params.kwargs)
        else:
            raise ValueError(
                f"Unrecognized config: {config}. "
                "Expected a compose file (*.yaml/*.yml), Dockerfile, "
                "ComposeConfig object, or None."
            )

        if modal_params is not None and modal_params.volumes:
            mount_paths = [spec.mount_path for spec in modal_params.volumes]
            if len(mount_paths) != len(set(mount_paths)):
                duplicates = sorted(
                    {path for path in mount_paths if mount_paths.count(path) > 1}
                )
                raise ValueError(
                    "x-modal.volumes has multiple entries with the same "
                    f"mount_path: {duplicates}. Each mounted Volume needs a "
                    "distinct mount_path."
                )
            sandbox_kwargs["volumes"] = {
                spec.mount_path: modal.Volume.from_name(spec.name).with_mount_options(
                    read_only=spec.read_only
                )
                for spec in modal_params.volumes
            }

        sandbox = await cls._create_sandbox(command, sandbox_kwargs)
        await sandbox.set_tags.aio(INSPECT_SANDBOX_TAG)
        running_sandboxes().append(sandbox.object_id)

        has_tunnels = any(sandbox_kwargs.get(key) for key in _MODAL_PORT_KEYS)
        return {"default": cls(sandbox, has_tunnels=has_tunnels)}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        if not environments or interrupted:
            return

        for env in environments.values():
            sandbox = None
            try:
                sandbox = env.as_type(ModalSandboxEnvironment).sandbox
                await cls._terminate_sandbox(sandbox)

            except Exception as e:
                sandbox_id = cls._get_sandbox_id(sandbox)
                trace_message(
                    logger,
                    "modal",
                    f"Error terminating Modal sandbox {sandbox_id} for task '{task_name}': {e}. "
                    "Will retry in task_cleanup.",
                )

    @override
    @classmethod
    async def task_cleanup(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool
    ) -> None:
        """Cleanup sandboxes at task completion.

        Note: ``terminate()`` is idempotent (no-op if already terminated).
        """
        if not cleanup:
            return

        failed_ids: list[str] = []

        for sandbox_id in running_sandboxes().copy():
            try:
                sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
                await cls._terminate_sandbox(sandbox)
                trace_message(logger, "modal", f"Terminated sandbox {sandbox_id}")

            except Exception as e:
                failed_ids.append(sandbox_id)
                logger.error(f"Failed to terminate sandbox {sandbox_id}: {e}")

        if failed_ids:
            logger.warning(
                f"Failed to cleanup {len(failed_ids)} sandbox(es). "
                f"Failed IDs: {', '.join(failed_ids)}"
            )

        running_sandboxes().clear()

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        if id is not None:
            # Single sandbox cleanup
            try:
                sandbox = await modal.Sandbox.from_id.aio(id)
                await cls._terminate_sandbox(sandbox)
                print(f"Successfully terminated sandbox {id}")
            except Exception as e:
                print(f"[red]Error terminating sandbox {id}: {e}[/red]")
                sys.exit(1)
        else:
            # Bulk cleanup
            sandboxes = [
                sb async for sb in modal.Sandbox.list.aio(tags=INSPECT_SANDBOX_TAG)
            ]

            if not sandboxes:
                print("No Modal sandboxes found to clean up.")
                return

            table = Table(
                box=box.SQUARE,
                show_lines=False,
                title_style="bold",
                title_justify="left",
            )
            table.add_column("Sandbox ID")
            for sb in sandboxes:
                table.add_row(sb.object_id)
            print(table)

            # Only prompt if in an interactive shell
            is_interactive = sys.stdin.isatty()
            is_ci = "CI" in os.environ
            is_pytest = "PYTEST_CURRENT_TEST" in os.environ

            if is_interactive and not is_ci and not is_pytest:
                if not Confirm.ask(
                    f"Are you sure you want to terminate ALL {len(sandboxes)} "
                    "sandbox(es) above?"
                ):
                    print("Cancelled.")
                    return

            success_count = 0
            failure_count = 0

            for sb in sandboxes:
                try:
                    await cls._terminate_sandbox(sb)
                    success_count += 1
                except Exception as e:
                    print(
                        f"[yellow]Error terminating sandbox {sb.object_id}: {e}[/yellow]"
                    )
                    failure_count += 1

            print(f"\n[green]Successfully terminated: {success_count}[/green]")
            if failure_count > 0:
                print(f"[red]Failed to terminate: {failure_count}[/red]")
                sys.exit(1)
            else:
                print("Complete.")

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        # See _build_exec_cmd's docstring for the full rationale: user= is
        # emulated via `su` because Modal's Sandbox.exec() has no user
        # parameter of its own.
        exec_cmd = _build_exec_cmd(cmd, user)

        # Modal requires absolute paths for workdir
        workdir = cwd
        if workdir is not None and not PurePosixPath(workdir).is_absolute():
            warn_once(
                logger,
                f"Relative path '{workdir}' for cwd parameter was converted to absolute path '/{workdir}' "
                "(relative to filesystem root). For clarity, consider using absolute paths.",
            )
            workdir = f"/{workdir}"

        @_exec_retry
        async def _run() -> ExecResult[str]:
            modal_env = cast(dict[str, str | None] | None, env)

            process = await self.sandbox.exec.aio(
                *exec_cmd,
                workdir=workdir,
                env=modal_env,
            )

            if input is not None:
                try:
                    data = input.encode("utf-8") if isinstance(input, str) else input
                    process.stdin.write(data)
                except modal.exception.InternalError as e:
                    logger.warning(f"Modal InternalError while writing stdin: {e}.")
                    raise
                finally:
                    # No kill() on Modal's ContainerProcess
                    # Close stdin to unblock the process
                    try:
                        process.stdin.write_eof()
                        await process.stdin.drain.aio()
                    except Exception:
                        pass

            try:
                stdout = await process.stdout.read.aio()
                stderr = await process.stderr.read.aio()
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Failed to decode command output: {e.reason}",
                ) from e

            await process.wait.aio()

            return ExecResult(
                success=process.returncode == 0,
                returncode=process.returncode if process.returncode is not None else 0,
                stdout=stdout,
                stderr=stderr,
            )

        # Timeout: Modal kills exec'd processes server-side when the gRPC
        # stream is cancelled. No in-container ``timeout`` wrapping needed.
        # On timeout, retry with a capped timeout: first retry ≤60s, second ≤30s.
        if timeout_retry:
            t1 = min(timeout, 60) if timeout is not None else 60
            t2 = min(timeout, 30) if timeout is not None else 30
            attempt_timeouts: list[int | None] = [timeout, t1, t2]
        else:
            attempt_timeouts = [timeout]

        last_timeout_exc: asyncio.TimeoutError | None = None
        for t in attempt_timeouts:
            try:
                if t is not None:
                    return await asyncio.wait_for(_run(), timeout=t)
                else:
                    return await _run()
            except asyncio.TimeoutError as e:
                last_timeout_exc = e

        assert last_timeout_exc is not None
        raise TimeoutError(
            f"Command timed out after {timeout} seconds"
        ) from last_timeout_exc

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Creates parent directories automatically if they don't exist.

        Raises:
            IsADirectoryError: File path already exists as a directory.
        """
        parent = str(PurePosixPath(file).parent)
        if parent and parent not in ("/", "."):
            await self._create_parent_folder(parent)

        try:
            await self._write_file_content(file, contents)
        except IsADirectoryError as e:
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read file from sandbox.

        Raises:
            FileNotFoundError: File does not exist.
            IsADirectoryError: Path is a directory.
            UnicodeDecodeError: Encoding error (text mode only).
            OutputLimitExceededError: File exceeds 100 MiB limit.
        """
        await self._verify_read_file_size(file)

        try:
            contents_bytes = await self._read_file_content(file)
        except modal.exception.FilesystemExecutionError as e:
            if await self._is_directory(file):
                raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

        if text:
            try:
                return contents_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Failed to decode {file}: {e.reason}",
                ) from e

        return contents_bytes

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        """Surface the sandbox's declared tunnels as port mappings.

        Modal declares tunnels at creation (from ``service.ports`` or
        ``x-modal.*_ports``) and exposes them at runtime via
        ``sandbox.tunnels()``, keyed by container port. We map each tunnel to a
        ``PortMapping`` so the declared ports reach the eval through inspect's
        standard ``SandboxConnection.ports`` field.

        When no tunnels were declared we skip ``tunnels()`` entirely: with none
        declared that RPC blocks for ~50s before raising, so calling it would
        slow every connection() on a Dockerfile/portless sandbox.
        """
        ports: list[PortMapping] | None = None
        tunnels: dict[int, Any] = {}
        if self._has_tunnels:
            try:
                tunnels = await self.sandbox.tunnels.aio()
            except Exception as e:
                # tunnels() can still raise if they aren't ready yet; a
                # connection without ports is useful, so don't fail here.
                trace_message(logger, "modal", f"Could not retrieve Modal tunnels: {e}")
                tunnels = {}

        if tunnels:
            ports = []
            for container_port, tunnel in tunnels.items():
                # Prefer the raw TCP socket (from unencrypted_ports). Encrypted
                # tunnels (encrypted_ports / h2_ports) have no TCP socket, so
                # fall back to the public TLS host:port.
                if tunnel.unencrypted_host:
                    host, host_port = tunnel.tcp_socket
                else:
                    host, host_port = tunnel.tls_socket
                ports.append(
                    PortMapping(
                        container_port=container_port,
                        protocol="tcp",
                        mappings=[HostMapping(host_ip=host, host_port=host_port)],
                    )
                )

        return SandboxConnection(
            type="modal",
            command=f"modal shell {self.sandbox.object_id}",
            ports=ports,
            container=self.sandbox.object_id,
        )

    @staticmethod
    @_standard_retry
    async def _lookup_app(app_name: str) -> modal.App:
        return await modal.App.lookup.aio(app_name, create_if_missing=True)

    @staticmethod
    @_standard_retry
    async def _create_sandbox(
        command: list[str], kwargs: dict[str, Any]
    ) -> modal.Sandbox:
        return await modal.Sandbox.create.aio(*command, **kwargs)

    @staticmethod
    @_standard_retry
    async def _terminate_sandbox(sandbox: modal.Sandbox) -> None:
        await sandbox.terminate.aio()
        # Verify the sandbox stopped — poll() returns None if still running
        if await sandbox.poll.aio() is None:
            raise RuntimeError(
                f"Sandbox {sandbox.object_id} still running after terminate()"
            )

    @staticmethod
    def _get_sandbox_id(sandbox: modal.Sandbox | None) -> str:
        if sandbox is None:
            return "unknown"
        return getattr(sandbox, "object_id", "unknown")

    @_standard_retry
    async def _write_file_content(self, file: str, contents: str | bytes) -> None:
        if isinstance(contents, str):
            async with await self.sandbox.open.aio(file, "w") as f:
                await f.write.aio(contents)
        else:
            async with await self.sandbox.open.aio(file, "wb") as f:
                await f.write.aio(contents)

    @_standard_retry
    async def _read_file_content(self, file: str) -> bytes:
        async with await self.sandbox.open.aio(file, "rb") as f:
            return await f.read.aio()

    @_standard_retry
    async def _create_parent_folder(self, path: str) -> None:
        try:
            await self.sandbox.mkdir.aio(path, parents=True)
        except FileExistsError:
            pass

    @_standard_retry
    async def _is_directory(self, file: str) -> bool:
        process = await self.sandbox.exec.aio("test", "-d", file)
        await process.wait.aio()
        return process.returncode == 0

    @_standard_retry
    async def _get_file_size(self, file: str) -> int:
        process = await self.sandbox.exec.aio("stat", "-c", "%s", file)
        stdout = await process.stdout.read.aio()
        await process.wait.aio()

        if process.returncode != 0:
            if process.returncode == 1:
                raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
            stderr = await process.stderr.read.aio()
            raise RuntimeError(
                f"stat command failed with code {process.returncode}: {stderr}"
            )

        try:
            return int(stdout.strip())
        except ValueError as e:
            raise RuntimeError(f"Failed to parse file size for {file}") from e

    async def _verify_read_file_size(self, file: str) -> None:
        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

        file_size = await self._get_file_size(file)
        if file_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )
