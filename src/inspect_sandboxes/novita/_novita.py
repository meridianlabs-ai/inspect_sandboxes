"""Novita sandbox provider."""

from __future__ import annotations

import os
import sys
import uuid
from contextvars import ContextVar
from logging import getLogger

from inspect_ai.util import (
    ComposeConfig,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    is_compose_yaml,
    is_dockerfile,
    parse_compose_yaml,
    sandboxenv,
    trace_message,
    warn_once,
)
from novita_sandbox.core import AsyncSandbox, SandboxQuery
from rich import box, print
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import override

from inspect_sandboxes._util.compose import find_default_service
from inspect_sandboxes._util.naming import make_sandbox_name

from ._compose import (
    NovitaSingleServiceParams,
    extract_novita_timeout,
    extract_x_novita,
    resolve_single_service_params,
    service_connection_ports,
)
from ._dind_env import NovitaDinDServiceEnvironment
from ._dind_project import (
    DEFAULT_DIND_CPU,
    DEFAULT_DIND_MEMORY_MB,
)
from ._single_env import NovitaSingleServiceEnvironment
from ._template import build_template_for_dockerfile, build_template_for_image

logger = getLogger(__name__)

INSPECT_SANDBOX_METADATA = {"created_by": "inspect-ai"}

# Default sandbox lifetime (seconds). Novita caps this at 3600s for Free tier
# and 86400s for Paid tier.
_DEFAULT_SANDBOX_TIMEOUT = 3600

_running_sandboxes: ContextVar[list[str]] = ContextVar("novita_running_sandboxes")
_run_id: ContextVar[str] = ContextVar("novita_run_id")


def _init_context() -> None:
    _running_sandboxes.set([])
    _run_id.set(uuid.uuid4().hex)


def _run_metadata(task_name: str | None = None) -> dict[str, str]:
    metadata = {**INSPECT_SANDBOX_METADATA, "inspect_run_id": _run_id.get()}
    if task_name:
        metadata["task"] = task_name
    return metadata


@sandboxenv(name="novita")
class NovitaSandboxEnvironment(SandboxEnvironment):
    """Novita sandbox provider.

    Owns all lifecycle class methods. ``sample_init`` returns instances of
    ``NovitaSingleServiceEnvironment`` or ``NovitaDinDServiceEnvironment``.
    """

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
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
    ) -> None:
        _init_context()
        # Warm the template cache; sample_init re-resolves from each sample's
        # own config (which may differ from the task's).
        if config is None:
            return
        if is_dockerfile(config):
            await build_template_for_dockerfile(str(config))
            return
        if is_compose_yaml(config) or isinstance(config, ComposeConfig):
            if isinstance(config, ComposeConfig):
                compose_config, compose_path = config, None
            else:
                compose_config = parse_compose_yaml(config, multiple_services=True)
                compose_path = config
            if len(compose_config.services) > 1:
                # DinD template is built lazily inside sample_init_dind.
                return
            params = resolve_single_service_params(compose_config, compose_path)
            if params.template is not None:
                return
            if params.dockerfile_path is not None:
                await build_template_for_dockerfile(
                    params.dockerfile_path,
                    cpu_count=params.cpu_count,
                    memory_mb=params.memory_mb,
                )
            elif params.image is not None:
                await build_template_for_image(
                    params.image,
                    cpu_count=params.cpu_count,
                    memory_mb=params.memory_mb,
                )

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        template: str | None = None
        params: NovitaSingleServiceParams | None = None
        compose_config: ComposeConfig | None = None

        if config is None:
            template = None
        elif is_dockerfile(config):
            template = await build_template_for_dockerfile(str(config))
        elif is_compose_yaml(config) or isinstance(config, ComposeConfig):
            if isinstance(config, ComposeConfig):
                compose_config, compose_path = config, None
            else:
                compose_config = parse_compose_yaml(config, multiple_services=True)
                compose_path = config
            if len(compose_config.services) > 1:
                return await cls._dind_sample_init(
                    task_name, compose_config, compose_path, metadata
                )
            params = resolve_single_service_params(compose_config, compose_path)
            if params.template is not None:
                template = params.template
            elif params.dockerfile_path is not None:
                template = await build_template_for_dockerfile(
                    params.dockerfile_path,
                    cpu_count=params.cpu_count,
                    memory_mb=params.memory_mb,
                )
            elif params.image is not None:
                template = await build_template_for_image(
                    params.image,
                    cpu_count=params.cpu_count,
                    memory_mb=params.memory_mb,
                )
        else:
            raise ValueError(
                f"Unrecognized config: {config}. "
                "Expected a compose file (*.yaml/*.yml), Dockerfile, "
                "ComposeConfig object, or None."
            )

        envs: dict[str, str] | None = None
        sandbox_timeout: float | int = _DEFAULT_SANDBOX_TIMEOUT
        allow_internet_access: bool = True
        extra_metadata: dict[str, str] = {}
        if params is not None:
            envs = params.envs or None
            if params.timeout is not None:
                sandbox_timeout = params.timeout
            elif compose_config is not None:
                timeout_override = extract_novita_timeout(compose_config.extensions)
                if timeout_override is not None:
                    sandbox_timeout = timeout_override
            extra_metadata = dict(params.metadata)
            allow_internet_access = params.allow_internet_access

        run_metadata = {
            **extra_metadata,
            **_run_metadata(task_name),
            "name": make_sandbox_name(task_name, metadata),
        }

        sandbox = await AsyncSandbox.create(
            template=template,
            timeout=int(sandbox_timeout),
            metadata=run_metadata,
            envs=envs,
            allow_internet_access=allow_internet_access,
        )
        _running_sandboxes.get().append(sandbox.sandbox_id)
        trace_message(
            logger,
            "novita",
            f"Created sandbox {sandbox.sandbox_id} for task '{task_name}'",
        )

        connection_ports: list[int] = []
        if compose_config is not None:
            _, default_service = find_default_service(compose_config)
            connection_ports = service_connection_ports(default_service)

        return {"default": NovitaSingleServiceEnvironment(sandbox, connection_ports)}

    @classmethod
    async def _dind_sample_init(
        cls,
        task_name: str,
        compose_config: ComposeConfig,
        compose_path: str | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        ext = extract_x_novita(compose_config.extensions)
        cpu = int(ext.get("cpu_count", DEFAULT_DIND_CPU))
        mem = int(ext.get("memory_mb", DEFAULT_DIND_MEMORY_MB))

        if ext.get("allow_internet_access") is not None:
            warn_once(
                logger,
                "x-novita.allow_internet_access is ignored for DinD multi-service "
                "sandboxes (the Docker daemon requires network access for image "
                "pulls).",
            )

        sandbox_timeout: float | int = _DEFAULT_SANDBOX_TIMEOUT
        timeout_override = extract_novita_timeout(compose_config.extensions)
        if timeout_override is not None:
            sandbox_timeout = timeout_override

        extra_metadata: dict[str, str] = {}
        meta_raw = ext.get("metadata")
        if isinstance(meta_raw, dict):
            extra_metadata = {str(k): str(v) for k, v in meta_raw.items()}

        run_metadata = {
            **extra_metadata,
            **_run_metadata(task_name),
            "name": make_sandbox_name(task_name, metadata),
        }

        envs_raw = ext.get("envs")
        sandbox_envs = (
            {str(k): str(v) for k, v in envs_raw.items()}
            if isinstance(envs_raw, dict)
            else None
        )

        envs = await NovitaDinDServiceEnvironment.sample_init_dind(
            compose_config,
            compose_path,
            metadata=run_metadata,
            cpu_count=cpu,
            memory_mb=mem,
            sandbox_timeout=sandbox_timeout,
            sandbox_envs=sandbox_envs,
        )
        any_env = next(iter(envs.values())).as_type(NovitaDinDServiceEnvironment)
        _running_sandboxes.get().append(any_env.project.sandbox.sandbox_id)
        return envs

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        if not environments:
            return
        any_env = next(iter(environments.values()))
        if isinstance(any_env, NovitaDinDServiceEnvironment):
            sandbox_ids = [
                any_env.as_type(NovitaDinDServiceEnvironment).project.sandbox.sandbox_id
            ]
            await NovitaDinDServiceEnvironment.sample_cleanup(
                task_name, config, environments, interrupted
            )
        else:
            sandbox_ids = [
                env.as_type(NovitaSingleServiceEnvironment).sandbox.sandbox_id
                for env in environments.values()
            ]
            await NovitaSingleServiceEnvironment.sample_cleanup(
                task_name, config, environments, interrupted
            )

        # Skip the redundant DELETE in task_cleanup's first pass (the Novita SDK
        # logs it as ERROR Response 404). Anything we failed to kill here is
        # still caught by the orphan-recovery pass via inspect_run_id metadata.
        if interrupted:
            return
        running = _running_sandboxes.get()
        for sandbox_id in sandbox_ids:
            try:
                running.remove(sandbox_id)
            except ValueError:
                pass

    @override
    @classmethod
    async def task_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        cleanup: bool,
    ) -> None:
        if not cleanup:
            return

        failed_ids: list[str] = []
        deleted_ids: set[str] = set()

        for sandbox_id in _running_sandboxes.get().copy():
            try:
                killed = await AsyncSandbox.kill(sandbox_id)
                deleted_ids.add(sandbox_id)
                if killed:
                    trace_message(logger, "novita", f"Killed sandbox {sandbox_id}")
                else:
                    trace_message(
                        logger,
                        "novita",
                        f"Sandbox {sandbox_id} already gone, skipping.",
                    )
            except Exception as e:
                failed_ids.append(sandbox_id)
                logger.error(f"Failed to kill sandbox {sandbox_id}: {e}")

        # Second pass: orphaned sandboxes by run metadata. Catches creation failures
        # whose IDs were never tracked in _running_sandboxes.
        try:
            run_id = _run_id.get()
        except LookupError:
            run_id = ""
        if run_id:
            try:
                paginator = AsyncSandbox.list(
                    query=SandboxQuery(metadata={"inspect_run_id": run_id})
                )
                while True:
                    items = await paginator.next_items()
                    for info in items:
                        sandbox_id = info.sandbox_id
                        if sandbox_id in deleted_ids:
                            continue
                        try:
                            await AsyncSandbox.kill(sandbox_id)
                            trace_message(
                                logger,
                                "novita",
                                f"Killed orphaned sandbox {sandbox_id}",
                            )
                        except Exception as e:
                            failed_ids.append(sandbox_id)
                            logger.error(
                                f"Failed to kill orphaned sandbox {sandbox_id}: {e}"
                            )
                    if not paginator.has_next:
                        break
            except Exception as e:
                logger.warning(f"Failed to list sandboxes for cleanup: {e}")

        if failed_ids:
            logger.warning(
                f"Failed to cleanup {len(failed_ids)} sandbox(es). "
                f"Failed IDs: {', '.join(failed_ids)}"
            )

        _running_sandboxes.get().clear()

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        if id is not None:
            try:
                killed = await AsyncSandbox.kill(id)
                if killed:
                    print(f"Successfully killed sandbox {id}")
                else:
                    print(f"Sandbox {id} not found (already deleted).")
            except Exception as e:
                print(f"[red]Error killing sandbox {id}: {e}[/red]")
                sys.exit(1)
            return

        paginator = AsyncSandbox.list(
            query=SandboxQuery(metadata=INSPECT_SANDBOX_METADATA)
        )
        sandboxes: list[str] = []
        while True:
            items = await paginator.next_items()
            sandboxes.extend(info.sandbox_id for info in items)
            if not paginator.has_next:
                break

        if not sandboxes:
            print("No Novita sandboxes found to clean up.")
            return

        table = Table(
            box=box.SQUARE,
            show_lines=False,
            title_style="bold",
            title_justify="left",
        )
        table.add_column("Sandbox ID")
        for sandbox_id in sandboxes:
            table.add_row(sandbox_id)
        print(table)

        is_interactive = sys.stdin.isatty()
        is_ci = "CI" in os.environ
        is_pytest = "PYTEST_CURRENT_TEST" in os.environ

        if is_interactive and not is_ci and not is_pytest:
            if not Confirm.ask(
                f"Are you sure you want to kill ALL {len(sandboxes)} sandbox(es) above?"
            ):
                print("Cancelled.")
                return

        success_count = 0
        failure_count = 0
        for sandbox_id in sandboxes:
            try:
                await AsyncSandbox.kill(sandbox_id)
                success_count += 1
            except Exception as e:
                print(f"[yellow]Error killing sandbox {sandbox_id}: {e}[/yellow]")
                failure_count += 1

        print(f"\n[green]Successfully killed: {success_count}[/green]")
        if failure_count > 0:
            print(f"[red]Failed to kill: {failure_count}[/red]")
            sys.exit(1)
        else:
            print("Complete.")
