"""Runloop sandbox provider."""

from __future__ import annotations

import os
import sys
import uuid
from contextvars import ContextVar
from logging import getLogger
from typing import Any

from inspect_ai.util import (
    ComposeConfig,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    is_compose_yaml,
    is_dockerfile,
    parse_compose_yaml,
    sandboxenv,
    trace_message,
)
from rich import box, print
from rich.prompt import Confirm
from rich.table import Table
from runloop_api_client import AsyncRunloop, NotFoundError
from typing_extensions import override

from inspect_sandboxes._util.naming import make_sandbox_name

from ._blueprint import build_blueprint_for_dockerfile, build_blueprint_for_image
from ._compose import (
    RunloopSingleServiceParams,
    resolve_single_service_params,
)
from ._single_env import RunloopSingleServiceEnvironment

logger = getLogger(__name__)

INSPECT_SANDBOX_METADATA = {"created_by": "inspect-ai"}

# Free tier caps concurrent Devboxes at 5; can be raised on Pro.
_DEFAULT_CONCURRENCY = 5

# Page size for paginated list responses.
_LIST_PAGE_LIMIT = 100

_runloop_client: ContextVar[AsyncRunloop | None] = ContextVar(
    "runloop_client", default=None
)
_running_sandboxes: ContextVar[list[str]] = ContextVar("runloop_running_sandboxes")
_run_id: ContextVar[str] = ContextVar("runloop_run_id")


def _init_context() -> None:
    _running_sandboxes.set([])
    _run_id.set(uuid.uuid4().hex)


def _run_metadata(task_name: str | None = None) -> dict[str, str]:
    metadata = {**INSPECT_SANDBOX_METADATA, "inspect_run_id": _run_id.get()}
    if task_name:
        metadata["task"] = task_name
    return metadata


async def list_devboxes(client: AsyncRunloop, metadata: dict[str, str]) -> list[Any]:
    """List running devboxes whose metadata matches *all* given key/value pairs.

    Runloop's ``devboxes.list`` API doesn't accept a metadata filter, so we
    paginate and filter client-side.
    """
    matches: list[Any] = []
    async for devbox in client.devboxes.list(status="running", limit=_LIST_PAGE_LIMIT):
        meta = getattr(devbox, "metadata", None) or {}
        if all(meta.get(k) == v for k, v in metadata.items()):
            matches.append(devbox)
    return matches


@sandboxenv(name="runloop")
class RunloopSandboxEnvironment(SandboxEnvironment):
    """Runloop sandbox provider.

    Owns all lifecycle class methods. ``sample_init`` returns instances of
    ``RunloopSingleServiceEnvironment`` or ``RunloopDinDServiceEnvironment``.
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

    @classmethod
    def default_concurrency(cls) -> int | None:
        return _DEFAULT_CONCURRENCY

    @override
    @classmethod
    async def task_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
    ) -> None:
        _init_context()
        client = AsyncRunloop()
        _runloop_client.set(client)

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        client = _runloop_client.get()
        if client is None:
            raise RuntimeError(
                "Runloop client not initialized. task_init must be called first."
            )

        sandbox_name = make_sandbox_name(task_name, metadata)
        run_metadata = {**_run_metadata(task_name), "name": sandbox_name}

        if config is None:
            devbox = await client.devboxes.create_and_await_running(
                metadata=run_metadata
            )
        elif is_dockerfile(config):
            blueprint_name = await build_blueprint_for_dockerfile(client, str(config))
            devbox = await client.devboxes.create_and_await_running(
                blueprint_name=blueprint_name,
                metadata=run_metadata,
            )
        elif is_compose_yaml(config) or isinstance(config, ComposeConfig):
            if isinstance(config, ComposeConfig):
                compose_config, compose_path = config, None
            else:
                compose_config = parse_compose_yaml(config, multiple_services=True)
                compose_path = config
            if len(compose_config.services) > 1:
                # Deferred import to avoid pulling in DinD machinery for single-service.
                from ._dind_env import RunloopDinDServiceEnvironment

                envs = await RunloopDinDServiceEnvironment.sample_init_dind(
                    client,
                    compose_config,
                    compose_path,
                    metadata=run_metadata,
                )
                any_env = next(iter(envs.values())).as_type(
                    RunloopDinDServiceEnvironment
                )
                _running_sandboxes.get().append(any_env.project.devbox_id)
                return envs

            params = resolve_single_service_params(compose_config, compose_path)
            blueprint_name = await cls._build_blueprint(client, params)
            extra_metadata = dict(params.metadata)
            run_metadata = {**extra_metadata, **run_metadata}

            create_kwargs: dict[str, object] = {"metadata": run_metadata}
            if blueprint_name is not None:
                create_kwargs["blueprint_name"] = blueprint_name
            if params.blueprint_id is not None:
                create_kwargs["blueprint_id"] = params.blueprint_id
            if params.snapshot_id is not None:
                create_kwargs["snapshot_id"] = params.snapshot_id
            if params.environment_variables:
                create_kwargs["environment_variables"] = params.environment_variables
            if params.launch_parameters is not None:
                create_kwargs["launch_parameters"] = params.launch_parameters
            if params.timeout is not None:
                create_kwargs["timeout"] = params.timeout

            devbox = await client.devboxes.create_and_await_running(**create_kwargs)  # type: ignore[arg-type]
        else:
            raise ValueError(
                f"Unrecognized config: {config}. "
                "Expected a compose file (*.yaml/*.yml), Dockerfile, "
                "ComposeConfig object, or None."
            )

        _running_sandboxes.get().append(devbox.id)
        trace_message(
            logger,
            "runloop",
            f"Created devbox {devbox.id} for task '{task_name}'",
        )
        return {"default": RunloopSingleServiceEnvironment(client, devbox.id)}

    @classmethod
    async def _build_blueprint(
        cls,
        client: AsyncRunloop,
        params: RunloopSingleServiceParams,
    ) -> str | None:
        """Build (or skip) a blueprint per the resolved params.

        Returns the blueprint name if one was built; otherwise None
        (e.g. when ``blueprint_id`` or ``snapshot_id`` was specified).
        """
        if params.blueprint_id is not None or params.snapshot_id is not None:
            return None
        if params.blueprint_name is not None:
            return params.blueprint_name
        if params.dockerfile_path is not None:
            return await build_blueprint_for_dockerfile(
                client,
                params.dockerfile_path,
                launch_parameters=params.launch_parameters,
            )
        if params.image is not None:
            return await build_blueprint_for_image(
                client,
                params.image,
                launch_parameters=params.launch_parameters,
            )
        return None

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
        # Deferred import to avoid pulling in DinD machinery for single-service.
        from ._dind_env import RunloopDinDServiceEnvironment

        any_env = next(iter(environments.values()))
        if isinstance(any_env, RunloopDinDServiceEnvironment):
            devbox_ids = [
                any_env.as_type(RunloopDinDServiceEnvironment).project.devbox_id
            ]
            await RunloopDinDServiceEnvironment.sample_cleanup(
                task_name, config, environments, interrupted
            )
        else:
            devbox_ids = [
                env.as_type(RunloopSingleServiceEnvironment).devbox_id
                for env in environments.values()
            ]
            await RunloopSingleServiceEnvironment.sample_cleanup(
                task_name, config, environments, interrupted
            )

        # Skip the redundant shutdown in task_cleanup's first pass. Anything
        # we failed to shut down here is still caught by the orphan-recovery
        # pass via inspect_run_id metadata.
        if interrupted:
            return
        running = _running_sandboxes.get()
        for devbox_id in devbox_ids:
            try:
                running.remove(devbox_id)
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

        client = _runloop_client.get()
        if client is None:
            return

        failed_ids: list[str] = []
        deleted_ids: set[str] = set()

        # First pass: tracked devboxes.
        for devbox_id in _running_sandboxes.get().copy():
            try:
                await client.devboxes.shutdown(devbox_id)
                deleted_ids.add(devbox_id)
                trace_message(logger, "runloop", f"Shut down devbox {devbox_id}")
            except NotFoundError:
                # Already shut down (e.g. by sample_cleanup) — nothing to do.
                deleted_ids.add(devbox_id)
                trace_message(
                    logger,
                    "runloop",
                    f"Devbox {devbox_id} already shut down, skipping.",
                )
            except Exception as e:
                failed_ids.append(devbox_id)
                logger.error(f"Failed to shut down devbox {devbox_id}: {e}")

        # Second pass: orphans by run metadata.
        try:
            run_id = _run_id.get()
        except LookupError:
            run_id = ""
        if run_id:
            try:
                orphans = await list_devboxes(client, {"inspect_run_id": run_id})
                for devbox in orphans:
                    if devbox.id in deleted_ids:
                        continue
                    try:
                        await client.devboxes.shutdown(devbox.id)
                        deleted_ids.add(devbox.id)
                        trace_message(
                            logger,
                            "runloop",
                            f"Shut down orphaned devbox {devbox.id}",
                        )
                    except NotFoundError:
                        deleted_ids.add(devbox.id)
                    except Exception as e:
                        failed_ids.append(devbox.id)
                        logger.error(
                            f"Failed to shut down orphaned devbox {devbox.id}: {e}"
                        )
            except Exception as e:
                logger.warning(f"Failed to list devboxes for cleanup: {e}")

        if failed_ids:
            logger.warning(
                f"Failed to cleanup {len(failed_ids)} devbox(es). "
                f"Failed IDs: {', '.join(failed_ids)}"
            )

        _running_sandboxes.get().clear()
        _runloop_client.set(None)

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        client = AsyncRunloop()

        if id is not None:
            try:
                await client.devboxes.shutdown(id)
                print(f"Successfully shut down devbox {id}")
            except Exception as e:
                print(f"[red]Error shutting down devbox {id}: {e}[/red]")
                sys.exit(1)
            return

        # Bulk cleanup.
        devboxes = await list_devboxes(client, INSPECT_SANDBOX_METADATA)

        if not devboxes:
            print("No Runloop devboxes found to clean up.")
            return

        table = Table(
            box=box.SQUARE,
            show_lines=False,
            title_style="bold",
            title_justify="left",
        )
        table.add_column("Devbox ID")
        for devbox in devboxes:
            table.add_row(devbox.id)
        print(table)

        is_interactive = sys.stdin.isatty()
        is_ci = "CI" in os.environ
        is_pytest = "PYTEST_CURRENT_TEST" in os.environ

        if is_interactive and not is_ci and not is_pytest:
            if not Confirm.ask(
                f"Are you sure you want to shut down ALL {len(devboxes)} devbox(es) above?"
            ):
                print("Cancelled.")
                return

        success_count = 0
        failure_count = 0
        for devbox in devboxes:
            try:
                await client.devboxes.shutdown(devbox.id)
                success_count += 1
            except Exception as e:
                print(f"[yellow]Error shutting down devbox {devbox.id}: {e}[/yellow]")
                failure_count += 1

        print(f"\n[green]Successfully shut down: {success_count}[/green]")
        if failure_count > 0:
            print(f"[red]Failed to shut down: {failure_count}[/red]")
            sys.exit(1)
        else:
            print("Complete.")
