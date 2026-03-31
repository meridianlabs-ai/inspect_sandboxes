"""Daytona sandbox provider"""

from __future__ import annotations

import os
import sys
import uuid
from contextvars import ContextVar
from logging import getLogger

from daytona_sdk import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaNotFoundError,
    Image,
)
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
from typing_extensions import override

from ._compose import create_single_service_params
from ._dind_env import DaytonaDinDServiceEnvironment
from ._sandbox_utils import (
    close_client,
    create_sandbox,
    delete_sandbox,
    list_sandboxes,
)
from ._single_env import DaytonaSingleServiceEnvironment

logger = getLogger(__name__)

INSPECT_SANDBOX_LABEL = {"created_by": "inspect-ai"}

_daytona_client: ContextVar[AsyncDaytona | None] = ContextVar(
    "daytona_client", default=None
)
_running_sandboxes: ContextVar[list[str]] = ContextVar("daytona_running_sandboxes")
_run_id: ContextVar[str] = ContextVar("daytona_run_id")


def _init_context() -> None:
    _running_sandboxes.set([])
    _run_id.set(uuid.uuid4().hex)


def _run_labels() -> dict[str, str]:
    return {**INSPECT_SANDBOX_LABEL, "inspect_run_id": _run_id.get()}


@sandboxenv(name="daytona")
class DaytonaSandboxEnvironment(SandboxEnvironment):
    """Daytona sandbox provider

    Owns all lifecycle class methods. sample_init returns instances of
    DaytonaSingleServiceEnvironment or DaytonaDinDServiceEnvironment.
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
        client = AsyncDaytona()
        _daytona_client.set(client)

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        client = _daytona_client.get()
        if client is None:
            raise RuntimeError(
                "Daytona client not initialized. task_init must be called first."
            )

        params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams

        if config is None:
            params = CreateSandboxFromSnapshotParams(
                snapshot=None,
                labels=_run_labels(),
                auto_stop_interval=0,
            )
        elif is_dockerfile(config):
            image = Image.from_dockerfile(config)
            params = CreateSandboxFromImageParams(
                image=image,
                labels=_run_labels(),
                auto_stop_interval=0,
            )
        elif is_compose_yaml(config) or isinstance(config, ComposeConfig):
            compose_config: ComposeConfig
            compose_file: str | None
            if is_compose_yaml(config):
                compose_config = parse_compose_yaml(config, multiple_services=True)
                compose_file = str(config)
            else:
                assert isinstance(config, ComposeConfig)
                compose_config = config
                compose_file = None

            if len(compose_config.services) > 1:
                envs = await DaytonaDinDServiceEnvironment.sample_init_dind(
                    client, compose_config, compose_file, _run_labels()
                )
                any_env = next(iter(envs.values())).as_type(
                    DaytonaDinDServiceEnvironment
                )
                _running_sandboxes.get().append(any_env.project.sandbox.id)
                return envs
            params = create_single_service_params(
                compose_config, compose_file, _run_labels()
            )
        else:
            raise ValueError(
                f"Unrecognized config: {config}. "
                "Expected a compose file (*.yaml/*.yml), Dockerfile, "
                "ComposeConfig object, or None."
            )

        sandbox = await create_sandbox(client, params)
        _running_sandboxes.get().append(sandbox.id)

        return {"default": DaytonaSingleServiceEnvironment(sandbox)}

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
        if isinstance(any_env, DaytonaDinDServiceEnvironment):
            await DaytonaDinDServiceEnvironment.sample_cleanup(
                task_name, config, environments, interrupted
            )
        else:
            await DaytonaSingleServiceEnvironment.sample_cleanup(
                task_name, config, environments, interrupted
            )

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

        client = _daytona_client.get()
        if client is None:
            return

        failed_ids: list[str] = []
        deleted_ids: set[str] = set()

        for sandbox_id in _running_sandboxes.get().copy():
            try:
                sandbox = await client.get(sandbox_id)
                await delete_sandbox(client, sandbox)
                deleted_ids.add(sandbox_id)
                trace_message(logger, "daytona", f"Deleted sandbox {sandbox_id}")
            except DaytonaNotFoundError:
                # Already deleted (e.g. by sample_cleanup) — nothing to do.
                deleted_ids.add(sandbox_id)
                trace_message(
                    logger,
                    "daytona",
                    f"Sandbox {sandbox_id} already deleted, skipping.",
                )
            except Exception as e:
                failed_ids.append(sandbox_id)
                logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")

        # Second pass: find orphaned sandboxes by run label.
        # This catches sandboxes where creation failed (e.g. build_failed)
        # and the ID was never tracked in _running_sandboxes.
        run_id = _run_id.get()
        if run_id:
            try:
                orphans = await list_sandboxes(client, {"inspect_run_id": run_id})
                for sb in orphans:
                    if sb.id in deleted_ids:
                        continue
                    try:
                        await delete_sandbox(client, sb)
                        trace_message(
                            logger,
                            "daytona",
                            f"Deleted orphaned sandbox {sb.id} (state={sb.state})",
                        )
                    except Exception as e:
                        failed_ids.append(sb.id)
                        logger.error(f"Failed to delete orphaned sandbox {sb.id}: {e}")
            except Exception as e:
                logger.warning(f"Failed to list sandboxes for cleanup: {e}")

        if failed_ids:
            logger.warning(
                f"Failed to cleanup {len(failed_ids)} sandbox(es). "
                f"Failed IDs: {', '.join(failed_ids)}"
            )

        _running_sandboxes.get().clear()

        await close_client(client)
        _daytona_client.set(None)

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        client = AsyncDaytona()
        try:
            if id is not None:
                # Single sandbox cleanup
                try:
                    sandbox = await client.get(id)
                    await delete_sandbox(client, sandbox)
                    print(f"Successfully deleted sandbox {id}")
                except Exception as e:
                    print(f"[red]Error deleting sandbox {id}: {e}[/red]")
                    sys.exit(1)
            else:
                # Bulk cleanup
                sandboxes = await list_sandboxes(client, INSPECT_SANDBOX_LABEL)

                if not sandboxes:
                    print("No Daytona sandboxes found to clean up.")
                    return

                table = Table(
                    box=box.SQUARE,
                    show_lines=False,
                    title_style="bold",
                    title_justify="left",
                )
                table.add_column("Sandbox ID")
                for sb in sandboxes:
                    table.add_row(sb.id)
                print(table)

                is_interactive = sys.stdin.isatty()
                is_ci = "CI" in os.environ
                is_pytest = "PYTEST_CURRENT_TEST" in os.environ

                if is_interactive and not is_ci and not is_pytest:
                    if not Confirm.ask(
                        f"Are you sure you want to delete ALL {len(sandboxes)} sandbox(es) above?"
                    ):
                        print("Cancelled.")
                        return

                success_count = 0
                failure_count = 0

                for sb in sandboxes:
                    try:
                        await delete_sandbox(client, sb)
                        success_count += 1
                    except Exception as e:
                        print(f"[yellow]Error deleting sandbox {sb.id}: {e}[/yellow]")
                        failure_count += 1

                print(f"\n[green]Successfully deleted: {success_count}[/green]")
                if failure_count > 0:
                    print(f"[red]Failed to delete: {failure_count}[/red]")
                    sys.exit(1)
                else:
                    print("Complete.")
        finally:
            await close_client(client)
