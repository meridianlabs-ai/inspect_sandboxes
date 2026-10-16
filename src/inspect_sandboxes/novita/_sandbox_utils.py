"""Shared helpers for Novita sandbox environments"""

from __future__ import annotations

import shlex
import uuid
from pathlib import PurePosixPath

from novita_sandbox.core import AsyncSandbox

from ._retry import standard_retry

FILE_REQUEST_TIMEOUT = 1800
# Writes above this size are split into chunks and reassembled in the sandbox:
# a single large upload to the envd files route fails (uncompressed → the
# connection resets with httpx.ReadError; gzipped → the gateway 504s), while a
# 1 GiB Sample.files upload in 8 MiB chunks is reliable.
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024


async def write_sandbox_file(
    sandbox: AsyncSandbox, file: str, data: bytes, user: str | None = None
) -> None:
    """Write *data* to *file* in the sandbox, chunking large payloads.

    Each chunk is uploaded to a temp file and appended to the target, so peak
    extra disk usage stays at one chunk (accumulating every chunk before
    reassembly can exhaust the disk — /tmp may be a small tmpfs).
    """
    if len(data) <= UPLOAD_CHUNK_SIZE:
        await _write_file_bytes(sandbox, file, data, user)
        return

    temp = f"/tmp/.inspect-chunk-{uuid.uuid4().hex}"
    quoted_temp = shlex.quote(temp)
    quoted = shlex.quote(file)
    parent = shlex.quote(str(PurePosixPath(file).parent))
    try:
        await sandbox.commands.run(
            f"mkdir -p {parent} && : > {quoted}",
            user=user,
            timeout=FILE_REQUEST_TIMEOUT,
        )
        for offset in range(0, len(data), UPLOAD_CHUNK_SIZE):
            await _write_file_bytes(
                sandbox, temp, data[offset : offset + UPLOAD_CHUNK_SIZE], user
            )
            # Not retried: a retry after a lost success reply would append the
            # chunk twice.
            await sandbox.commands.run(
                f"cat {quoted_temp} >> {quoted}",
                user=user,
                timeout=FILE_REQUEST_TIMEOUT,
            )
    finally:
        try:
            await sandbox.commands.run(f"rm -f {quoted_temp}", user=user, timeout=60)
        except Exception:
            # Best-effort cleanup: the SDK leaks raw httpx/httpcore transport
            # errors, and a cleanup failure must never mask the primary error.
            pass


@standard_retry
async def _write_file_bytes(
    sandbox: AsyncSandbox, file: str, data: bytes, user: str | None
) -> None:
    # gzip compresses the request body in transit; it only takes effect with
    # use_octet_stream — the default multipart path ignores the flag.
    await sandbox.files.write(
        file,
        data,
        user=user,
        request_timeout=FILE_REQUEST_TIMEOUT,
        gzip=True,
        use_octet_stream=True,
    )
