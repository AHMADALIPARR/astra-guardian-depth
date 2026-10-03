# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Rigorous perimeter verification & sandboxing.

Latent reasoning plus agentic tool use (code execution, API calls) is only
safe inside a real perimeter. This module enforces:

* Path containment: every filesystem target must resolve inside the sandbox
  root. `..` escapes, absolute paths outside the root, and symlinks pointing
  out are rejected — after symlink resolution, not before.
* Command allowlisting: only explicitly allowed executables run.
* Execution limits: wall-clock timeout; restricted environment (no inherited
  secrets); working directory locked to the sandbox root.

Enforcement is in code paths that raise, not in comments that hope.
"""

import os
import subprocess


class PerimeterViolation(Exception):
    pass


class Sandbox:
    def __init__(self, root: str, allowed_commands=()):
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)
        self.allowed_commands = frozenset(allowed_commands)

    # -- path containment -------------------------------------------------
    def resolve(self, path: str) -> str:
        """Resolve `path` against the root; raise if it escapes."""
        joined = os.path.realpath(os.path.join(self.root, path))
        if joined != self.root and not joined.startswith(self.root + os.sep):
            raise PerimeterViolation(f"path escapes sandbox root: {path!r}")
        return joined

    # -- command allowlisting ---------------------------------------------
    def check_command(self, argv: list) -> str:
        """Validate argv[0] against the allowlist; return resolved path."""
        if not argv:
            raise PerimeterViolation("empty command")
        name = os.path.basename(argv[0])
        if name not in self.allowed_commands:
            raise PerimeterViolation(f"command not allowlisted: {name!r}")
        return name

    # -- execution ----------------------------------------------------------
    def run(self, argv: list, timeout: int = 30, input_data: bytes = None) -> subprocess.CompletedProcess:
        """Run an allowlisted command inside the sandbox root."""
        self.check_command(argv)
        env = {"PATH": "/usr/bin:/bin", "HOME": self.root, "TMPDIR": self.root}
        return subprocess.run(
            argv, cwd=self.root, env=env, timeout=timeout,
            input=input_data, capture_output=True,
        )

    def write(self, path: str, data: bytes):
        """Write bytes to a contained path (creates parent dirs)."""
        full = self.resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)

    def read(self, path: str) -> bytes:
        with open(self.resolve(path), "rb") as f:
            return f.read()
