"""Entry point for ``python -m rotorid``.

The console script installed by ``pyproject.toml`` is the normal way in; this
exists so the tool is still reachable in an environment where the script is not
on PATH, which is the common case on Windows.
"""

from __future__ import annotations

from rotorid.cli import main

raise SystemExit(main())
