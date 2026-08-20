"""Build the standalone executable (milestone M11).

One file, no Python needed on the machine it runs on, double-click and the wizard
opens. Run this script on the platform you want a binary for:

    python build.py

PyInstaller cannot cross-compile. It works by bundling *this* interpreter and
*these* compiled extension modules, so a Windows build has to happen on Windows
and a Linux build on Linux. There is no flag for it and no way around it short of
a virtual machine; ``docs/building.md`` says so plainly rather than leaving it to
be discovered.

Three things here are not defaults and each exists because of a specific failure:

**pymavlink's message definitions.** They are XML files that pymavlink locates on
the filesystem at run time, not modules it imports, so PyInstaller's dependency
analysis never sees them. Leave them out and the build works perfectly on the
developer's machine -- where the installed copy is still on disk -- and cannot
open a single log anywhere else. Collected explicitly, along with the dialect
submodules, which are imported by name.

**control and matplotlib, excluded.** ``control`` is imported in exactly one
place: :func:`rotorid.core.analysis.model_eval.airframe_tf`, an escape hatch that
hands the identified model to somebody's own Python session as a
``control.TransferFunction``. There is no Python session inside a frozen GUI, so
the hatch opens onto nothing -- and ``control`` drags in the whole of matplotlib
including its font cache and backends, which is by far the largest thing that
would be in the bundle.

**The configuration file.** Every threshold the tool judges by lives in
``rotorid.toml``. A build without it does not fall back to defaults, because
there are no defaults in the code to fall back to; it fails on the first lookup.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
NAME = "rotorid"

#: Bundled because nothing imports them. ``(source, destination inside the bundle)``.
DATA: tuple[tuple[Path, str], ...] = ((ROOT / "rotorid.toml", "rotorid"),)

#: Packages whose contents are reached by name at run time rather than by import,
#: which is invisible to static analysis.
COLLECT_SUBMODULES = ("pymavlink", "pyulog")
COLLECT_DATA = ("pymavlink",)

#: Left out deliberately. See the module docstring. Measured on this machine:
#: with these excluded the binary is 116 MB and takes 139 s to build, without them
#: 137 MB and 172 s -- almost all of the difference being matplotlib, which arrives
#: as a dependency of ``control`` and is never drawn with.
#:
#: Excluding Qt submodules was tried and is not here, because it does nothing: the
#: large Qt libraries come in as transitive dependencies of QtGui and QtWidgets
#: rather than as modules, so ``--exclude-module`` never sees them. The bundle came
#: out marginally *larger* with a list of forty of them in place.
EXCLUDE = (
    "control",
    "matplotlib",
    "pytest",
    "IPython",
    "tkinter",
    "sympy",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="a flight log to verify the finished binary against. Strongly "
        "recommended: reading a real log is the only check that exercises the "
        "message definitions, which is what a broken bundle gets wrong.",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="build only, do not run the binary"
    )
    parser.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    args = parser.parse_args(argv)

    if not _have_pyinstaller():
        print(
            "build.py needs PyInstaller:\n    python -m pip install -e '.[build]'",
            file=sys.stderr,
        )
        return 1

    if args.clean:
        for directory in (BUILD, DIST):
            shutil.rmtree(directory, ignore_errors=True)

    print(f"building for {platform.system()} {platform.machine()}, Python {_python_version()}")
    started = time.perf_counter()
    completed = subprocess.run(_command(), cwd=ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode

    binary = _artefact()
    if not binary.exists():
        print(f"PyInstaller reported success but {binary} is not there", file=sys.stderr)
        return 1

    size_mb = binary.stat().st_size / 1e6
    print(f"\nbuilt {binary} -- {size_mb:.0f} MB in {time.perf_counter() - started:.0f} s")

    if args.skip_verify:
        print("not verified. A build that produces a file and one that works are different events.")
        return 0
    return _verify(binary, args.log)


def _command() -> list[str]:
    """The PyInstaller invocation, assembled so it can be read and argued with."""
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        # No console. The whole point of the packaged build is that double-clicking
        # it opens the wizard, and a console window flashing up first is the
        # difference between an application and a script someone wrapped.
        "--windowed",
        "--name",
        NAME,
        "--paths",
        str(ROOT / "src"),
    ]
    for source, destination in DATA:
        command += ["--add-data", f"{source}{_data_separator()}{destination}"]
    for package in COLLECT_SUBMODULES:
        command += ["--collect-submodules", package]
    for package in COLLECT_DATA:
        command += ["--collect-data", package]
    for module in EXCLUDE:
        command += ["--exclude-module", module]
    command.append(str(ROOT / "src" / "rotorid" / "__main__.py"))
    return command


def _verify(binary: Path, log: Path | None) -> int:
    """Run the built binary's own self-check.

    Through the binary rather than through this interpreter, which is the entire
    point: the failure being looked for is one that only exists inside the bundle.
    """
    report = DIST / "selftest.json"
    command = [str(binary), "selftest"]
    if log is not None:
        command.append(str(log))
    command += ["--out", str(report)]

    print(f"\nverifying: {' '.join(command)}")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started

    if report.exists():
        print(report.read_text(encoding="utf-8"))
    if completed.returncode == 0:
        print(f"verified in {elapsed:.0f} s (includes one-file unpacking on every launch)")
        if log is None:
            print(
                "note: no log was read, so the message definitions were never exercised. "
                "Re-run with --log to check the part most likely to be missing."
            )
        return 0

    print(f"the binary built but failed its own self-check: exit {completed.returncode}")
    return completed.returncode


def _artefact() -> Path:
    return DIST / (f"{NAME}.exe" if platform.system() == "Windows" else NAME)


def _data_separator() -> str:
    """``--add-data`` uses the platform path separator between source and target."""
    return ";" if platform.system() == "Windows" else ":"


def _have_pyinstaller() -> bool:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        return False
    return True


def _python_version() -> str:
    return ".".join(str(n) for n in sys.version_info[:3])


if __name__ == "__main__":
    raise SystemExit(main())
