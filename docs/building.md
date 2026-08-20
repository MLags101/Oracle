# Building the executable

One file, no Python on the machine it runs on, double-click and the wizard opens.

```bash
python -m pip install -e ".[gui,build]"
```

```bash
python build.py --log path/to/some-flight.bin
```

That produces `dist/rotorid.exe` on Windows and `dist/rotorid` on Linux, then runs the
finished binary's own self-check and prints what it found.

## PyInstaller cannot cross-compile

There is no flag for it. PyInstaller works by bundling *this* interpreter and *these*
compiled extension modules — numpy, scipy and Qt are all platform-specific machine
code — so a Windows binary has to be built on Windows and a Linux binary on Linux.
Run `build.py` on each platform you want to ship for. Nothing else about the command
changes; the script detects the host and names the artefact accordingly.

## Always pass `--log`

A build that produces a file and a build that works are different events, and on a
one-file bundle they come apart in a specific way. Some dependencies reach their data
through the filesystem rather than through an import, so PyInstaller's dependency
analysis never sees it and leaves it out. `pymavlink` is exactly that: its message
definitions are XML files. A build with the imports right and the data wrong reads
every log perfectly on the machine it was built on — where the installed copy is
still sitting on disk — and cannot open a single one anywhere else.

Reading a real log is the only step that catches that, so `build.py --log` is what
you should run and `build.py` on its own says so:

```
note: no log was read, so the message definitions were never exercised.
      Re-run with --log to check the part most likely to be missing.
```

## The self-check

It is not only a build step. It ships in the binary and in the Python package:

```bash
rotorid selftest path/to/some-flight.bin
```

It imports each layer in the order they would fail, reads the log, runs the analysis
and draws all eight stages offscreen — then says which of those worked.

A log the analysis *refuses* still passes. A refusal is the machinery working and
reaching a correct conclusion; if it counted as a failure, the only logs that could
verify a build would be the ones with nothing wrong with them.

The packaged binary is windowed, so it has no console to print to. That is why
`--out` exists:

```bash
rotorid selftest flight.bin --out result.json
```

Exit code 0 means every layer worked, 2 means one did not, and the JSON says which.

## What you get

Measured on the machine this was written on — Windows 11, Python 3.14, PySide6 6.11:

| | |
|---|---|
| Binary size | 116 MB |
| Build time | ~140 s |
| Startup | ~3.3 s, every launch |

A one-file build unpacks itself into a temporary directory each time it starts, so
that 3.3 s is paid on every launch rather than once on install. It is mostly Qt and
the two copies of OpenBLAS that numpy and scipy each bring their own of.

`build.py` excludes `control` and, through it, matplotlib: worth 21 MB and 30 s of
build time, and costs nothing, because `control` is used in exactly one place —
`AirframeModel.to_tf()`, which hands the identified model to somebody's own Python
session. There is no Python session inside a frozen GUI, so that escape hatch opens
onto nothing there. It still works normally in a `pip install`.

Excluding Qt submodules was tried and does **not** help. The large Qt libraries come
in as transitive binary dependencies of QtGui and QtWidgets rather than as Python
modules, so `--exclude-module` never sees them; a list of forty of them produced a
marginally *larger* bundle. The finding is recorded in `build.py` so nobody spends
the afternoon on it twice.

## Using the binary

Double-click it and the window opens with nothing loaded — pick a log with the file
picker, or drop one anywhere on the window. Dropping a log onto the executable's icon
works too: a single argument that is a file is read as "open this", because a windowed
program that rejected it would have nowhere to say so and would simply appear not to
start.

The CLI is still in there (`rotorid.exe analyze flight.bin ...`), but with no console
attached it cannot print, so use the Python package for scripted work.
