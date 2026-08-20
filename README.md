# RotorID

Offline multirotor **filter and PID** tuning, from ArduPilot `.bin` and PX4 `.ulg`
flight logs.

RotorID reads a flight log, identifies the vehicle's rate-loop dynamics in the
frequency domain, and designs a *joint* filter and gain configuration against
explicit stability margins — then shows its working, so you can see why the new
numbers are better rather than taking them on trust.

**Nothing is ever written to a vehicle.** RotorID writes parameter files that a human
loads deliberately, one stage at a time.

---

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[gui,dev]"
```

On Linux and macOS the interpreter is at `.venv/bin/python` instead. Drop `gui` if you
only want the command line; drop `dev` if you are not running the tests.

## Use it

### The window

```bash
rotorid
```

That is the whole invocation. The window opens with nothing loaded and the first
screen asks for a log — a file picker, or drag one anywhere onto the window. You can
still name the file up front if you prefer:

```bash
rotorid gui flight.bin
```

Eight stages, in the order the analysis actually depends on: **Load**, **Health &
Noise**, **Segment**, **Identify**, **Filters**, **Design**, **Review & Export**,
**Next Flight**. Health comes before Identify because a model fitted to a shaking
frame is not a weak model, it is confident nonsense.

The Filters and Design stages are a live sandbox. Move a control and the margins, the
predicted step, the predicted spectrum and the phase budget all move together, so the
trade-off is something you watch rather than something you are told about. Every
recommended number carries a **why?** that opens its trace: the model it came from,
the band that model is valid over, the constraint that stopped it going further, and
what was rejected on the way.

### The command line

Everything the window can do, the CLI can do headlessly. On an install without the
`gui` extra, bare `rotorid` prints this list instead of opening a window.

**What is in this log, and what is missing:**

```bash
rotorid inspect flight.bin
```

**Identify and recommend, with a report you can read offline:**

```bash
rotorid analyze flight.bin -o report.html
```

**The full run — report, staged parameter files, and a saved session:**

```bash
rotorid analyze flight.bin --axes roll,pitch,yaw -o report.html --export ./params --session flight.rotorid
```

**Reopen a saved analysis without re-reading the log:**

```bash
rotorid session flight.rotorid
```

Useful flags on `analyze`:

| Flag | What it does |
|---|---|
| `--axes roll,pitch` | Which axes to analyse. Default is all three. |
| `--conservatism 0.7` | 0 is aggressive, 1 is docile. Default 0.5. |
| `--export DIR` | Staged `.param` files, one per test flight, in the order to fly them. |
| `--session FILE` | The whole analysis in one `.rotorid` bundle, reopenable later. |
| `--acknowledge CODE,CODE` | Accept named blocking findings so the export can proceed. Recorded in the file header. |
| `--config FILE` | Override thresholds from `rotorid.toml`. |
| `--json` | Machine-readable output. |

### What the export gives you

Filters and gains land in **separate files**, numbered in flight order, because they
are separate flights. Load the filter file, fly it, confirm the noise came down, and
only then load the gains. Each file's header records the log it came from, the tool
version, the config hash, and any finding you acknowledged to get it written.

If anything blocking is unresolved, nothing is written at all — a partial export is
worse than none, because the files that did appear look complete.

## What your log needs

RotorID tells you what is missing and what each gap costs, so the fastest way to find
out is to run `rotorid inspect` on a log you already have. The full setup is in
[`docs/logging-setup-ardupilot.md`](docs/logging-setup-ardupilot.md) and
[`docs/logging-setup-px4.md`](docs/logging-setup-px4.md).

The two that decide whether a log is usable at all:

- **`LOG_BITMASK` bit 0 (ATTITUDE_FAST).** With it clear, `RATE` and `ATT` go out on
  the 10 Hz medium-rate schedule regardless of how fast your loop runs, and the log
  carries nothing above 5 Hz. Nothing downstream notices on its own — the resampler
  splines it, coherence stays high because both signals were smoothed by the same
  interpolator, and a confident model comes back fitted to the shape of a cubic.
  RotorID checks the logged rate against the loop rate and refuses instead.
- **`LOG_BITMASK` bit 2 (IMU).** Gives you `VIBE`: vibration and accelerometer
  clipping. Without it RotorID cannot tell whether the gyro measured your aircraft or
  your frame, and says so rather than assuming the frame was fine.

A deliberate SYSTEMID sweep is much better evidence than ordinary flight and is what
`high` confidence requires — but ordinary flight is what most people have, and making
it answer as much as it honestly can is the current line of work.

## Why it exists

Autotune gives you gains. It does not tell you that your notch filter never tracked
the motor peak, that your D-term is amplifying noise into the motors, or that your
gyro low-pass is costing more phase than the gains can buy back. On most real vehicles
the filter configuration, not the gain arithmetic, is what limits achievable
bandwidth — so RotorID treats filters and gains as one design problem with one shared
phase budget.

They are solved as a fixed point rather than in sequence: the filter choice depends on
where the loop crosses over, the crossover depends on the phase the filters cost, and
the D-term noise ceiling depends on a derivative gain that is not known until the
gains are designed. A filter change is only recommended when it widens
disturbance-rejection bandwidth; otherwise the report says the filters you already
have are the right ones, and shows the spectrum that says so.

## Two things it refuses to do

- **Guess.** Poor coherence, a missing message, a filter model that disagrees with the
  log, a frame shaking hard enough to move its own sensors — each surfaces as a
  blocking or warning finding. Silent degradation to a confident-looking bad tune is
  the failure mode the tool exists to prevent.
- **Hide its reasoning.** Every recommended number records the model it came from, the
  band it was identified over, the constraint that bounded it, and what the
  alternatives cost.

## How the identification works

Every flight log is closed-loop data. The mixer command is the controller's own
output, so it carries gyro noise fed back through the controller, and estimating the
plant directly from it is biased towards the inverse of the controller rather than
towards the aircraft. ArduPilot's SYSTEMID sweep does not escape this: the waveform is
injected *inside* the loop, at the rate target for `SID_AXIS` 7–9 and at the mixer for
10–12.

RotorID uses the Joint Input-Output estimator instead. For any exogenous signal `r`
that is measured, whatever the controller is and wherever `r` enters:

```
G(jw) = (r -> y) / (r -> u)
```

which is the bare airframe for both injection points. The instrument is taken from a
ladder, best first, and the result always names which rung it got: the injected chirp,
then the pilot's commanded lean angle (exogenous in Stabilize and AltHold, and what
makes an ordinary flight identifiable at all), then the rate setpoint, then nothing —
at which point the estimate is labelled biased and blocks.

On a simulated flight with a known airframe of `K = 12`, the instrument-variable
estimator recovers 11.4–12.3 where the direct estimator returns 33–61 and reports high
coherence to 100 Hz, because up there both signals are the same noise arriving through
the same loop.

## Both stacks

PX4 differs from ArduPilot in more than parameter names: its notches are true nulls
with no attenuation setting, its D-term filter is 2-pole rather than 1-pole, its gains
are stored in standard form behind a `K` factor, and it has no throttle-derived notch
tracking at all — so a PX4 log with neither ESC RPM nor the onboard FFT gets a refusal
rather than a static notch pinned to one hover point. Each of those, taken from the
wrong stack, produces numbers that look entirely reasonable and are wrong.

## Status

Early development. The full specification and milestone plan is in
[`rotorid-implementation-plan.md`](rotorid-implementation-plan.md).

| Milestone | State |
|---|---|
| M0 — skeleton, config, data contracts | done |
| M1 — ArduPilot CLI walking skeleton | done; reads real `.bin` logs |
| M2 — filter engine | core done; firmware-parity test needs a real pre/post-filter log |
| M3 — noise, peak classification, notch recommendation | done against synthetic logs |
| M4 — joint filter + gain optimizer | done; re-solve ~150 ms against the 300 ms budget |
| M5 — guidance engine and staged flight plan | done |
| M6/M7 — GUI shell, all eight stages, live sandbox | done |
| M8 — staged `.param` export, session save/load, safety gates | done |
| M9 — PX4 `.ulg` reader, filter chain and design | done against synthetic uLog bytes |
| M10 — validation mode | in progress |
| M11 — single-file executables | planned |
| General flight logs — unbiased estimator, vibration, step response | in progress |

The pipeline runs end to end: log → segment → frequency response with the loop
divided out → filter chain deconvolved → airframe fit → noise spectrum → filters and
gains designed together → HTML report. It recovers a known airframe from a synthetic
sweep within 10% on `K`, `wn`, `zeta` and `tau`, classifies known motor harmonics and
a known frame resonance correctly, and reports margins that hold when the loop is
rebuilt from the published parameters alone.

Real ArduPilot logs read correctly and are refused for the right reason — the ones on
hand were flown without ATTITUDE_FAST, so they carry `RATE` at 10 Hz. A log flown with
that bit set is what the current work still needs to close against real data.

## Develop

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/python -m ruff check src tests
```

```bash
.venv/Scripts/python -m ruff format --check src tests
```

```bash
.venv/Scripts/python -m mypy
```

All four run in CI on Python 3.12 and 3.13. `mypy` is strict over `rotorid.core`,
which holds every contract and all the numerics.

## Layout

```
src/rotorid/core/     analysis library, no Qt imports anywhere
src/rotorid/gui/      PySide6 presentation layer over the core
rotorid.toml          every threshold, with its source
docs/                 how to set your vehicle up to produce a usable log
tests/synthetic/      ground-truth generators, including a closed-loop simulator
```

Thresholds never live in code. Every number the tool judges by is in `rotorid.toml`
with a comment naming where it came from — a firmware document, a published
methodology, or an explicit project default — and the resolved config is hashed into
each saved session so a result is reproducible.

## Safety

Output goes onto a flying aircraft.

- Back up your parameters first.
- Apply changes one stage at a time: filters, then gains, in the order the export
  numbers them.
- Test at altitude with room to recover.

A RotorID recommendation is a well-justified starting point with designed margins. It
is not a validated final tune.
