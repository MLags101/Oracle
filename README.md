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

### As an executable

One file, no Python needed. Build it once for your platform:

```bash
python build.py --log path/to/some-flight.bin
```

That produces `dist/rotorid.exe` on Windows or `dist/rotorid` on Linux, then runs the
finished binary's own self-check to confirm it can read a log and draw every stage —
building a file and having one that works are different events, and on a one-file
bundle they come apart in ways that only appear on a machine without Python. Details,
including why the Linux binary has to be built on Linux, are in
[`docs/building.md`](docs/building.md).

### As a Python package

Python 3.11 or newer.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[gui,dev]"
```

On Linux and macOS the interpreter is at `.venv/bin/python` instead. Drop `gui` if you
only want the command line; drop `dev` if you are not running the tests; add `build`
if you want to produce an executable.

### Without installing anything

The package needs no install of its own — put `src` on the path and run it from the
checkout. The dependencies still have to exist somewhere.

```bash
PYTHONPATH=src python -m rotorid
```

## Use it

### The window

```bash
rotorid
```

That is the whole invocation. The window opens with nothing loaded and the first
screen asks for two things: which log, and **what kind of flight it was**. You can
still name the file up front if you prefer:

```bash
rotorid gui flight.bin
```

**Dropping a log in analyses it.** You do not have to ask twice: opening a file
starts the run, and a couple of minutes later the findings panel says what is
wrong with the flight and what to do about it. The toolbar's **Analyse** button
is for re-running after you have moved the conservatism slider, and the
**Analyse on open** switch beside it turns the automatic run off for anyone who
would rather look at the log first.

Nine stages, in the order the analysis actually depends on: **Load**, **Health &
Noise**, **Segment**, **Identify**, **Filters**, **Design**, **Review & Export**,
**Next Flight**, **Validate**. Health comes before Identify because a model fitted
to a shaking frame is not a weak model, it is confident nonsense. Validate comes
last because it is what you do when you come back with the flight the previous
stage told you to fly.

The rail down the left is that order, drawn as one numbered sequence rather than
nine names. Each step says what it is for in a few words, ticks off once you have
been there, and — when it is not open yet — says what would open it. "Open a log
first" and "nothing could be identified in this log" are different problems with
different fixes, and the rail names which one you have.

The Filters and Design stages are a live sandbox. Move a control and the margins, the
predicted step, the predicted spectrum and the phase budget all move together, so the
trade-off is something you watch rather than something you are told about. Every
recommended number carries a **why?** that opens its trace: the model it came from,
the band that model is valid over, the constraint that stopped it going further, and
what was rejected on the way.

### Two kinds of flight

A tuning flight and an ordinary one are different evidence, and RotorID asks which
one you have rather than guessing. Guessing is what it used to do — look for a
sweep, and quietly fall back to stick input when it found none — and the fallback
was invisible in every number downstream.

| | **Tuning flight** | **General flight** |
|---|---|---|
| What it is | a SYSTEMID sweep, or an autotune run | an ordinary flight |
| Identified from | deliberate excitation only | ordinary stick activity only |
| Confidence ceiling | `high` | `medium`, however well it fits |
| Design | as bold as you ask for | held to at least 0.6 conservatism |
| Operating-point sensitivity | not available | **yes** |

Neither is a degraded version of the other. A two-minute sweep excites a known band
on one axis at a time with a signal the controller did not choose, which is what
makes a wide-band fit trustworthy — but it is flown at one throttle on one battery
state, so it has nothing to say about how the airframe gain moves across the
envelope. An ordinary flight visits the envelope, and that is a measurement a sweep
cannot make.

The declaration is honoured, not overridden. A log declared as a tuning flight is
never quietly identified from stick input; a log declared general never uses a sweep
that happens to be in it. If the file disagrees with what you said, RotorID says so
(`LOG_KIND_MISMATCH`) rather than resolving it silently. Detection is the default and
is right whenever the excitation was actually recorded — and only recorded excitation
counts, because `SID_AXIS` says what *would* be injected, not that anything was.

### Did it work?

The Validate stage, and `rotorid validate`, take a second log — a flight flown after
applying a recommendation — and put what the tool predicted next to what the aircraft
did:

```bash
rotorid validate before.bin after.bin --session before.rotorid -o comparison.html
```

Three claims live in that comparison and it keeps them apart. *The aircraft changed*
and *the aircraft improved* need only the two logs. *The tool was right* needs the
saved session from the first analysis, because nothing else records what was
predicted — without one, the report says at the top that it is an outcome comparison
rather than a validation, instead of leaving a column quietly empty.

It checks two predictions. The closed-loop step, against the step deconvolved from
the new flight. And the predicted post-filter spectrum against the measured one —
the half of a recommendation that normally goes unchecked, because a gain change
announces itself in how the aircraft feels and a notch two hertz off the motor line
does not.

A prediction is only ever tested against a flight that flew it. The staged export
deliberately loads filters one flight and gains the next, so an after-log flying the
old gains is the *expected* outcome of following the plan; that reads as
`TUNE_NOT_APPLIED`, not as a failed prediction.

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

**Set the vehicle up before you fly it** — the parameters that decide whether the
log is usable at all, as a file rather than as a checklist:

```bash
rotorid profile --stack ardupilot -o collect.param
```

**Filters only, without identifying the airframe** — works on a log with no usable
excitation in it, because the noise does not need a model:

```bash
rotorid filters flight.bin
```

**Re-render or re-export from a saved session**, without the log and without
re-analysing:

```bash
rotorid report flight.rotorid -o report.html
```

```bash
rotorid recommend flight.rotorid -o ./params --acknowledge VIBRATION_HIGH
```

**Check that this build works, and if not, which layer:**

```bash
rotorid selftest flight.bin
```

Useful flags on `analyze`:

| Flag | What it does |
|---|---|
| `--axes roll,pitch` | Which axes to analyse. Default is all three. |
| `--kind general\|tuning` | What the flight was. Default is to detect it from the log. |
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
`high` confidence requires — but ordinary flight is what most people have, and it is
supported on its own terms rather than as a degraded sweep. See **Two kinds of
flight** above for what each one buys and what it costs.

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
  the failure mode the tool exists to prevent, and it is why a log declared as a
  tuning flight is refused rather than quietly identified from stick input.
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

Every milestone in the plan is implemented. What is not closed is *evidence*:
several of them are verified against synthetic logs whose ground truth is known
exactly, and are still waiting on real flight data to confirm against. The table
says which is which, because "implemented" and "shown to be right on a real
aircraft" are different claims and this tool of all tools should not blur them.

The full specification and milestone plan is in
[`rotorid-implementation-plan.md`](rotorid-implementation-plan.md).

| Milestone | State |
|---|---|
| M0 — skeleton, config, data contracts | done |
| M1 — ArduPilot CLI walking skeleton | done; reads real `.bin` logs |
| M2 — filter engine | core done; reads real pre-filter gyro, parity test needs a pre+post log |
| M3 — noise, peak classification, notch recommendation | done against synthetic logs |
| M4 — joint filter + gain optimizer | done; re-solve ~150 ms against the 300 ms budget |
| M5 — guidance engine and staged flight plan | done |
| M6/M7 — GUI shell, all nine stages, live sandbox | done |
| M8 — staged `.param` export, session save/load, safety gates | done |
| M9 — PX4 `.ulg` reader, filter chain and design, autotune ingest | done against synthetic uLog bytes |
| M10 — validation mode | done; before/after screen, `rotorid validate`, HTML comparison |
| M11 — single-file executables | done; `build.py`, verified by the binary's own self-check |
| General flight logs — declared kind, unbiased estimator, vibration, step response | done |
| Operating-point sensitivity (spec 5.9) | done against a synthetic mis-set thrust curve |
| Real-log hardening — units, ground data, aliasing, shared spectral grid | done; see below |

The pipeline runs end to end: log → segment → frequency response with the loop
divided out → filter chain deconvolved → airframe fit → noise spectrum → filters and
gains designed together → HTML report. It recovers a known airframe from a synthetic
sweep within 10% on `K`, `wn`, `zeta` and `tau`, classifies known motor harmonics and
a known frame resonance correctly, and reports margins that hold when the loop is
rebuilt from the published parameters alone.

Validation mode closes the loop on the synthetic side: gains recommended from one
simulated flight, re-flown through the same simulated airframe, produce a measured
step that matches the prediction. Nothing about that is true by construction — the
prediction comes from a fitted model driven through the controller model and the
measurement from a regularized deconvolution — so it holds only if the
identification, the controller model and the step recovery are all right at once.

A well-instrumented real log — half an hour of ArduPilot 4.7 with `RATE` at the loop
rate, raw gyro at 1.6 kHz and the onboard FFT running — found five defects that no
synthetic fixture could have, because the fixtures were built out of the same
assumptions the code was:

- **`ATT.Yaw` entered the analysis in degrees** under a key whose contract says
  radians, because its `degheading` unit was unrecognised and unrecognised units were
  passed through unconverted. An unknown unit now drops the signal instead: a missing
  signal is something this tool reports, a mislabelled one is not reportable by
  anything.
- **Per-segment Welch windows could not be combined.** Segments are merged by summing
  spectra, which needs one frequency grid. Invisible on a fixture whose segments are
  all the same length; fatal on eighteen stick inputs that are not.
- **"Excited" meant 30% of the axis's peak**, which is right inside a sweep and
  backwards across half an hour of flying, where the peak is set by the single most
  violent moment and ordinary stick work never approaches it.
- **Ten of eighteen candidate windows were on the ground**, disarmed, with the yaw
  controller swinging against nothing. An aircraft that cannot rotate is not a weak
  measurement of the rate loop, it is a measurement of the landing gear, and nothing
  downstream can tell. The vehicle's own landing-state verdict now gates the search.
- **Raw gyro aliased and rang.** Logged faster than the analysis grid it folded its
  top half down into the notch designer's band; splined across logging dropouts it
  produced excursions of forty thousand radians per second.

That log yields no airframe model — the pilot flew roll and pitch together throughout
and no axis was excited alone for five seconds — and the tool now says so with the
numbers rather than a shrug. What it does yield is a *measured* pre-filter spectrum
from `GYR` instead of a reconstructed one, and motor tracking from the onboard FFT on
a vehicle with no ESC telemetry.

Still waiting on data rather than on code:

- the firmware-parity check on the filter engine (M2), which needs a log with pre-
  *and* post-filter gyro in it — `INS_LOG_BAT_OPT = 4`, which `rotorid profile` sets;
- a flight with deliberate single-axis input, to put an identification against a real
  aircraft rather than a simulated one;
- a real before/after pair, for the prediction check;
- a real PX4 autotune log, to confirm the ingest against bytes the vendor wrote
  rather than bytes we wrote.

Each is a test that exists and is waiting for its input, not a feature that is
missing.

## Develop

```bash
.venv/Scripts/python -m pytest
```

Characterization against whatever real logs are sitting in `logs/` is opt-in,
because a modern log with raw IMU logging on runs to hundreds of megabytes and
takes minutes to parse:

```bash
.venv/Scripts/python -m pytest -m real_log
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
src/rotorid/core/logkind.py   what each kind of flight unlocks, and what it caps
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
