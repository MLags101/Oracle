# Flying a log RotorID can actually use — ArduPilot

RotorID identifies your aircraft from the log. If the log does not contain a
deliberate, single-axis, wide-band excitation, no amount of analysis recovers one,
and the tool will tell you so rather than guess. Fifteen minutes of setup here is
worth more than anything else you can do for the quality of the result.

Back up your parameters before changing any of this.

## 1. Set up the sweep

One axis per flight. Set `SID_AXIS` for the axis you are doing:

| `SID_AXIS` | What it excites |
|---|---|
| 7 / 8 / 9 | rate controller input — roll / pitch / yaw. **Use these.** |
| 10 / 11 / 12 | mixer input directly — roll / pitch / yaw |
| 13 | mixer thrust |

The rate-loop injection points (7–9) are what the identification is built around.

| Parameter | Value | Why |
|---|---|---|
| `SID_F_START_HZ` | `0.05` | ArduPilot's published multicopter methodology |
| `SID_F_STOP_HZ` | `5` (larger craft) to `20` (small, fast) | above this there is nothing left to identify |
| `SID_T_FADE_IN` | `5` | avoids a step at the start of the record |
| `SID_T_REC` | `120` | the sweep itself; longer resolves the low end better |
| `SID_T_FADE_OUT` | `5` | |
| `SID_MAGNITUDE` | see below | |

`SID_MAGNITUDE` is the one that needs judgement: large enough that the response
stands clearly above the noise, small enough that no motor saturates. Start low and
raise it. You can adjust it in flight by setting `TUNE = 58` and assigning the tuning
knob — much faster than landing between attempts.

Repeat the sweep two or three times per axis. RotorID averages repeated sweeps by
summing their spectra, which is the standard method and measurably reduces variance.

## 2. Set up logging

### The one that decides whether the log is usable at all

`SCHED_LOOP_RATE` is how fast the controller runs. `LOG_BITMASK` is how fast it
writes down what it did, and those are two separate decisions. With **bit 0
(ATTITUDE_FAST) clear**, `RATE` and `ATT` go to the card on the 10 Hz medium-rate
schedule while the loop keeps running at 400 Hz, and nothing else in the file
says so.

A 10 Hz log carries no information above 5 Hz. RotorID will resample it onto its
analysis grid, coherence will look excellent all the way up — both signals were
smoothed by the same interpolator, so of course they agree — and the model that
comes back will be a confident description of a cubic spline. RotorID checks the
logged rate against the loop rate and refuses rather than doing that, but the fix
is on the aircraft, and it is one parameter:

```
LOG_BITMASK        bit 0  (ATTITUDE_FAST)   RATE and ATT at the loop rate
                   bit 2  (IMU)             VIBE: vibration and accel clipping
                   bit 12 (PID)             the PID* messages
                   bit 18 (IMU_FAST)        worth adding while you are in there
```

Bit 2 is worth setting even on a flight you are not tuning from. Vibration is the
one condition that makes every other measurement meaningless rather than merely
weaker: a frame shaking hard enough to move its own sensors produces a gyro trace
of the shaking, and an identification fitted to that will look confident and
describe nothing. Without `VIBE` in the log RotorID cannot rule that out, and says
so as `VIBRATION_NOT_LOGGED` rather than assuming the frame was fine.

To get **pre-filter** gyro as well — which is what lets RotorID verify its model of
your filter chain against your own aircraft rather than against arithmetic:

```
INS_LOG_BAT_MASK   1
INS_LOG_BAT_OPT    4      pre AND post filter, 1 kHz
```

On H7-class boards, prefer:

```
INS_RAW_LOG_OPT    9      bit 0 + bit 3: primary gyro, pre and post filter
```

**Turn batch logging back off** (`INS_LOG_BAT_MASK = 0`) when the tuning campaign is
done. It consumes log bandwidth and RAM continuously.

## 3. Fly it

- At altitude, with room to recover.
- Low wind. Wind is a disturbance the identification cannot distinguish from your
  excitation, and it lands squarely in the frequency band you care about.
- Hover first for 20–30 seconds. That gives the noise analysis a clean sample and
  fixes the operating point the notch filters track against.
- Then run the sweep. Hold position as best you can; do not fight the oscillation
  the sweep produces, since your corrections are indistinguishable from the
  vehicle's own response.

## 4. Check what you got

```bash
rotorid inspect flight.bin
```

That lists the signals found, the excitation segments detected, and anything
missing. If it reports no excitation, the sweep did not make it into the log and
nothing downstream is worth running.

`rotorid analyze` reports the rate each message was actually logged at whenever
it is too slow to design against, as `LOG_RATE_TOO_LOW`. That one is a blocker:
it is not a weaker answer, it is a different aircraft.

## What each thing buys you

| If this is missing | What you lose |
|---|---|
| `LOG_BITMASK` bit 0 (ATTITUDE_FAST) | Everything. `RATE` logs at 10 Hz and the log carries nothing above 5 Hz. |
| SYSTEMID sweep | Everything. Ordinary flight gives a low-confidence estimate at best. |
| `PID*` messages | Term-level diagnosis: slew limiting, D-term noise, integrator windup. |
| Pre-filter gyro (`ISBH`/`ISBD`) | Verification that the modeled filter chain matches your aircraft's. |
| ESC telemetry (`ESC.RPM`) | Per-motor notch tracking, and honest classification of which peaks track RPM. |
| `PM` (CPU load) | The check on whether expensive notch options fit in your board's headroom. |
| `VIBE` (`LOG_BITMASK` bit 2) | Any way to tell whether the gyro measured the aircraft or the frame. |
