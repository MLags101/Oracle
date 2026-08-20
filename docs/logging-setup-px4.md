# Flying a log RotorID can actually use — PX4

RotorID identifies your aircraft from the log. If the log does not contain a
deliberate, single-axis, wide-band excitation, no amount of analysis recovers one,
and the tool will say so rather than guess.

PX4 makes this harder than ArduPilot in one specific way, and it is worth knowing
before you fly: **there is no SYSTEMID mode.** ArduPilot can inject a frequency
sweep and log the injected signal on its own, which is by some distance the best
identification input there is. On PX4 the excitation has to come from somewhere
else, and the log records only the total command — the sweep and the controller's
reaction to the aircraft, mixed together. Expect a lower confidence rating from a
PX4 log than from an equivalent ArduPilot one. That is a real difference in the
evidence, not a limitation of the tool.

Back up your parameters before changing any of this.

## 0. Let RotorID write the file

```bash
rotorid profile --stack px4 -o collect.param
```

Logging only, safe to leave loaded. Add `--which sweep` to configure the autotune
as well, with `MC_AT_APPLY = 0` so it identifies and reports without writing gains
to the vehicle — which is what you want when the analysis is being done offline.
The rest of this page is what those files contain and why.

## 1. Get excitation into the log

Two options, best first.

**Multicopter autotune.** `MC_AT_EN = 1`, then run autotune from your ground
station. It injects a deliberate excitation on each axis in turn and identifies
its own model, so it produces exactly the kind of single-axis, wide-band data
this tool wants. Fly it in stable air, at altitude, with room to recover.

RotorID reads autotune's own conclusion out of the log as well as the flight data,
and compares the two. Where they agree you have two independent estimates of one
aircraft, which is evidence neither can produce alone; where they disagree
(`VENDOR_TUNE_DISAGREES`) one of them describes a vehicle that does not exist, and
that is worth knowing before you fly either. Set `MC_AT_APPLY = 0` so autotune
reports without writing its gains to the vehicle.

**Deliberate stick sweeps.** These are a *general* flight as far as RotorID is
concerned — nothing in the log records that anybody asked for them — so load them
with `--kind general` and expect `medium` confidence at best. That is the truth
about the evidence rather than a limitation: what makes an autotune run worth more
is that the vehicle wrote down that it was exciting itself. One axis per flight. Hold the other two as still as
you can and sweep the stick from very slow (about one cycle every few seconds) to
as fast as you can move it, over 60–120 seconds, without saturating the motors.
Slow to fast, smoothly, not a series of flicks: RotorID needs low-frequency
information as much as high, and a burst of fast stick has none of it.

The cross-axis requirement is not a nicety. The identification assumes one axis
at a time, and RotorID will reject a window where the other two axes were active
— so a sweep flown while fighting a crosswind on pitch produces nothing usable
on roll.

## 2. Set up logging

| Parameter | Value | Why |
|---|---|---|
| `SDLOG_PROFILE` | include **High rate** (bit 1) | `vehicle_angular_velocity` and `vehicle_torque_setpoint` at full rate |
| `SDLOG_MODE` | `0` or `1` | logging from boot, so the whole flight is there |
| `IMU_GYRO_RATEMAX` | leave as configured | RotorID reads it to know the loop rate |

### The one that changes what RotorID can tell you

The gyro PX4 logs in `vehicle_angular_velocity` is **post-filter** — it has
already been through `IMU_GYRO_CUTOFF` and every notch. So to see the noise your
filters are removing, RotorID has to divide the modelled filter chain back out of
what was logged, and inside a deep notch that reconstruction is blind: the peak
the notch removed cannot be recovered from the quiet it produced.

The fix is to log the raw gyro as well:

| Parameter | Value | Why |
|---|---|---|
| `SDLOG_PROFILE` | add **Sensor comparison** / raw IMU (bit 5) | logs `sensor_gyro_fifo`, the unfiltered gyro |

With it, the pre-filter spectrum is a measurement. Without it, it is a
reconstruction, and RotorID will say so on every plot that uses it — and will
refuse to remove a notch on the strength of quiet that notch itself produced.

### If you want a notch recommendation

A dynamic notch has to follow the motors, and PX4 gives it exactly two ways to
know where they are. Without one of them, RotorID will not recommend a tracking
notch at all — PX4 has no throttle-derived mode to fall back on, and a static
notch pinned to whatever frequency this flight happened to hover at would be in
the wrong place at every other throttle setting.

| Parameter | Value | Why |
|---|---|---|
| ESC RPM telemetry | enabled on your ESCs (DShot bidirectional, or a telemetry wire) | `esc_status` is the best source there is |
| `IMU_GYRO_FFT_EN` | `1` | the fallback: PX4 finds the peak itself, but lags fast throttle changes |

Also log a steady hover of 20–30 seconds somewhere in the flight, with the
throttle moving gently up and down rather than pinned. RotorID measures the noise
spectrum over the steadiest stretch it can find, but it decides whether a peak
*tracks the motors* by watching it move — so a flight at exactly one throttle
setting cannot tell a motor harmonic from a frame resonance, and the two want
opposite fixes.

## 3. What RotorID reads

| Topic | Used for |
|---|---|
| `vehicle_angular_velocity` | the rate measurement — **post-filter** |
| `vehicle_torque_setpoint` (or `actuator_controls_0` on older releases) | the identification input |
| `vehicle_attitude` | the outer loop, converted from the logged quaternion |
| `esc_status` | motor speed, for notch tracking and the operating point |
| `sensor_gyro_fifo` | raw gyro, if you logged it |
| `battery_status`, `cpuload` | operating point, and whether the board can afford the filters |

## 4. Before you fly

- Props balanced, arms tight, flight controller mount in good condition. A frame
  resonance is a mechanical fault; a notch only hides it, and hides it from you
  as well as from the controller.
- Fly in calm air. Wind is an unmeasured input, and it lowers coherence exactly
  where the identification needs it.
- Fly at a normal hover weight. The airframe gain moves with mass, and the model
  describes the aircraft you flew.
