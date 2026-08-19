"""Evaluate a fitted airframe model, in the frequency domain and as a transfer function.

The frequency-domain path uses the exact ``exp(-tau*s)`` delay. The ``control``
transfer function path has to approximate it with a Pade expansion, because a
rational object cannot carry a true delay -- so anything that decides a margin or a
gain should use :func:`airframe_response`, and :func:`airframe_tf` exists for
interoperability and plotting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rotorid.core.types import AirframeModel
from rotorid.core.units import hz_to_rads

__all__ = ["PADE_ORDER", "airframe_response", "airframe_tf", "required_params"]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

#: Pade order used when a rational approximation of the delay is unavoidable.
PADE_ORDER = 3

#: Parameters each model structure needs. Used to validate a fit result.
_REQUIRED: dict[str, tuple[str, ...]] = {
    "so_delay": ("K", "wn", "zeta", "tau"),
    "fo_delay": ("K", "T", "tau"),
    "so_zero_delay": ("K", "wn", "zeta", "wz", "tau"),
}


def required_params(structure: str) -> tuple[str, ...]:
    """Parameter names a given model structure requires.

    Raises:
        KeyError: if the structure is unknown.
    """
    try:
        return _REQUIRED[structure]
    except KeyError:
        raise KeyError(
            f"unknown model structure {structure!r}; known: {sorted(_REQUIRED)}"
        ) from None


def _check(model: AirframeModel) -> None:
    """Raise if the model is missing a parameter its structure needs."""
    missing = [p for p in required_params(model.structure) if p not in model.params]
    if missing:
        raise ValueError(
            f"{model.structure} model for {model.axis} is missing {missing}; "
            f"has {sorted(model.params)}"
        )


def airframe_response(model: AirframeModel, f_hz: FloatArray | float) -> ComplexArray:
    """Exact complex response of the identified airframe at the given frequencies.

    Structures:

    * ``so_delay``      ``K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2) * exp(-tau*s)``
    * ``fo_delay``      ``K / (T*s + 1) * exp(-tau*s)``
    * ``so_zero_delay`` as ``so_delay``, with an extra ``(s/wz + 1)`` numerator term

    Args:
        model: The fitted model. ``wn`` and ``wz`` are in rad/s, ``tau`` and ``T``
            in seconds.
        f_hz: Frequencies in Hz.

    Returns:
        Complex response.

    Raises:
        ValueError: if the model is missing a required parameter.
    """
    _check(model)
    s = 1j * hz_to_rads(np.asarray(f_hz, dtype=np.float64))
    p = model.params
    delay = np.exp(-p["tau"] * s)

    if model.structure == "fo_delay":
        return np.asarray(p["K"] / (p["T"] * s + 1.0) * delay, dtype=np.complex128)

    wn, zeta = p["wn"], p["zeta"]
    denominator = s * s + 2.0 * zeta * wn * s + wn * wn
    response = p["K"] * wn * wn / denominator
    if model.structure == "so_zero_delay":
        response = response * (s / p["wz"] + 1.0)
    return np.asarray(response * delay, dtype=np.complex128)


def airframe_tf(model: AirframeModel):  # type: ignore[no-untyped-def]
    """The model as a ``control`` transfer function, delay approximated by Pade.

    Prefer :func:`airframe_response` anywhere the delay's phase matters, which is
    everywhere that sets a gain. Pade error grows with ``tau * omega`` and would
    quietly flatter the design at high crossover.

    Returns:
        ``control.TransferFunction``.

    Raises:
        ValueError: if the model is missing a required parameter.
    """
    import control

    _check(model)
    p = model.params
    s = control.tf("s")

    if model.structure == "fo_delay":
        rational = p["K"] / (p["T"] * s + 1.0)
    else:
        wn, zeta = p["wn"], p["zeta"]
        rational = p["K"] * wn**2 / (s**2 + 2.0 * zeta * wn * s + wn**2)
        if model.structure == "so_zero_delay":
            rational = rational * (s / p["wz"] + 1.0)

    if p["tau"] <= 0.0:
        return rational
    num, den = control.pade(p["tau"], PADE_ORDER)
    return rational * control.tf(num, den)
