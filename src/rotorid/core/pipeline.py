"""The analysis, start to finish, in one place (spec section 5).

The CLI and the GUI must not each carry their own copy of the running order.
They diverge -- one gains a step, one keeps an old threshold -- and then the
report says something the screen does not, which is the one failure this whole
tool cannot survive. So the order lives here, once, and both front ends call it.

The result is a :class:`Session`: everything that was measured, everything that
was decided, and the provenance to say which build decided it. That is also
exactly what gets saved to a ``.rotorid`` bundle, which is not a coincidence --
a session that could not be saved and reopened would be a session the user has
to finish in one sitting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rotorid.config import Config
from rotorid.core.design.recommend import AxisAnalysis, identify_axis, recommend_from
from rotorid.core.guidance.findings import GuidanceContext, collect_findings
from rotorid.core.guidance.plan import build_plan
from rotorid.core.types import (
    Axis,
    LogBundle,
    Session,
    TuneRecommendation,
)

__all__ = ["AnalysisCancelled", "AnalysisResult", "Progress", "analyze"]

#: Progress callback: a fraction in [0, 1] and what is happening. A plain
#: callable, not a Qt signal -- core must not import Qt (spec section 9).
Progress = Callable[[float, str], None]


class AnalysisCancelled(Exception):  # noqa: N818 - a cancellation, not an error
    """Raised when the caller asked to stop. Nothing partial is returned."""


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """A session, plus what could not be analysed and why.

    Failures are carried alongside rather than raised: one unusable axis is a
    normal outcome of a log flown with a sweep on roll only, and it should not
    cost the user the two axes that did work.
    """

    session: Session
    analyses: dict[Axis, AxisAnalysis]
    failures: dict[Axis, str] = field(default_factory=dict)

    @property
    def blockers(self) -> tuple[str, ...]:
        """Findings that stop an export until they are acknowledged by name."""
        return tuple(
            f.code
            for f in self.session.findings
            if f.severity == "blocker" and f.code not in self.session.acknowledgements
        )

    def unresolved(self, acknowledgements: dict[str, str] | None = None) -> tuple[str, ...]:
        """Blocking codes still outstanding given a set of acknowledgements."""
        accepted = {**self.session.acknowledgements, **(acknowledgements or {})}
        return tuple(
            f.code
            for f in self.session.findings
            if f.severity == "blocker" and f.code not in accepted
        )


def analyze(
    bundle: LogBundle,
    axes: tuple[Axis, ...],
    config: Config,
    *,
    tool_version: str,
    conservatism: float = 0.5,
    acknowledgements: dict[str, str] | None = None,
    progress: Progress | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> AnalysisResult:
    """Identify, design, check and plan, for each requested axis.

    The order matters and is the reason this function exists:

    1. Identify every axis first, so that checks which compare axes against each
       other have all of them.
    2. Design against each identification.
    3. Only then collect findings, because a finding about a recommendation
       cannot be made before the recommendation exists.
    4. Build the plan last, so it can attribute each flight to the findings that
       motivated it.

    Args:
        progress: Called with a fraction and a description as the run advances.
        should_cancel: Polled between steps. When it returns True the run raises
            :class:`AnalysisCancelled` rather than returning a partial session,
            because a session that is missing an axis for a reason the user has
            forgotten is worse than no session.

    Raises:
        AnalysisCancelled: if ``should_cancel`` asked it to stop.
    """
    analyses: dict[Axis, AxisAnalysis] = {}
    recommendations: dict[Axis, TuneRecommendation] = {}
    failures: dict[Axis, str] = {}

    steps = 2 * len(axes) + 1
    done = 0

    def step(what: str) -> None:
        nonlocal done
        if should_cancel is not None and should_cancel():
            raise AnalysisCancelled(what)
        if progress is not None:
            progress(done / steps, what)
        done += 1

    for axis in axes:
        step(f"identifying {axis}")
        try:
            analyses[axis] = identify_axis(bundle, axis, config)
        except ValueError as exc:
            failures[axis] = str(exc)

    for axis, analysis in analyses.items():
        step(f"designing {axis}")
        try:
            recommendations[axis] = recommend_from(
                analysis, bundle, config, conservatism=conservatism
            )
        except ValueError as exc:
            failures[axis] = str(exc)

    step("checking and planning")

    # Run unconditionally, even when no axis produced a recommendation. That is
    # the case where the user most needs the log-level checks: an aircraft whose
    # rate messages were logged at 10 Hz fails on all three axes at once, and
    # three copies of "no usable excitation" do not name the thing to change.
    # The per-axis checks key off ``analyses``, so they simply find nothing.
    findings = collect_findings(
        GuidanceContext(
            bundle=bundle,
            analyses={a: analyses[a] for a in recommendations},
            recommendations=recommendations,
            config=config,
        )
    )
    plan = build_plan(recommendations, findings, bundle.params) if recommendations else None

    session = Session(
        log=bundle,
        segments=tuple(seg for a in analyses.values() for seg in a.segments),
        effective={axis: a.effective for axis, a in analyses.items()},
        models={axis: a.airframe for axis, a in analyses.items()},
        noise={axis: a.noise for axis, a in analyses.items() if a.noise is not None},
        measured_steps={axis: a.measured for axis, a in analyses.items() if a.measured is not None},
        recommendations=recommendations,
        findings=findings,
        config_hash=config.hash,
        tool_version=tool_version,
        created_utc=datetime.now(UTC),
        next_steps=plan,
        acknowledgements=dict(acknowledgements or {}),
    )
    if progress is not None:
        progress(1.0, "done")
    return AnalysisResult(session=session, analyses=analyses, failures=failures)
