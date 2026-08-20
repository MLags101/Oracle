"""Session save and load (milestone M8).

The point of a session file is to be able to close the tool and come back to the
same argument tomorrow, so the tests are about fidelity: what comes back has to
be able to produce the same report, the same export and the same numbers as what
went in. A format that loses one field quietly is worse than no format at all,
because the loss shows up as a different recommendation rather than as an error.
"""

from __future__ import annotations

import dataclasses
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rotorid.config import load_config
from rotorid.core.export.session import (
    FORMAT_VERSION,
    load_session,
    save_session,
)
from rotorid.core.pipeline import analyze
from tests.synthetic.generators import make_airframe, make_bundle, make_chain

CONFIG = load_config()


@pytest.fixture(scope="module")
def result():
    bundle = make_bundle(make_airframe(), make_chain())
    return analyze(bundle, ("roll",), CONFIG, tool_version="test-1.0")


@pytest.fixture
def saved(tmp_path: Path, result):
    return save_session(tmp_path / "flight.rotorid", result.session)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def _same(a: object, b: object, path: str = "") -> None:
    """Structural equality that knows about arrays, which do not compare as bools."""
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), path
        assert a.shape == b.shape, path
        assert np.array_equal(a, b, equal_nan=True), path
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), path
        for f in dataclasses.fields(a):
            _same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            _same(a[key], b[key], f"{path}[{key!r}]")
        return
    if isinstance(a, list | tuple):
        assert type(a) is type(b), f"{path}: {type(a).__name__} became {type(b).__name__}"
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
        return
    assert a == b, path


def test_a_session_survives_the_round_trip_field_for_field(saved: Path, result) -> None:
    """The strong one: every field of every nested object, arrays included."""
    reloaded, _ = load_session(saved)
    _same(result.session, reloaded, "session")


def test_arrays_come_back_bit_exact(saved: Path, result) -> None:
    """A float that changes in its last bit changes a margin in its last digit."""
    reloaded, _ = load_session(saved)
    axis = next(iter(result.session.effective))
    before = result.session.effective[axis].frf
    after = reloaded.effective[axis].frf
    assert np.array_equal(before.f_hz, after.f_hz)
    assert np.array_equal(before.coherence, after.coherence)
    assert after.valid_mask.dtype == before.valid_mask.dtype


def test_tuples_do_not_come_back_as_lists(saved: Path, result) -> None:
    """Json flattens both to arrays; the difference matters to frozen dataclasses."""
    reloaded, _ = load_session(saved)
    assert isinstance(reloaded.segments, tuple)
    assert isinstance(reloaded.findings, tuple)
    rec = next(iter(reloaded.recommendations.values()))
    assert isinstance(rec.filters.chain.notches, tuple)
    assert isinstance(rec.model.valid_band_hz, tuple)


def test_a_reloaded_session_still_exports_the_same_parameters(
    saved: Path, result, tmp_path: Path
) -> None:
    """The end that matters: same file on disk, whether or not it was reopened."""
    from rotorid.core.export.params import write_param_files

    reloaded, _ = load_session(saved)
    assert reloaded.next_steps is not None
    assert result.session.next_steps is not None

    def write(where: Path, plan) -> dict[str, str]:
        paths = write_param_files(
            where,
            plan,
            log_name="flight.bin",
            tool_version="test-1.0",
            config_hash=CONFIG.hash,
        )
        return {
            p.name: "\n".join(
                line for line in p.read_text(encoding="utf-8").splitlines() if line[:1] != "#"
            )
            for p in paths
        }

    assert write(tmp_path / "a", result.session.next_steps) == write(
        tmp_path / "b", reloaded.next_steps
    )


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_a_matching_build_reports_no_mismatch(saved: Path) -> None:
    _, mismatch = load_session(saved, tool_version="test-1.0", config_hash=CONFIG.hash)
    assert not mismatch


def test_a_different_build_is_reported_rather_than_ignored(saved: Path) -> None:
    """Old numbers are evidence about the old build, not about this one."""
    _, mismatch = load_session(saved, tool_version="test-2.0", config_hash="deadbeef")
    assert mismatch
    assert mismatch.tool_version == ("test-1.0", "test-2.0")
    assert mismatch.config_hash == (CONFIG.hash, "deadbeef")
    described = mismatch.describe()
    assert "test-2.0" in described
    assert "Re-run" in described


def test_loading_without_a_build_to_compare_against_is_allowed(saved: Path) -> None:
    """Opening a session to look at it should not require anything to check."""
    session, mismatch = load_session(saved)
    assert not mismatch
    assert session.tool_version == "test-1.0"


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_file_from_another_format_version_is_refused_not_guessed_at(
    saved: Path, tmp_path: Path
) -> None:
    import json

    with zipfile.ZipFile(saved) as zf:
        tree = json.loads(zf.read("session.json"))
        arrays = zf.read("arrays.npz")
    tree["format_version"] = FORMAT_VERSION + 1

    other = tmp_path / "future.rotorid"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("session.json", json.dumps(tree))
        zf.writestr("arrays.npz", arrays)

    with pytest.raises(ValueError, match="session format"):
        load_session(other)


def test_a_zip_that_is_not_a_session_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "notes.zip"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("readme.txt", "not a session")
    with pytest.raises(ValueError, match="not a RotorID session"):
        load_session(other)


def test_a_session_cannot_name_an_arbitrary_class(saved: Path, tmp_path: Path) -> None:
    """A session file is data. Data does not get to choose what it becomes."""
    import json

    with zipfile.ZipFile(saved) as zf:
        tree = json.loads(zf.read("session.json"))
        arrays = zf.read("arrays.npz")
    tree["session"]["$type"] = "subprocess.Popen"

    other = tmp_path / "hostile.rotorid"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("session.json", json.dumps(tree))
        zf.writestr("arrays.npz", arrays)

    with pytest.raises(ValueError, match="unknown type"):
        load_session(other)


def test_an_interrupted_save_does_not_replace_a_good_file(tmp_path: Path, result) -> None:
    path = save_session(tmp_path / "s.rotorid", result.session)
    assert not list(tmp_path.glob("*.partial")), "the staging file must be renamed away"
    assert path.exists()
