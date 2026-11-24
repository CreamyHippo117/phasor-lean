"""The frame stopwatch (``core/timing.py``)."""

from __future__ import annotations

import time

import pytest

from phasor.core.timing import Stopwatch, stage


def _busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_nothing_is_recorded_and_sync_never_runs_without_a_stopwatch():
    called = []
    with stage("draw", sync=lambda: called.append(1)):
        _busy(0.001)
    assert called == []


def test_nested_stages_are_counted_once_under_the_inner_name():
    with Stopwatch() as sw, stage("draw"):
        _busy(0.02)
        with stage("simulation"):
            _busy(0.03)
    assert sw.seconds["simulation"] == pytest.approx(0.03, abs=0.01)
    assert sw.seconds["draw"] == pytest.approx(0.02, abs=0.01)  # not 0.05
    assert sum(sw.seconds.values()) <= sw.wall
    assert sw.other() >= 0


def test_repeated_stages_add_up_and_count_their_calls():
    with Stopwatch() as sw:
        for _ in range(3):
            with stage("encode"):
                _busy(0.005)
    assert sw.calls["encode"] == 3
    assert sw.seconds["encode"] == pytest.approx(0.015, abs=0.008)


def test_sync_runs_inside_its_stage_and_its_wait_is_counted_there():
    with Stopwatch() as sw, stage("post", sync=lambda: _busy(0.02)):
        pass
    assert sw.seconds["post"] == pytest.approx(0.02, abs=0.01)


def test_a_failing_stage_still_closes_and_the_stopwatch_ends():
    with Stopwatch() as sw, pytest.raises(ValueError), stage("draw"):
        raise ValueError("boom")
    assert sw.calls["draw"] == 1
    with stage("draw"):  # the stopwatch has stopped: no error, nothing recorded
        pass
    assert sw.calls["draw"] == 1


def test_only_one_stopwatch_at_a_time():
    with Stopwatch(), pytest.raises(RuntimeError), Stopwatch():
        pass
