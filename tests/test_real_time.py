"""The real-time vector of a run (real_time.csv), and how the app reads it. No FMUs are run.

License: MIT (see LICENSE in the repository root)
"""
from types import SimpleNamespace
from time import perf_counter

import pandas as pd
import pytest

from app.build.data import REAL_TIME_FILE, load_results, real_time_factor, summary, unit
from ospfmusim.run import run_steps


class Execution:
    """The part of libcosim's execution run_steps() uses. Takes steps of 0.1 s."""

    def __init__(self, fail_at_step=None):
        self.time, self.steps, self.fail_at_step = 0, 0, fail_at_step

    def status(self):
        return SimpleNamespace(current_time=self.time)

    def step(self):
        self.steps += 1
        self.time += 100_000_000
        return self.steps != self.fail_at_step


def test_one_row_per_step_on_the_same_time_axis_as_the_fmus():
    vector = run_steps(Execution(), 300_000_000, perf_counter())
    assert list(vector.index) == [0.0, 0.1, 0.2, 0.3]         # as the FMUs' CSV files
    assert vector["real_time_s"].is_monotonic_increasing
    assert vector["step_time_s"].iloc[0] == 0.0               # nothing has been stepped yet
    assert vector["step_time_s"].sum() == pytest.approx(
        vector["real_time_s"].iloc[-1] - vector["real_time_s"].iloc[0])


def test_a_failed_step_stops_the_run():
    with pytest.raises(RuntimeError, match="failed"):
        run_steps(Execution(fail_at_step=2), 300_000_000, perf_counter())


def test_the_app_reads_the_vector(tmp_path):
    vector = pd.DataFrame({"real_time_s": [4.0, 5.0, 6.0], "step_time_s": [0.0, 1.0, 1.0]},
                          index=pd.Index([0.0, 5.0, 10.0], name="Time"))
    vector.to_csv(tmp_path / REAL_TIME_FILE)

    results = load_results(tmp_path)
    pd.testing.assert_frame_equal(results["real_time"], vector)
    # 10 s simulated in the 2 s of stepping. The 4 s of loading only count in the real time.
    assert real_time_factor(results["real_time"]) == 5.0
    numbers = summary(results, {})
    assert numbers["Real time"] == (6.0, "s")
    assert numbers["Real-time factor"] == (5.0, "")
    assert unit("real_time.real_time_s") == unit("real_time.step_time_s") == "s"


def test_a_run_without_the_vector_still_loads(tmp_path):
    assert load_results(tmp_path) == {}
    assert summary({}, {}) == {}
