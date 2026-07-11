import numpy as np

from finrl.backtest.release import validate_release


def test_validate_release_returns_stress_and_gate() -> None:
    returns = np.array([0.01, 0.01, 0.01, 0.01])
    result = validate_release(
        returns,
        np.zeros(4),
        np.ones(4) * 0.1,
        np.ones(4) * 0.0001,
        np.ones(4) * 1_000_000.0,
        4,
        max_drawdown=0.1,
        min_information_ratio=-10.0,
    )
    assert set(result.cost_stress) == {0.5, 1.0, 2.0, 3.0}
    assert result.capacity.estimated_capacity == 100_000.0
    assert result.passed
