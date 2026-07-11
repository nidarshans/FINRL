import numpy as np
from numpy.testing import assert_allclose

from finrl.backtest.robustness import (
    estimate_capacity,
    execution_delay_returns,
    release_gate,
    stress_transaction_costs,
    subperiod_metrics,
)


def test_robustness_stress_and_capacity_utilities() -> None:
    returns = np.array([0.01, -0.005, 0.02, 0.0])
    spy = np.zeros(4)
    turnover = np.ones(4) * 0.1
    costs = np.ones(4) * 0.001
    stress = stress_transaction_costs(returns, spy, turnover, costs, 4)
    assert stress[2.0].cumulative_return < stress[0.5].cumulative_return
    assert_allclose(execution_delay_returns(returns, 1), [0.0, 0.01, -0.005, 0.02])
    capacity = estimate_capacity(np.array([100.0, 200.0]), 0.1)
    assert capacity.estimated_capacity == 15.0
    assert len(subperiod_metrics(returns, spy, 4, 2)) == 2
    assert release_gate(stress[1.0], 1.0)
