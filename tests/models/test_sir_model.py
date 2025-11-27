import pytest
from src.models import SIRModel, SIRModelState, SIRModelParam


def test_initialization():
    state = SIRModelState(S=100)
    
    model = SIRModel(state=state)
    assert model.state.S == 100.0
    assert model.state.I == 0.0
    assert model.state.R == 0.0
    assert model.time_step == 0.0


def test_calculate_change():
    num_susceptibles = 90
    num_infectives = 10
    state = SIRModelState(S=num_susceptibles, I=num_infectives)
    params = SIRModelParam(beta=0.5, mu=0)
    new_state = SIRModel.calculate_change(state, params)
    expect_new_infectives = 4.5
    assert new_state.S == pytest.approx(num_susceptibles - expect_new_infectives)
    assert new_state.I == pytest.approx(num_infectives + expect_new_infectives)
    assert new_state.R == pytest.approx(0.0)
    assert new_state.N == pytest.approx(100.0)

def test_sir_model_state_negative_values():
    with pytest.raises(ValueError):
        SIRModelState(S=-1, I=0, R=0)

