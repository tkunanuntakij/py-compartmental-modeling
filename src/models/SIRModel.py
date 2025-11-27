from dataclasses import dataclass
from src.models import BaseModel


@dataclass(frozen=True)
class SIRModelState:
    """
    Represent the state of the SIR model
    
    Args:
        S (float): Susceptible population at a given time step.
        I (float): Infectious population at a given time step.
        R (float): Recovered or removed population at a given time step.
    """
    S: float = 0.0
    I: float = 0.0
    R: float = 0.0
    
    def __post_init__(self):
        if self.S < 0:
            raise ValueError(
                f"S (number of susceptibles) must be > 0, got {self.S}")
        if self.I < 0:
            raise ValueError(
                f"I (number of infectives) must be > 0, got {self.I}")
        if self.R < 0:
            raise ValueError(
                f"R (number of recovered cases) must be > 0, got {self.R}")

    @property
    def N(self):
        """Total number of population"""
        return self.S + self.I + self.R


@dataclass
class SIRModelStateChange:
    """Represent the change in each compartment
    
    Args:
        dS (float): Change in the number of susceptibles.
        dI (float): Change in the number of infectives.
        dR (float): Change in the number of recovered cases.
    """
    dS: float
    dI: float
    dR: float



@dataclass(frozen=True)
class SIRModelParam:
    """The parameters of the SIR model
    
    Args:
        beta (float): Infectious rate for susceptibles
        mu (float): Recovery rate for infectives
    """
    beta: float
    mu: float
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.mu < 0:
            raise ValueError(f"mu must be >= 0, got {self.mu}")


class SIRModel(BaseModel):
    """The classic Compartmental models SIR model in Epidemiology

    Args:
        BaseModel (SIRModelState): Model state
    """
    def __init__(self, state: SIRModelState):
        self.time_step = 0.0
        self.state = state


    @staticmethod
    def calculate_change(state: SIRModelState, params: SIRModelParam):
        """Calcuate the change according to the SIR model

        Args:
            state (SIRModelState): The current state of the SIR model
            params (SIRModelParam): The params of the SIR model

        Returns:
            SIRModelStateChange: The change in each compartment in the next
            time step
        """
        S, I, R = state.S, state.I, state.R
        mu, beta = params.mu, params.beta
        N = state.N
        dSt_dt = -beta * S * I / N
        dIt_dt = beta * S * I / N - mu * I
        dRt_dt = mu * I
        return SIRModelStateChange(dSt_dt, dIt_dt, dRt_dt)

    def update_change(self, change: SIRModelStateChange):
        "Apply the change to the current state"
        self.state = SIRModelState(
            self.state.S + change.dS,
            self.state.I + change.dI,
            self.state.R + change.dR,
        )
