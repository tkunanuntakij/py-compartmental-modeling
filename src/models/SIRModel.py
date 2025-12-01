from __future__ import annotations
from dataclasses import dataclass
from types import NotImplementedType
from typing import Self

from src.models.BaseModel import BaseModel


@dataclass(frozen=True)
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

    def __add__(self, other: object) -> SIRModelStateChange | NotImplementedType:
        """This function allows adding two change objects.
        Note: To avoid conflicts with addition involving `SIRModelState`, we
        first check whether the operand is of the same type. If not, we defer
        to the other class to handle the addition.

        Ref:
        https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

        Args:
            other (Self): _description_

        Returns:
            Self | NotImplemented: _description_
        """
        if isinstance(other, SIRModelStateChange):
            # State + Change -> new State
            return SIRModelStateChange(
                self.dS + other.dS,
                self.dI + other.dI,
                self.dR + other.dR,
            )
        return NotImplemented

    def __radd__(self, change: Self) -> SIRModelStateChange | NotImplementedType:
        if isinstance(change, SIRModelStateChange):
            return self.__add__(change)
        return NotImplemented


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
            raise ValueError(f"S (number of susceptibles) must be > 0, got {self.S}")
        if self.I < 0:
            raise ValueError(f"I (number of infectives) must be > 0, got {self.I}")
        if self.R < 0:
            raise ValueError(f"R (number of recovered cases) must be > 0, got {self.R}")

    @property
    def N(self):
        """Total number of population"""
        return self.S + self.I + self.R

    def __add__(self, change: SIRModelStateChange):
        if not isinstance(change, SIRModelStateChange):
            raise TypeError(
                "Only supports addition between SIRModelState "
                "and SIRModelStateChange."
            )
        new_state = SIRModelState(
            self.S + change.dS, self.I + change.dI, self.R + change.dR
        )
        return new_state

    def __radd__(self, change: SIRModelStateChange):
        return self.__add__(change)


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


class SIRModel(BaseModel[SIRModelState, SIRModelStateChange]):
    """The classic Compartmental models SIR model in Epidemiology

    Args:
        state (SIRModelState): Model state
        params (SIRModelParam): Model parameters
    """

    def __init__(self, state: SIRModelState, params: SIRModelParam):
        self.time_step = 0.0
        self.state = state
        self.params = params
        self.history: list[SIRModelState] = []

    def calculate_change(self) -> SIRModelStateChange:
        """Calcuate the change according to the SIR model

        Args:
            state (SIRModelState): The current state of the SIR model
            params (SIRModelParam): The params of the SIR model

        Returns:
            SIRModelStateChange: The change in each compartment in the next
            time step
        """
        S, I, R = self.state.S, self.state.I, self.state.R
        mu, beta = self.params.mu, self.params.beta
        N = self.state.N
        dSt_dt = -beta * S * I / N
        dIt_dt = beta * S * I / N - mu * I
        dRt_dt = mu * I
        return SIRModelStateChange(dSt_dt, dIt_dt, dRt_dt)

    def step(
        self
        ) -> None:
        change = self.calculate_change()
        self.state = self.state + change
        self.time_step += 1

    def loop(
        self,
        num_time_step: int
    ):
        for _ in range(num_time_step):
            self.step()
            self.history.append(self.state)
