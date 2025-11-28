from typing import Protocol, TypeVar, runtime_checkable


StateT = TypeVar("StateT", covariant=True)
ChangeT = TypeVar("ChangeT", covariant=True)


@runtime_checkable
class BaseModel(Protocol[StateT, ChangeT]):
    @staticmethod
    def calculate_change(*args, **kwargs) -> ChangeT:
        """Compute the state change for the current step."""
        ...

    def update_state(self, *args, **kwargs) -> StateT:
        """Apply the change to the state."""
        ...
