from typing import Protocol


class BaseModel(Protocol):
    def calculate_change(self): ...
    
    def update_change(self): ...
