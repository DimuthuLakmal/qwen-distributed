from abc import ABC, abstractmethod
from typing import Any

from src.data.schema import Conversation


class DatasetAdapter(ABC):

    @abstractmethod
    def convert(self, sample: dict[str, Any]) -> Conversation:
        pass