from abc import ABC, abstractmethod


class ProcessingStage(ABC):
    @abstractmethod
    def validate(self) -> None:
        """Raise ValueError if inputs are invalid."""

    @abstractmethod
    def run(self):
        pass

    def run_batch(self, items: list, batch_size: int = 1000):
        """Process a list of items in batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        results = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            results.extend(self._run_batch(batch))

        return results

    def _run_batch(self, batch: list):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement batch processing."
        )
