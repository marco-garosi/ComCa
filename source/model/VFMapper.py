from typing import Any

class VFMapper:
    """Base class for a vocabulary-free mapper, i.e. a function that maps predicted attributes
        into the ground truth based on a certain policy.
    """

    def __init__(self, ground_truth: list[str]) -> None:
        # Constructing as a dictionary (hashmap) for fast access when mapping
        # This won't require a linear search as a pain list would
        self.ground_truth: dict[str, int] = {
            gt: idx
            for idx, gt in enumerate(ground_truth)
        }

    def __call__(self, prediction: list[str], **kwargs) -> list[Any]:
        """Maps prediction to the ground truth

        Args:
            prediction (list[str]): list of predicted attributes to be mapped to the ground truth
                (already stored in the instance of the class)

        Returns:
            list[Any]: mapping from prediction (index i) to the ground truth
                (either as single element or list of elements)
        """

        pass
