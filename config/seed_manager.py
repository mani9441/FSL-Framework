"""
Random Seed Manager for FSL Research Framework.
Manages deterministic seed distribution for multi-trial experiments.
"""

from typing import List
from utilities.helpers import set_seed
from utilities.logger import get_logger

logger = get_logger("seed_manager")


class SeedManager:
    """Manages global and trial-specific random seeds for reproducibility."""

    def __init__(self, base_seed: int = 42, total_trials: int = 3):
        self.base_seed = base_seed
        self.total_trials = total_trials
        self.trial_seeds: List[int] = [base_seed + i for i in range(total_trials)]

    def get_trial_seed(self, trial_index: int) -> int:
        """Returns the deterministic random seed for a specific zero-indexed trial."""
        if not (0 <= trial_index < self.total_trials):
            # Fallback for out-of-bound indices
            return self.base_seed + trial_index
        return self.trial_seeds[trial_index]

    def set_trial_seed(self, trial_index: int) -> int:
        """
        Applies random seed for the given trial index across standard random, NumPy, and PyTorch.
        Returns the applied seed value.
        """
        seed = self.get_trial_seed(trial_index)
        set_seed(seed)
        logger.info(f"SeedManager applied seed {seed} for Trial {trial_index + 1}/{self.total_trials}")
        return seed

    def get_all_seeds(self) -> List[int]:
        """Returns list of seeds for all scheduled trials."""
        return self.trial_seeds.copy()
