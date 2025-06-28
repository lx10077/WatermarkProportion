## Copied from https://github.com/XuandongZhao/llm-watermark-location
from typing import List
import numpy as np
import torch
import cpp_src.aligator as aligator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OpenaiAligator:
    def __init__(self, threshold: float):
        """
        Initialize the Aligator detector with a threshold.
        
        Args:
            threshold (float): Detection threshold.
        """
        self.prev_pred = 0
        self.threshold = threshold

    def detect_fraction(self, y: List[float]) -> (np.ndarray, np.ndarray):
        """
        Perform bidirectional circular detection on a list of token scores.
        
        Args:
            y (List[float]): List of token scores (typically the Y values).
        
        Returns:
            detect_mask (np.ndarray): Boolean array indicating watermark detection per position.
            averaged_scores (np.ndarray): Averaged Aligator scores across all rotations.
        """
        n = len(y)
        step = max(1, int(n / 30))
        rotated_scores = []

        for i in range(0, n, step):
            # Forward and reverse runs of Aligator
            alig_forward = aligator.run_aligator(n, y, np.arange(0, n), 0, 1, 1e-5)
            alig_reverse = aligator.run_aligator(n, y, np.flip(np.arange(0, n)), 0, 1, 1e-5)
            averaged = np.nanmean(np.array([alig_forward, alig_reverse]), axis=0)

            # Rotate scores circularly
            rotated = np.concatenate((averaged[n - i:], averaged[:n - i]))
            rotated_scores.append(rotated)

            # Shift the scores for the next rotation
            y = np.concatenate((y[step:], y[:step]))

        averaged_scores = np.nanmean(np.array(rotated_scores), axis=0)
        detect_mask = averaged_scores > self.threshold

        return detect_mask, averaged_scores

    def compute_fraction(self, score_Ys: np.ndarray) -> np.ndarray:
        """
        Compute the estimated watermark fraction for each sequence in a batch.

        Args:
            score_Ys (np.ndarray): Array of shape (N, m), where N is the number of sequences.

        Returns:
            np.ndarray: Estimated watermark fraction per sequence.
        """
        estimated_fractions = []
        for single_score_sequence in score_Ys:
            detect_mask, _ = self.detect_fraction(single_score_sequence)
            estimated_fractions.append(detect_mask)

        detect_array = np.array(estimated_fractions)
        return np.mean(detect_array, axis=1)

    def compute_fraction_whole(self, score_Ys: np.ndarray) -> float:
        """
        Compute the estimated watermark fraction over the entire batch as one sequence.

        Args:
            score_Ys (np.ndarray): Array of shape (N, m)

        Returns:
            float: Estimated overall watermark fraction.
        """
        flattened_scores = score_Ys.reshape(-1)
        detect_mask, _ = self.detect_fraction(flattened_scores)
        return np.mean(detect_mask)
