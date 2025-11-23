import numpy as np


def compute_desalineacion_probability(residues: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    """
    Calcula la probabilidad de desalineación para cada punto en 'residues' usando los umbrales dados.

    Para cada valor en 'residues':
      - Si el residuo <= lower_threshold, la probabilidad es 0 (totalmente desbalanceo).
      - Si el residuo >= upper_threshold, la probabilidad es 1 (totalmente desalineación).
      - En caso intermedio, se interpola linealmente:
            p = (residuo - lower_threshold) / (upper_threshold - lower_threshold)

    Args:
        residues (np.ndarray): Vector 1D de residuos (ya escalados).
        lower_threshold (float): Umbral inferior.
        upper_threshold (float): Umbral superior.

    Returns:
        np.ndarray: Vector 1D de probabilidades de desalineación.
    """
    p = np.zeros_like(residues, dtype=float)
    p[residues <= lower_threshold] = 0.0
    p[residues >= upper_threshold] = 1.0
    mask = (residues > lower_threshold) & (residues < upper_threshold)
    p[mask] = (residues[mask] - lower_threshold) / (upper_threshold - lower_threshold)
    return p