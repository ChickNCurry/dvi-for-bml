from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_jsd(
    p_log_probs: NDArray[np.float32],
    q_log_probs: NDArray[np.float32],
) -> Any:
    # (dim1, dim2, ...)

    p_probs, q_probs = np.exp(p_log_probs), np.exp(q_log_probs)
    m_probs = 0.5 * (p_probs + q_probs)

    m_log_probs = np.log(m_probs)

    kl_p_m = p_probs * (p_log_probs - m_log_probs)
    kl_q_m = q_probs * (q_log_probs - m_log_probs)

    jsd = 0.5 * np.sum(kl_p_m + kl_q_m)

    return jsd


def compute_bd(
    p_log_probs: NDArray[np.float32],
    q_log_probs: NDArray[np.float32],
) -> Any:
    # (dim1, dim2, ...)

    p_probs, q_probs = np.exp(p_log_probs), np.exp(q_log_probs)

    bc = np.sum(np.sqrt(p_probs * q_probs))
    bd = -np.log(np.clip(bc, a_min=1e-300, a_max=None))

    return bd
