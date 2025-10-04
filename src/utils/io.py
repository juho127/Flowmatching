from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv_2d(array: np.ndarray, path: str, header: Sequence[str] | None = None) -> None:
    if header is not None:
        header_line = ",".join(header)
        np.savetxt(path, array, delimiter=",", fmt="%.8f", header=header_line, comments="")
    else:
        np.savetxt(path, array, delimiter=",", fmt="%.8f")


