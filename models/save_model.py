from __future__ import annotations

from pathlib import Path

from joblib import dump


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
