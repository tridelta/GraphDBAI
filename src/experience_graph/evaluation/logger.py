from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_graph.core.serialization import to_jsonable


class EvaluationLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, name: str, row: Any) -> None:
        path = self.run_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
