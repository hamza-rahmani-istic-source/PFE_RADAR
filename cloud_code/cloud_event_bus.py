#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class LocalEventBus:
    """
    Simple append-only JSONL event bus.
    Useful as a production-friendly interface boundary:
    - cloud pipeline publishes processing events
    - RAG/chatbot reads these events independently
    """

    def __init__(self, events_file: str = "cloud_events/events.jsonl") -> None:
        self.events_file = Path(events_file).resolve()
        self.events_file.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, event_type: str, payload: Dict[str, Any], trace_id: str) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_type": event_type,
            "trace_id": trace_id,
            "payload": payload,
        }
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.events_file.exists():
            return []
        lines = self.events_file.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines[-max(1, limit) :]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

