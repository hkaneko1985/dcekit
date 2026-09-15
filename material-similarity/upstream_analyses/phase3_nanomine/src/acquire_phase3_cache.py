#!/usr/bin/env python3
"""Acquire public MaterialsMine search/cache exports used in Phase 3.

The public SPARQL service may be unavailable during the 2026 knowledge-graph
migration.  The website's public Elasticsearch-backed search routes expose the
same cached sample-view responses and are therefore used as a reproducible
fallback.  No property/target query is requested.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


BASE = "https://materialsmine.org/api/search/filter"


def fetch(params: dict[str, str], path: Path) -> None:
    url = BASE + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "phase3-material-similarity/1.0"})
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = response.read()
    parsed = json.loads(payload)
    path.write_text(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw_cache"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    common = {"field": "label", "type": "knowledge", "page": "1", "pageSize": "5000"}
    fetch(
        {"search": "http", "field": "identifier", "type": "samples", "page": "1", "pageSize": "2000"},
        args.output / "sample_index.json",
    )
    for filename, term in {
        "components_knowledge.json": "std_name",
        "processing_knowledge.json": "param_label",
        "headers_knowledge.json": "process_label",
    }.items():
        fetch({"search": term, **common}, args.output / filename)


if __name__ == "__main__":
    main()

