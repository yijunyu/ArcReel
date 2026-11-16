#!/usr/bin/env python3
"""Prove ArcReel's minimax-h3 declarative endpoint works against h3 serve.

Uses arcreel_market_core's template engine and response extraction —
the same code ArcReel's DeclarativeVideoBackend uses in production —
to submit, poll, and verify a job through h3 serve.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from arcreel_market_core.endpoint_definition import (
    build_context,
    extract_value,
    render_request,
)

H3_SERVE = "http://192.168.1.123:8321"
DEFINITION = json.loads(
    (Path(__file__).parent / "lib/custom_provider/builtin_endpoints/minimax-h3.json").read_text()
)


def http_request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> None:
    print("=== ArcReel declarative endpoint integration test ===")
    print(f"Endpoint definition: minimax-h3.json")
    print(f"Target: {H3_SERVE}")

    params = {
        "base_url": H3_SERVE,
        "api_key": "",
        "model": "MiniMax-H3",
        "prompt": "A golden retriever puppy plays in autumn leaves, warm sunlight, cinematic",
        "resolution": "768p",
        "duration": 5,
        "aspect_ratio": "16:9",
    }
    inputs = {
        "first_frame": None,
        "last_frame": None,
        "refs": [],
        "ref_audio": [],
    }

    # 1. Build template context (what ArcReel's DeclarativeVideoBackend does)
    ctx = build_context(params, inputs=inputs, defaults=DEFINITION.get("defaults"))
    print(f"\n1. Template context built: {len(ctx)} keys")

    # 2. Render the submit request using ArcReel's template engine
    rendered = render_request(
        DEFINITION["submit"], ctx,
        enum_maps=DEFINITION.get("enum_maps"),
        auth=DEFINITION.get("auth"),
    )
    print(f"2. Submit request rendered:")
    print(f"   Method: {rendered.method}")
    print(f"   URL: {rendered.url}")
    submit_body = json.loads(rendered.body) if isinstance(rendered.body, str) else rendered.body
    print(f"   Body model: {submit_body.get('model')}")
    prompt_text = submit_body.get("content", [{}])[0].get("text", "")[:60]
    print(f"   Body prompt: {prompt_text}...")

    # 3. Actually submit to h3 serve
    print(f"\n3. Submitting to h3 serve...")
    resp = http_request(rendered.method, rendered.url, submit_body)
    print(f"   Raw response: {json.dumps(resp)}")

    # 4. Extract task_id using ArcReel's extraction logic
    extract_spec = DEFINITION["submit"]["extract"]
    task_id = extract_value(extract_spec["task_id"], resp)
    print(f"4. Extracted task_id via ArcReel jsonpath: {task_id}")
    assert task_id, "task_id extraction failed"

    # 5. Build poll context and render poll request
    poll_ctx = {**ctx, "task_id": task_id}
    poll_rendered = render_request(
        DEFINITION["poll"], poll_ctx,
        enum_maps=DEFINITION.get("enum_maps"),
        auth=DEFINITION.get("auth"),
    )
    print(f"\n5. Poll request rendered:")
    print(f"   URL: {poll_rendered.url}")

    # 6. Poll using ArcReel's extraction
    status_map = DEFINITION["status_map"]
    poll_extract = DEFINITION["poll"]["extract"]
    print(f"\n6. Polling (ArcReel status_map: {status_map})...")

    start = time.time()
    while True:
        time.sleep(15)
        elapsed = time.time() - start
        raw = http_request(poll_rendered.method, poll_rendered.url)

        raw_status = extract_value(poll_extract["status"], raw)
        mapped = status_map.get(str(raw_status), "unknown")
        video_url = extract_value(poll_extract["video_url"], raw)
        error = extract_value(poll_extract.get("error", []), raw)

        print(f"   [{elapsed:.0f}s] raw={raw_status} mapped={mapped} url={video_url or '-'}")

        if mapped == "succeeded":
            print(f"\n=== SUCCESS ===")
            print(f"ArcReel's minimax-h3.json endpoint definition correctly:")
            print(f"  - Rendered submit request → h3 serve accepted it")
            print(f"  - Extracted task_id from submit response")
            print(f"  - Rendered poll request with task_id")
            print(f"  - Extracted status and mapped it via status_map")
            print(f"  - Extracted video_url from succeeded response")
            print(f"  Video URL: {video_url}")
            print(f"  Total time: {elapsed:.0f}s")
            return
        elif mapped == "failed":
            print(f"\n=== FAILED: {error} ===")
            sys.exit(1)
        elif elapsed > 3600:
            print("\n=== TIMEOUT ===")
            sys.exit(1)


if __name__ == "__main__":
    main()
