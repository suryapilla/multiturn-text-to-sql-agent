# /// script
# requires-python = ">=3.10"
# dependencies = ["data-designer"]
# ///
"""Repro: ValidationColumnConfig misaligns rows after an upstream row is dropped.

Pipeline: seed rows (item_id 0..N-1) -> LLM text column -> ValidationColumnConfig (local callable).
A local mock OpenAI-compatible endpoint returns HTTP 400 (non-retryable) for the LLM request of
one chosen row, so that row is dropped before the validation column runs. No credentials or
network access needed.

    uv run --script repro_validation_index.py                # row 1 fails -> whole batch is lost (bug)
    uv run --script repro_validation_index.py --fail-row 3   # last row fails -> works (index stays contiguous)
    uv run --script repro_validation_index.py --fail-row -1  # control: no failure -> works

Pin a version with e.g. `uv run --with data-designer==0.9.6 --script repro_validation_index.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner
from data_designer.interface.errors import DataDesignerGenerationError


def start_mock(fail_row: int) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            prompt = json.dumps(req.get("messages", []))
            item = int(m.group(1)) if (m := re.search(r"item (\d+)", prompt)) else None
            if item == fail_row:
                status, body = 400, {"error": {"message": "mock bad request", "type": "invalid_request_error"}}
            else:
                status, body = 200, {
                    "id": f"mock-{item}", "object": "chat.completion", "created": 0, "model": "mock",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": f"answer for item {item}"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def check_answer(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"is_valid": df["answer"].str.startswith("answer")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-row", type=int, default=1, help="item_id whose LLM call gets a 400 (-1: none)")
    ap.add_argument("--num-records", type=int, default=4)
    args = ap.parse_args()
    n = args.num_records
    expected = n - (1 if 0 <= args.fail_row < n else 0)

    print(f"data-designer=={version('data-designer')}  pandas=={pd.__version__}  fail_row={args.fail_row}")
    srv, url = start_mock(args.fail_row)

    b = dd.DataDesignerConfigBuilder(model_configs=[
        dd.ModelConfig(alias="mock", model="mock", provider="mock", skip_health_check=True)])
    b.with_seed_dataset(dd.DataFrameSeedSource(df=pd.DataFrame({"item_id": range(n)})))
    b.add_column(dd.LLMTextColumnConfig(name="answer", model_alias="mock", prompt="Describe item {{ item_id }}."))
    b.add_column(dd.ValidationColumnConfig(
        name="answer_check", target_columns=["answer"], validator_type=dd.ValidatorType.LOCAL_CALLABLE,
        validator_params=dd.LocalCallableValidatorParams(validation_function=check_answer)))

    designer = DataDesigner(
        artifact_path=tempfile.mkdtemp(),
        model_providers=[dd.ModelProvider(name="mock", endpoint=url, provider_type="openai", api_key="x")])
    try:
        df = designer.preview(b, num_records=n).dataset
    except DataDesignerGenerationError as e:
        # Raised only when 0 records survive: the validation failure discarded the whole batch.
        print(f"\nNo records produced ({type(e).__name__}).")
        print(f"rows: 0 (expected {expected})")
        print("BUG REPRODUCED")
        return
    finally:
        srv.shutdown()

    print("\nResult:")
    print(df[[c for c in ("item_id", "answer", "answer_check") if c in df.columns]].to_string())
    ok = "answer_check" in df.columns and df["answer_check"].notna().all() and len(df) == expected
    print(f"\nrows: {len(df)} (expected {expected})")
    print("PASS" if ok else "BUG REPRODUCED")


if __name__ == "__main__":
    main()
