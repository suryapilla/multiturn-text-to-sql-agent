"""SQL Agent Trajectory Forge — Day 1 core.

Generates schema-grounded text-to-SQL pairs against a REAL database (Chinook),
validates them by EXECUTING the SQL read-only (not just linting), and scores
them with an LLM judge.

Run:
    uv run forge.py --preview            # 10 rows, inspect quickly
    uv run forge.py --num-records 200    # full run
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

DB_PATH = Path(__file__).parent / "chinook.sqlite"
MODEL = "nvidia-text"  # default NVIDIA Build alias; needs NVIDIA_API_KEY


# --------------------------------------------------------------------------
# Schema context: read the real schema so questions are grounded, not invented
# --------------------------------------------------------------------------
def load_schema(db_path: Path = DB_PATH) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    return "\n\n".join(r[0] for r in rows)


SCHEMA = load_schema()


# --------------------------------------------------------------------------
# Execution validator: the key upgrade over lint-only validation.
# Runs each query read-only with a timeout and records what happened.
# --------------------------------------------------------------------------
def execute_sql(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for sql in df["sql"]:
        rec = {"is_valid": False, "error_message": "", "row_count": 0}
        try:
            con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)  # read-only
            con.set_progress_handler(lambda: 1, 2_000_000)  # abort runaway queries
            rows = con.execute(sql).fetchmany(1000)
            con.close()
            rec["row_count"] = len(rows)
            rec["is_valid"] = len(rows) > 0  # empty result = likely wrong filter
            if not rows:
                rec["error_message"] = "query ran but returned no rows"
        except Exception as e:  # syntax error, missing column, timeout, write attempt
            rec["error_message"] = f"{type(e).__name__}: {e}"[:300]
        results.append(rec)
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Pipeline config
# --------------------------------------------------------------------------
def build_config() -> dd.DataDesignerConfigBuilder:
    b = dd.DataDesignerConfigBuilder()

    # Diversity controls
    b.add_column(dd.SamplerColumnConfig(
        name="difficulty", sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["easy", "medium", "hard"],
                                        weights=[0.3, 0.4, 0.3]),
    ))
    b.add_column(dd.SamplerColumnConfig(
        name="query_type", sampler_type=dd.SamplerType.SUBCATEGORY,
        params=dd.SubcategorySamplerParams(category="difficulty", values={
            "easy": ["single-table filter", "sorting and limits", "simple count"],
            "medium": ["two-table join", "group by with aggregation", "date filtering"],
            "hard": ["multi-join aggregation", "subquery or CTE", "window function ranking"],
        }),
    ))
    b.add_column(dd.SamplerColumnConfig(
        name="user_role", sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=[
            "sales manager", "finance analyst", "marketing lead",
            "customer support lead", "music catalog curator",
        ]),
    ))

    # Natural-language question grounded in the real schema
    b.add_column(dd.LLMTextColumnConfig(
        name="question", model_alias=MODEL,
        system_prompt="You write realistic business questions that analysts ask a SQL assistant.",
        prompt=(
            "Database schema (SQLite, a digital music store):\n{{ schema }}\n\n"
            "Write ONE question a {{ user_role }} would ask about this data. "
            "It should require a {{ difficulty }} query of type: {{ query_type }}. "
            "Use business language, not table or column names. "
            "Only ask about data that exists in the schema. Output only the question."
        ),
    ))

    # SQL answer
    b.add_column(dd.LLMCodeColumnConfig(
        name="sql", model_alias=MODEL, code_lang=dd.CodeLang.SQL_SQLITE,
        system_prompt="You are an expert SQLite engineer. Write read-only SELECT queries only.",
        prompt=(
            "Schema:\n{{ schema }}\n\nQuestion: {{ question }}\n\n"
            "Write a single SQLite SELECT query that answers the question. "
            "Use only tables and columns in the schema. No comments."
        ),
    ))

    # Gate 1: does it actually run and return rows?
    b.add_column(dd.ValidationColumnConfig(
        name="execution_check", target_columns=["sql"],
        validator_type=dd.ValidatorType.LOCAL_CALLABLE,
        validator_params=dd.LocalCallableValidatorParams(validation_function=execute_sql),
        batch_size=20,
    ))

    # Gate 2: does it answer the question that was asked?
    b.add_column(dd.LLMJudgeColumnConfig(
        name="judge", model_alias=MODEL,
        prompt=(
            "Schema:\n{{ schema }}\n\nQuestion: {{ question }}\n\nSQL:\n{{ sql }}\n\n"
            "Score how well the SQL answers the question."
        ),
        scores=[
            dd.Score(name="Correctness",
                     description="Does the SQL return exactly what the question asks?",
                     options={4: "Fully correct", 3: "Minor issue", 2: "Partially correct",
                              1: "Mostly wrong", 0: "Wrong or unrelated"}),
            dd.Score(name="Question Realism",
                     description="Is this a natural question a business user would ask?",
                     options={4: "Very natural", 3: "Natural", 2: "Somewhat artificial",
                              1: "Artificial", 0: "Nonsensical"}),
        ],
    ))
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--num-records", type=int, default=10)
    args = ap.parse_args()

    # schema is passed as a constant column so every prompt can reference it
    seed = pd.DataFrame({"schema": [SCHEMA]})
    builder = build_config()
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed))

    designer = DataDesigner()
    if args.preview:
        preview = designer.preview(config_builder=builder, num_records=args.num_records)
        preview.display_sample_record()
        df = preview.dataset
    else:
        results = designer.create(builder, num_records=args.num_records)
        df = results.load_dataset()

    out = Path("output"); out.mkdir(exist_ok=True)
    df.drop(columns=["schema"], errors="ignore").to_json(
        out / "forge_day1.jsonl", orient="records", lines=True)
    print(f"\nSaved {len(df)} rows to {out/'forge_day1.jsonl'}")


if __name__ == "__main__":
    main()