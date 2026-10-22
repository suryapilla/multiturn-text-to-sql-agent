**Title:** `ValidationColumnConfig` misaligns rows and discards the whole batch after an upstream row is dropped

---

### Priority Level

Medium (Annoying but has workaround)

### Describe the bug

If a column that runs before a `ValidationColumnConfig` fails non-retryably for one row (e.g. the LLM
endpoint returns a 400), the validation column returns **more** rows than the batch contains. The
scheduler then rejects the validation result and the **entire batch** is lost, not just the failed row.

```text
[WARNING] Non-retryable failure on answer[rg=0, row=1]
[WARNING] Non-retryable failure on answer_check[rg=0, row=None]: Batch generator for column
'answer_check' returned 4 rows but 3 were expected (rg=0).
```

With 4 records and one failed LLM call, `preview()` produces 0 records and raises
`DataDesignerGenerationError` instead of returning the 3 surviving rows.

It only happens when the dropped row is **not** the last one in the row group. Dropping the last row
leaves the index contiguous (0, 1, 2), so the failure looks intermittent when real API errors hit
random rows.

**Root cause.** The async scheduler builds the batch input with the surviving rows' original labels:

`data_designer/engine/dataset_builders/async_scheduler.py:2294-2296`
```python
active_row_indices = [ri for ri in range(rg_size) if ri not in pre_dropped]
...
input_df = lazy.pd.DataFrame(records, index=active_row_indices)
```

`ValidationColumnGenerator.generate()` builds its results with a fresh `RangeIndex` and joins
column-wise on labels:

`data_designer/engine/column_generators/generators/validation.py:102-103`
```python
validation_results = lazy.pd.DataFrame({self.config.name: outputs_as_dicts})
return lazy.pd.concat([data, validation_results], axis=1)
```

With `data.index == [0, 2, 3]` and `validation_results.index == [0, 1, 2]`, the outer join yields
4 rows (labels 0–3), and the misaligned rows hold NaN in the data columns or in the validation column.

### Steps/Code to reproduce bug

The attached script uses a local mock OpenAI-compatible endpoint (stdlib `http.server`) that returns
HTTP 400 for one chosen row. Real Data Designer handles everything else: the seed dataset, the LLM
column, the async scheduler and the validation column. No credentials or network access are needed.

```sh
uv run --script repro_validation_index.py                # row 1 fails  -> BUG REPRODUCED (0 rows)
uv run --script repro_validation_index.py --fail-row 0   # row 0 fails  -> BUG REPRODUCED (0 rows)
uv run --script repro_validation_index.py --fail-row 3   # last row     -> PASS (3 rows)
uv run --script repro_validation_index.py --fail-row -1  # no failure   -> PASS (4 rows)
```

Pipeline: `DataFrameSeedSource(item_id 0..3)` → `LLMTextColumnConfig` → `ValidationColumnConfig`
(`LOCAL_CALLABLE`).

Results (same for every version tested):

| data-designer | fail row 0 | fail row 1 | fail row 3 (last) | no failure |
|---|---|---|---|---|
| 0.9.5 | bug | bug | pass | pass |
| 0.9.6 (latest) | bug | bug | pass | pass |
| 1.0.0rc1 | bug | bug | pass | pass |

Tested with Python 3.12, pandas 3.0.6, Linux. Pin a version with
`uv run --with data-designer==<version> --script repro_validation_index.py`.

[repro_validation_index.py](<attach file>)

### Expected behavior

The dropped row is excluded and the validation column is generated for the surviving rows. In the
repro, `preview()` returns 3 rows, each with an `answer_check` result.

### Agent Diagnostic / Prior Investigation

This bug was found while building a multi-turn text-to-SQL trajectory pipeline on Data Designer, when
rate-limit errors on single rows wiped out whole batches. An AI coding agent helped write the issue and
the repro; I reviewed both and ran the repro myself.

- Searched existing issues and PRs (`ValidationColumnConfig`, "rows but … were expected", validation
  index/`reset_index`). Found nothing related.
- Diffed the installed 0.9.5 engine against 0.9.6 and 1.0.0rc1. `validation.py` and
  `async_scheduler.py` are unchanged.
- Confirmed the fix direction by monkeypatching `ValidationColumnGenerator.generate` to call the
  original with `data.reset_index(drop=True)` and then restore `data.index` on the result. With that
  patch, the repro passes on 0.9.6 and 1.0.0rc1 (3 rows, all validated).

**Suggested fix** (in `ValidationColumnGenerator.generate`):

```python
validation_results = lazy.pd.DataFrame({self.config.name: outputs_as_dicts}, index=data.index)
return lazy.pd.concat([data, validation_results], axis=1)
```

Other full-column generators that rebuild a frame and `concat(axis=1)` may need the same check.

### Additional context

Workaround: do the check in a cell-by-cell `CustomColumnConfig` (`@dd.custom_column_generator`)
instead of `ValidationColumnConfig`. It is row-safe.

### Checklist

- [x] I reproduced this issue or provided a minimal example
- [x] I searched the docs/issues myself, or had my agent do so
- [x] If I used an agent, I included its diagnostics above
