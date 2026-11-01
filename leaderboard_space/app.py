"""
ConflictBench — Public leaderboard Space (Gradio)

A visitor uploads a predictions JSONL, one line per scenario:
    {"scenario_id": "...", "completion": "<the raw model output>"}
plus a model name. The app scores it with the repo's own ``verifier.score`` against the PRIVATE
answer set (never shown), shows the per-difficulty and per-rubric breakdown with a bootstrap
confidence interval, and appends the result to a leaderboard file.

Configuration (environment variables; nothing is read from the network at import time):
  CONFLICTBENCH_PRIVATE_SET     local path to private/test_full.jsonl (or a folder holding it)
  CONFLICTBENCH_PRIVATE_REPO    OR the id of a PRIVATE HF dataset repo that holds test_full.jsonl
  HF_TOKEN                      read token for that repo (a Space secret)
  CONFLICTBENCH_PRIVATE_FILE    file name inside that repo (default test_full.jsonl)
  CONFLICTBENCH_LEADERBOARD     path of the leaderboard JSON file (default: leaderboard.json next to this file;
                                point it at /data/leaderboard.json when the Space has persistent storage)
  CONFLICTBENCH_REPO_DIR        folder holding the ConflictBench code (verifier.py, generator.py, scripts/...)
  CONFLICTBENCH_REF             git TAG or COMMIT SHA of the repo to fetch when no code folder is found
  CONFLICTBENCH_REPO_URL        repo to fetch from (default: the GitHub repo)

Where the scoring code comes from: the verifier, generator and evaluate helpers are NOT copied here.
Inside the repo they sit one level up. On a Space (which holds only this folder) the app fetches the
repo at the pinned CONFLICTBENCH_REF the first time it starts. See docs/BENCHMARK.md.
"""

import datetime
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_REPO_URL = "https://github.com/Harsh-4210/Conflict_Bench.git"

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 25 * 1024 * 1024      # whole predictions file
MAX_COMPLETION_CHARS = 60_000            # one completion
MAX_MODEL_NAME_CHARS = 80
MAX_ERRORS_COLLECTED = 200               # stop reading a hopeless file after this many problems
MAX_ERRORS_SHOWN = 12

# ---------------------------------------------------------------------------
# Finding the ConflictBench code (no network unless the code folder is missing)
# ---------------------------------------------------------------------------

_BAD_REFS = {"", "main", "master", "head", "replace_with_tag_or_commit"}
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _is_repo_dir(path: Path) -> bool:
    return (path / "verifier.py").is_file() and (path / "generator.py").is_file() \
        and (path / "scripts" / "evaluate.py").is_file()


def validate_ref(ref: str) -> str:
    """A pinned ref is a tag or a commit SHA. Branch names move, so they are refused."""
    ref = (ref or "").strip()
    if ref.lower() in _BAD_REFS or not _REF_RE.match(ref):
        raise RuntimeError(
            f"CONFLICTBENCH_REF must be a git tag or commit SHA, not a branch name (got {ref!r}).")
    return ref


def fetch_repo_at_ref(ref: str, dest: Path, url: str = DEFAULT_REPO_URL) -> Path:
    """Fetch exactly one commit of the repo into dest (works for tags and SHAs). Uses the network."""
    ref = validate_ref(ref)
    dest.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q"],
                 ["fetch", "-q", "--depth", "1", url, ref],
                 ["checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, timeout=300)
    return dest


def find_repo_root() -> Path:
    """The folder holding the scoring code: env override, the parent of this folder, a local clone, or a pinned fetch."""
    env_dir = os.environ.get("CONFLICTBENCH_REPO_DIR")
    candidates = ([Path(env_dir)] if env_dir else []) + [HERE.parent, HERE / "conflictbench_repo"]
    for c in candidates:
        if _is_repo_dir(c):
            return c
    ref = os.environ.get("CONFLICTBENCH_REF")
    if ref:
        dest = HERE / "conflictbench_repo"
        fetch_repo_at_ref(ref, dest, os.environ.get("CONFLICTBENCH_REPO_URL", DEFAULT_REPO_URL))
        if _is_repo_dir(dest):
            return dest
    raise RuntimeError(
        "ConflictBench code not found. Run this app from inside the repository, set CONFLICTBENCH_REPO_DIR, "
        "or set CONFLICTBENCH_REF to a git tag or commit SHA so the Space can fetch it.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scenario_io import load_scenarios                                      # noqa: E402
from scripts.evaluate import RUBRICS, sha256_file, summarise                # noqa: E402
from verifier import RUBRIC_WEIGHTS, score                                  # noqa: E402

# ---------------------------------------------------------------------------
# Private answer set
# ---------------------------------------------------------------------------


class PrivateSetError(Exception):
    """The private answer set cannot be loaded. The message is safe to show to visitors."""


@dataclass
class PrivateSet:
    scenarios: list
    sha256: str


_PRIVATE_CACHE: Dict[Tuple[str, str], PrivateSet] = {}


def reset_cache() -> None:
    _PRIVATE_CACHE.clear()


def _resolve_local(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        for rel in ("test_full.jsonl", "private/test_full.jsonl"):
            if (p / rel).is_file():
                return p / rel
    return p


def _download_private(repo: str, token: Optional[str]) -> Path:
    """Fetch the private file from the Hub. The huggingface_hub import is lazy: only this branch needs it."""
    if not token:
        raise PrivateSetError("The leaderboard is not configured (missing HF_TOKEN for the private answers).")
    filename = os.environ.get("CONFLICTBENCH_PRIVATE_FILE", "test_full.jsonl")
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", token=token))
    except Exception as exc:  # never include the token or the raw error text
        print(f"[leaderboard] could not fetch private set: {type(exc).__name__}", file=sys.stderr)
        raise PrivateSetError("The private answer set could not be fetched. Try again later.") from None


def load_private_set() -> PrivateSet:
    """Load (and cache) the full scenarios from CONFLICTBENCH_PRIVATE_SET or the private HF repo."""
    local = os.environ.get("CONFLICTBENCH_PRIVATE_SET")
    repo = os.environ.get("CONFLICTBENCH_PRIVATE_REPO")
    if local:
        path, key = _resolve_local(local), ("local", local)
    elif repo:
        key = ("hub", repo)
        path = None
    else:
        raise PrivateSetError("The leaderboard is not configured (no private answer set).")
    if key in _PRIVATE_CACHE:
        return _PRIVATE_CACHE[key]
    if path is None:
        path = _download_private(repo, os.environ.get("HF_TOKEN"))
    if not path.is_file():
        raise PrivateSetError("The private answer set file was not found.")
    try:
        scenarios = load_scenarios(path)
    except Exception as exc:
        print(f"[leaderboard] could not read private set: {type(exc).__name__}", file=sys.stderr)
        raise PrivateSetError("The private answer set could not be read.") from None
    ids = [s.scenario_id for s in scenarios]
    if not scenarios or len(set(ids)) != len(ids):
        raise PrivateSetError("The private answer set is empty or has duplicate IDs.")
    loaded = PrivateSet(scenarios=scenarios, sha256=sha256_file(path))
    _PRIVATE_CACHE[key] = loaded
    return loaded


# ---------------------------------------------------------------------------
# Validating a submission
# ---------------------------------------------------------------------------


def _short(value: object, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def validate_model_name(name: object) -> Tuple[str, Optional[str]]:
    """(clean name, error). Any printable text is allowed; it is escaped wherever it is shown."""
    text = name.strip() if isinstance(name, str) else ""
    if not text:
        return "", "Enter a model name."
    if len(text) > MAX_MODEL_NAME_CHARS:
        return "", f"The model name is too long (at most {MAX_MODEL_NAME_CHARS} characters)."
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return "", "The model name contains control characters."
    return text, None


def parse_predictions(raw: bytes, valid_ids: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Read the uploaded JSONL. Returns (scenario_id -> completion, errors).
    Errors are plain text. The submission is acceptable only when errors is empty: every scenario
    must be answered exactly once, with no unknown IDs and no malformed lines.
    """
    valid = set(valid_ids)
    errors: List[str] = []
    if len(raw) > MAX_UPLOAD_BYTES:
        return {}, [f"The file is too large ({len(raw) // (1024 * 1024)} MB; the limit is "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."]
    if not raw.strip():
        return {}, ["The file is empty."]
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {}, ["The file is not valid UTF-8 text."]

    preds: Dict[str, str] = {}
    first_line: Dict[str, int] = {}
    for lineno, line in enumerate(text.split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        if len(errors) >= MAX_ERRORS_COLLECTED:
            errors.append("Too many problems; stopped reading the file.")
            return preds, errors
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError) as exc:
            errors.append(f"Line {lineno}: not valid JSON ({_short(exc, 60)}).")
            continue
        if not isinstance(obj, dict):
            errors.append(f"Line {lineno}: expected a JSON object with scenario_id and completion.")
            continue
        sid, completion = obj.get("scenario_id"), obj.get("completion")
        if not isinstance(sid, str) or not sid:
            errors.append(f"Line {lineno}: scenario_id is missing or not a string.")
            continue
        if not isinstance(completion, str):
            errors.append(f"Line {lineno}: completion is missing or not a string.")
            continue
        if len(completion) > MAX_COMPLETION_CHARS:
            errors.append(f"Line {lineno}: completion is longer than {MAX_COMPLETION_CHARS} characters.")
            continue
        if sid not in valid:
            errors.append(f"Line {lineno}: unknown scenario_id {_short(sid)!r}.")
            continue
        if sid in preds:
            errors.append(f"Line {lineno}: scenario_id {_short(sid)!r} already appeared on line {first_line[sid]}.")
            continue
        preds[sid] = completion
        first_line[sid] = lineno

    missing = [i for i in valid_ids if i not in preds]
    if missing:
        errors.append(f"{len(missing)} of {len(valid_ids)} scenarios have no prediction "
                      f"(for example {_short(missing[0])!r}). Every scenario must be answered once.")
    return preds, errors


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@dataclass
class SubmissionResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    summary: Optional[dict] = None
    entry: Optional[dict] = None


def score_predictions(preds: Dict[str, str], private: PrivateSet) -> dict:
    """Score every scenario with verifier.score and aggregate with scripts/evaluate.summarise."""
    rows = []
    for sc in private.scenarios:
        rows.append({"scenario_id": sc.scenario_id, "difficulty": sc.difficulty, "sample": 0,
                     "completion": "", "score": score(preds[sc.scenario_id], sc).to_dict()})
    return summarise(rows, 0, 1)


def evaluate_submission(raw: bytes, model_name: object, private: PrivateSet) -> SubmissionResult:
    """Validate and score one upload. Never raises on bad input."""
    name, name_error = validate_model_name(model_name)
    preds, errors = parse_predictions(raw, [s.scenario_id for s in private.scenarios])
    if name_error:
        errors = [name_error, *errors]
    if errors:
        return SubmissionResult(ok=False, errors=errors)
    summary = score_predictions(preds, private)
    per_diff = summary["per_difficulty"]
    comp = summary["composite"]
    entry = {
        "model": name,
        "composite": comp["mean"],
        "ci95_low": comp["ci95_low"],
        "ci95_high": comp["ci95_high"],
        "per_difficulty": {d: v["mean"] for d, v in per_diff.items()},
        "parse_failure_rate": summary["parse_failure_rate"],
        "n": summary["n"],
        "submitted_utc": _utc_now(),
        "submission_sha256": hashlib.sha256(raw).hexdigest(),
        "set_sha256": private.sha256,
        "code_ref": os.environ.get("CONFLICTBENCH_REF", ""),  # which verifier produced the score
    }
    return SubmissionResult(ok=True, summary=summary, entry=entry)


# ---------------------------------------------------------------------------
# Leaderboard file
# ---------------------------------------------------------------------------

_LB_LOCK = threading.Lock()


class LeaderboardError(Exception):
    """The leaderboard file cannot be read or updated. The message is safe to show to visitors."""


def leaderboard_path() -> Path:
    return Path(os.environ.get("CONFLICTBENCH_LEADERBOARD") or (HERE / "leaderboard.json"))


def read_leaderboard(path: Optional[Path] = None) -> List[dict]:
    path = Path(path) if path else leaderboard_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise LeaderboardError("The leaderboard file is unreadable. The maintainer has to repair it.") from None
    if not isinstance(data, list):
        raise LeaderboardError("The leaderboard file is unreadable. The maintainer has to repair it.")
    return [e for e in data if isinstance(e, dict)]


def sort_entries(entries: Sequence[dict]) -> List[dict]:
    """Best composite first; ties go to the earlier submission, then the model name."""
    return sorted(entries, key=lambda e: (-float(e.get("composite", 0.0)),
                                          str(e.get("submitted_utc", "")), str(e.get("model", ""))))


def append_entry(entry: dict, path: Optional[Path] = None) -> List[dict]:
    """Add one row (atomic write) and return the sorted table. The same file for the same model is refused."""
    path = Path(path) if path else leaderboard_path()
    with _LB_LOCK:
        entries = read_leaderboard(path)
        for e in entries:
            if e.get("model") == entry["model"] and e.get("submission_sha256") == entry["submission_sha256"] \
                    and e.get("set_sha256") == entry["set_sha256"]:
                raise LeaderboardError("This exact file is already on the leaderboard for that model name.")
        entries.append(entry)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            os.replace(tmp, path)
        except OSError:
            raise LeaderboardError("The leaderboard file could not be written.") from None
    return sort_entries(entries)


# ---------------------------------------------------------------------------
# HTML (every piece of user text goes through esc())
# ---------------------------------------------------------------------------

_STYLE = ("<style>.cb table{border-collapse:collapse;width:100%;margin:.5em 0}"
          ".cb th,.cb td{border:1px solid rgba(128,128,128,.4);padding:4px 8px;text-align:right}"
          ".cb th:nth-child(-n+2),.cb td:nth-child(-n+2){text-align:left}"
          ".cb .err{color:#b3261e}.cb .muted{opacity:.7}</style>")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _f(x: object, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def render_errors(errors: Sequence[str]) -> str:
    shown = [f"<li>{esc(e)}</li>" for e in errors[:MAX_ERRORS_SHOWN]]
    if len(errors) > MAX_ERRORS_SHOWN:
        shown.append(f"<li>... and {len(errors) - MAX_ERRORS_SHOWN} more.</li>")
    return (f'{_STYLE}<div class="cb"><p class="err"><b>Submission rejected. Nothing was added to the '
            f'leaderboard.</b></p><ul>{"".join(shown)}</ul></div>')


def render_message(message: str) -> str:
    return f'{_STYLE}<div class="cb"><p class="err"><b>{esc(message)}</b></p></div>'


def render_summary(model: str, summary: dict) -> str:
    c = summary["composite"]
    diff_rows = "".join(f"<tr><td>Difficulty {esc(d)}</td><td></td><td>{_f(v['mean'])}</td><td>{esc(v['n'])}</td></tr>"
                        for d, v in summary["per_difficulty"].items())
    rubric_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc('weight ' + format(RUBRIC_WEIGHTS[k], '.2f') if k in RUBRIC_WEIGHTS else 'logged only')}"
        f"</td><td>{_f(v)}</td><td></td></tr>"
        for k, v in summary["per_rubric"].items() if k in RUBRICS)
    return (f'{_STYLE}<div class="cb"><h3>{esc(model)}</h3>'
            f'<p>Composite <b>{_f(c["mean"])}</b> (95% CI {_f(c["ci95_low"])} to {_f(c["ci95_high"])}, '
            f'{esc(c["n_resamples"])} bootstrap resamples over {esc(summary["n"])} scenarios). '
            f'Replies with no parseable answer: {_f(summary["parse_failure_rate"])}.</p>'
            f'<table><tr><th>Difficulty</th><th></th><th>Mean composite</th><th>Scenarios</th></tr>{diff_rows}</table>'
            f'<table><tr><th>Rubric</th><th>Role</th><th>Mean</th><th></th></tr>{rubric_rows}</table>'
            f'<p class="muted">The composite is the weighted sum of the three weighted rubrics times the format gate; '
            f'the other rubrics are logged only.</p></div>')


def render_leaderboard(entries: Sequence[dict]) -> str:
    if not entries:
        return f'{_STYLE}<div class="cb"><p class="muted">No submissions yet.</p></div>'
    rows = []
    for rank, e in enumerate(sort_entries(entries), start=1):
        pd = e.get("per_difficulty") or {}
        rows.append(
            f"<tr><td>{rank}</td><td>{esc(e.get('model', ''))}</td><td>{_f(e.get('composite'))}</td>"
            f"<td>{_f(e.get('ci95_low'))} - {_f(e.get('ci95_high'))}</td>"
            f"<td>{_f(pd.get('1'))}</td><td>{_f(pd.get('2'))}</td><td>{_f(pd.get('3'))}</td>"
            f"<td>{_f(e.get('parse_failure_rate'))}</td><td>{esc(str(e.get('submitted_utc', ''))[:10])}</td></tr>")
    return (f'{_STYLE}<div class="cb"><table><tr><th>Rank</th><th>Model</th><th>Composite</th><th>95% CI</th>'
            f'<th>D1</th><th>D2</th><th>D3</th><th>Parse fail</th><th>Date</th></tr>{"".join(rows)}</table>'
            f'<p class="muted">Self-reported predictions, scored by the repo verifier. Not independently verified.</p></div>')


# ---------------------------------------------------------------------------
# One submission end to end (the Gradio handler calls this; tests call it directly)
# ---------------------------------------------------------------------------


def process_submission(file_path: Optional[str], model_name: object) -> Tuple[str, str, str]:
    """Returns (status html, result html, leaderboard html). Never raises."""
    board = lambda: _safe_board_html()  # noqa: E731
    try:
        if not file_path:
            return render_message("Choose a predictions file first."), "", board()
        try:
            size = os.path.getsize(file_path)
            if size > MAX_UPLOAD_BYTES:
                return render_errors([f"The file is too large (the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."]), "", board()
            with open(file_path, "rb") as f:
                raw = f.read(MAX_UPLOAD_BYTES + 1)
        except OSError:
            return render_message("The uploaded file could not be read."), "", board()
        try:
            private = load_private_set()
        except PrivateSetError as exc:
            return render_message(str(exc)), "", board()
        result = evaluate_submission(raw, model_name, private)
        if not result.ok:
            return render_errors(result.errors), "", board()
        try:
            entries = append_entry(result.entry)
        except LeaderboardError as exc:
            return render_message(str(exc)), render_summary(result.entry["model"], result.summary), board()
        status = f'{_STYLE}<div class="cb"><p><b>Scored and added to the leaderboard.</b></p></div>'
        return status, render_summary(result.entry["model"], result.summary), render_leaderboard(entries)
    except Exception as exc:  # last resort: a visitor must never see a traceback
        print(f"[leaderboard] unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return render_message("Something went wrong while scoring. Nothing was added."), "", board()


def _safe_board_html() -> str:
    try:
        return render_leaderboard(read_leaderboard())
    except LeaderboardError as exc:
        return render_message(str(exc))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INTRO = """# ConflictBench leaderboard

Run your model on every row of the public `test.jsonl` and upload one line per scenario:
`{"scenario_id": "...", "completion": "<raw model output>"}`. The file is scored against private answers with the
repo verifier (`verifier.score`). Each scenario must appear exactly once. Max file size 25 MB."""


def build_ui():
    import gradio as gr  # imported here so the scoring code above works without gradio

    with gr.Blocks(title="ConflictBench leaderboard", analytics_enabled=False) as demo:
        gr.Markdown(INTRO)
        model = gr.Textbox(label="Model name", max_length=MAX_MODEL_NAME_CHARS)
        upload = gr.File(label="Predictions (.jsonl)", file_types=[".jsonl", ".json", ".txt"], type="filepath")
        go = gr.Button("Score and submit", variant="primary")
        status = gr.HTML()
        result = gr.HTML()
        gr.Markdown("## Leaderboard")
        board = gr.HTML()
        refresh = gr.Button("Refresh leaderboard")
        go.click(process_submission, [upload, model], [status, result, board])
        refresh.click(_safe_board_html, None, board)
        demo.load(_safe_board_html, None, board)
    return demo


if __name__ == "__main__":
    build_ui().launch()
