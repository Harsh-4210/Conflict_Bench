# ConflictBench public benchmark (P7-3)

This is the tooling for publishing `data/test_v1.jsonl` as a Hugging Face dataset with hidden answers, and for a
leaderboard Space that scores uploaded predictions. **Nothing here has been published.** The commands below are what
Harsh runs to do it. Every step before "Publish" works offline.

| Piece | File |
|---|---|
| Export script | `scripts/export_hf_dataset.py` |
| Leaderboard app | `leaderboard_space/app.py` (+ `README.md`, `requirements.txt`) |
| Tests | `tests/test_export_dataset.py`, `tests/test_leaderboard_space.py` |

## Why the answers are hidden

A benchmark only measures something if models cannot read the answers. If `test_full.jsonl` (which holds the ground
truth, the conflicts and the structured fields of every instruction) were public, a model, or a person fine-tuning
one, could train on it or look the answers up, and the score would say nothing. So the export splits the data in two:

- `public/test.jsonl` holds only `scenario_id`, `difficulty`, `domain`, `split`, `generator_version` and `messages`
  (exactly what `prompts.build_messages` produces). The tests check this file has no answer fields, no ID lists, no
  conflict explanations, and that instruction order in the prompt does not predict which instructions are followed.
- `private/test_full.jsonl` has the full records. It goes into a private dataset repo and is read only by the
  leaderboard Space, which returns scores and never the answers.

**Known limit, read before publishing.** The answers are only hidden if the private file is. Today
`data/test_v1.jsonl` (the same data, with answers) is committed in the public GitHub repository, and the generator is
public. Anyone can read the answers there, or regenerate similar sets. Also `resolver.py` solves the task exactly (see
`results/leaderboard.md`), so the benchmark ranks language models, not the task's intrinsic difficulty. Before
announcing the leaderboard as contamination-resistant, generate a fresh set with a seed you do not publish (for
example `python scripts/evaluate.py --generate 600 --seed <secret> --split test` writes under `results/data/`; copy the
file out of the repo and do not commit it), export that, and keep it out of git. Until then the leaderboard is still
useful, but the dataset card says so under "Caveats".

## 1. Export

```
python scripts/export_hf_dataset.py --data data/test_v1.jsonl --out dist/conflictbench-test-v1
```

Output (`dist/` is not in `.gitignore` yet; do not `git add` it, because `private/` holds the answers):

```
dist/conflictbench-test-v1/
  README.md                  dataset card (YAML front matter, task, answer format, scoring, caveats)
  manifest.json              SHA-256 and size of every other file
  public/test.jsonl          prompts only
  private/test_full.jsonl    full records, for the scorer only
```

The export is deterministic: running it again gives byte-identical files. Because `data/test_v1.sha256` sits next to
the input, the script checks the input against it and stops on a mismatch. The dataset card computes its counts and its
scoring formula from the data and from `verifier.py`; it contains no result numbers.

## 2. Publish (not done yet)

You need a Hugging Face account, a write token (`hf auth login`), and two dataset repos. Replace `YOUR_ORG`. The
current CLI is `hf`; older versions of `huggingface_hub` call the same commands `huggingface-cli upload` (same
arguments). The flags below were checked against `hf upload --help` and `hf repos create --help` of
huggingface_hub 1.12; the uploads themselves have not been run.

Create the private repo first, so it can never be created public by accident:

```
hf repos create YOUR_ORG/conflictbench-test-v1-private --repo-type dataset --private
hf repos create YOUR_ORG/conflictbench-test-v1 --repo-type dataset --public
```

Upload the public part (card, manifest, prompts; the private folder is excluded):

```
hf upload YOUR_ORG/conflictbench-test-v1 dist/conflictbench-test-v1 . --repo-type dataset --exclude "private/*" --commit-message "ConflictBench test v1 (prompts only)"
```

Upload the answers to the PRIVATE repo (the file lands at the repo root as `test_full.jsonl`):

```
hf upload YOUR_ORG/conflictbench-test-v1-private dist/conflictbench-test-v1/private . --repo-type dataset --commit-message "ConflictBench test v1 (answers)"
```

Then open both repos in the browser and confirm the first is public and shows only `README.md`, `manifest.json` and
`public/test.jsonl`, and the second is private. Use a **read-only fine-grained token** for the Space (next section),
not your write token.

## 3. Deploy the leaderboard Space (not done yet)

This is a new, separate Space. Do not push to the existing `hf` git remote (the live ConflictBench demo Space).

```
hf repos create YOUR_ORG/conflictbench-leaderboard --repo-type space --space-sdk gradio
hf upload YOUR_ORG/conflictbench-leaderboard leaderboard_space . --repo-type space
```

Then in the Space, Settings, Variables and secrets:

| Name | Kind | Value |
|---|---|---|
| `CONFLICTBENCH_REF` | variable | git tag or commit SHA of the ConflictBench repo to score with (branch names are refused) |
| `CONFLICTBENCH_PRIVATE_REPO` | variable | `YOUR_ORG/conflictbench-test-v1-private` |
| `HF_TOKEN` | secret | read-only token for that private repo |
| `CONFLICTBENCH_LEADERBOARD` | variable | optional: `/data/leaderboard.json` if you enable persistent storage |

Where the scoring code comes from: `leaderboard_space/` holds only the app. `verifier.py`, `generator.py`,
`scenario_io.py` and `scripts/evaluate.py` are not copied into it; on first start the app runs `git init`,
`git fetch --depth 1 <repo> <CONFLICTBENCH_REF>` and `git checkout FETCH_HEAD` into `conflictbench_repo/` (needs `git`
in the Space image and outbound access to GitHub; **not verified on a real Space**). Create the tag first, for example
`git tag benchmark-v1 && git push origin benchmark-v1`, and use it as `CONFLICTBENCH_REF`. If the fetch does not work
on your Space, copy those five files into `leaderboard_space/conflictbench_repo/` (keep `scripts/__init__.py` and
`scripts/evaluate.py` in a `scripts/` subfolder) from a checkout of the pinned ref and upload again; the app uses that
folder when it exists. Each leaderboard row stores the `code_ref` it was scored with. If you change the verifier, old
rows are not comparable until you re-score them.

Without persistent storage the leaderboard file is lost when the Space restarts or rebuilds. Back it up by downloading
the file, or enable storage and set `CONFLICTBENCH_LEADERBOARD`.

Run it locally first (inside a checkout, offline):

```
export CONFLICTBENCH_PRIVATE_SET=dist/conflictbench-test-v1/private/test_full.jsonl
export CONFLICTBENCH_LEADERBOARD=/tmp/leaderboard.json
python leaderboard_space/app.py
```

On Windows PowerShell use `$env:NAME = "value"`.

## 4. Submit (for model builders)

1. Download `public/test.jsonl` from the public dataset repo.
2. Send each row's `messages` to your model unchanged. Use greedy decoding or a low temperature.
3. Write one line per row to a file: `{"scenario_id": "<the row's scenario_id>", "completion": "<raw model output>"}`.
4. Open the Space, enter a model name, upload the file, press **Score and submit**.

A file is accepted only if it has exactly one line for every scenario: missing IDs, duplicate IDs, unknown IDs, lines
that are not JSON objects, a non-string `completion`, a completion longer than 60,000 characters, a file larger than
25 MB, or a bad model name all give a clear message and add nothing to the table. The same file under the same model
name cannot be added twice. You see the composite score, a 95% bootstrap interval (1000 resamples over scenarios,
seed 0, the same `scripts/evaluate.py` code as the repo), the mean per difficulty and the mean per rubric. The same file can also be scored offline against
your own copy of the set with `verifier.score`.

To compare your row with rule-based baselines, see `results/leaderboard.md` in the repository; those rows were made
by `scripts/evaluate.py` on `data/test_v1.jsonl` and are the same data the Space scores.

## Safety notes

- All user text (model name, IDs in error messages) is HTML-escaped before it is shown. Visitors never see a traceback.
- The private set is read from a local path or a private Hub repo; error messages shown to visitors say only that it
  could not be loaded, and never contain the token or paths.
- `huggingface_hub` is imported only when the private repo is used, and nothing contacts the network at import time.
- The Space cannot tell whether a completion really came from the named model. Treat rows as self-reported.
