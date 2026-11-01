"""
Documentation checks (FIX_BLUEPRINT.md Phase 6).

- every relative markdown link resolves, with exact letter case (scripts/check_links.py);
- the weight table in docs/REWARD_DESIGN.md equals verifier.RUBRIC_WEIGHTS;
- openenv.yaml points at files that exist and matches the code and configs;
- withdrawn claims do not come back into the public docs.
"""

import os
import re
from pathlib import Path

import pytest
import yaml

import verifier
from scripts import check_links

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def test_all_markdown_links_resolve():
    problems = check_links.check_all(ROOT)
    assert problems == [], "\n".join(problems)


def test_link_checker_catches_missing_and_wrong_case_targets(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "Guide.md").write_text("# guide\n", encoding="utf-8")
    md = tmp_path / "index.md"
    md.write_text(
        "\n".join([
            "[ok](docs/Guide.md) [ok with anchor](./docs/Guide.md#part) [page anchor](#top)",
            "[web](https://example.com/missing.md) [mail](mailto:someone@example.com)",
            "[wrong case](docs/guide.md)",
            "![missing image](assets/nope.png)",
            "`[in code](not/a/link.md)`",
            "```",
            "[in a fence](also/not/a/link.md)",
            "```",
            "[ref]: missing/ref.md",
        ]),
        encoding="utf-8",
    )
    problems = check_links.check_file(md, tmp_path)
    targets = sorted(p.rsplit("-> ", 1)[1] for p in problems)
    assert targets == ["assets/nope.png", "docs/guide.md", "missing/ref.md"]


def test_link_checker_exit_code(tmp_path):
    (tmp_path / "a.md").write_text("[x](b.md)\n", encoding="utf-8")
    assert check_links.main(["--root", str(tmp_path)]) == 1
    (tmp_path / "b.md").write_text("back to [a](a.md)\n", encoding="utf-8")
    assert check_links.main(["--root", str(tmp_path)]) == 0


def test_architecture_doc_has_upper_case_name():
    assert "ARCHITECTURE.md" in os.listdir(ROOT / "docs")
    assert "architecture.md" not in os.listdir(ROOT / "docs")


# ---------------------------------------------------------------------------
# Reward weights stated in the docs
# ---------------------------------------------------------------------------

def _doc_weights() -> dict:
    text = (ROOT / "docs" / "REWARD_DESIGN.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- rubric-weights:start -->(.*?)<!-- rubric-weights:end -->", text, re.S)
    assert m, "docs/REWARD_DESIGN.md lost its rubric-weights table markers"
    weights = {}
    for row in m.group(1).splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        key = re.fullmatch(r"`(\w+)`", cells[1])
        if key and re.fullmatch(r"\d+(\.\d+)?", cells[2]):
            weights[key.group(1)] = float(cells[2])
    return weights


def test_reward_design_weights_match_verifier():
    assert _doc_weights() == verifier.RUBRIC_WEIGHTS


# ---------------------------------------------------------------------------
# openenv.yaml
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest() -> dict:
    with (ROOT / "openenv.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_openenv_yaml_paths_exist(manifest):
    paths = [
        manifest["entry_point"],
        manifest["training_entry"],
        manifest["local_training_entry"],
        manifest["server_entry"],
        manifest["environment"]["file"],
        manifest["environment"]["test_set"],
        manifest["reward_spec"],
        *manifest["training_configs"],
    ]
    missing = [p for p in paths if not (ROOT / p).is_file()]
    assert missing == []


def test_openenv_yaml_reward_matches_verifier(manifest):
    stated = {r["name"]: r["weight"] for r in manifest["reward_rubrics"]}
    assert stated == verifier.RUBRIC_WEIGHTS
    assert manifest["reward_format_gate"]["schema_errors"] == verifier.SCHEMA_ERROR_FACTOR
    assert "results" not in manifest  # measured numbers live in results/, not in the manifest


def test_openenv_yaml_training_matches_configs(manifest):
    stated = manifest["training"]["configs"]
    on_disk = {Path(p).stem: p for p in manifest["training_configs"]}
    assert set(stated) == set(on_disk)
    for name, path in on_disk.items():
        with (ROOT / path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert stated[name]["model"] == cfg["model"], name
        assert stated[name]["lora_rank"] == cfg["lora_r"], name
        assert stated[name]["lora_alpha"] == cfg["lora_alpha"], name


# ---------------------------------------------------------------------------
# Withdrawn claims
# ---------------------------------------------------------------------------

PUBLIC_DOCS = ("README.md", "blog.md", "PRD.md", "openenv.yaml")
WITHDRAWN = ("257%", "never stated in the prompt", "ungameable", "non-gameable", "0.5003")


@pytest.mark.parametrize("name", PUBLIC_DOCS)
def test_withdrawn_claims_stay_out(name):
    text = (ROOT / name).read_text(encoding="utf-8").lower()
    found = [s for s in WITHDRAWN if s.lower() in text]
    assert found == [], f"{name} contains withdrawn claim(s): {found}"


def test_readme_keeps_hf_spaces_front_matter():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n", 2)[1])
    assert front["sdk"] == "gradio" and front["app_file"] == "app.py"
