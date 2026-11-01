"""
ConflictBench — Markdown link checker
Every relative link in the repo's markdown files must point to a file or folder
that exists, with the exact letter case (GitHub is case-sensitive even when the
local disk is not). Anchors (#...) and queries (?...) are ignored; http(s),
mailto and other URL schemes are skipped. Links inside code blocks and inline
code are not links and are skipped too.

Usage: python scripts/check_links.py [--root DIR]     exit 0 = all good, 1 = broken links
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+?\1")
# [text](target "title") and ![alt](target). The text part may hold one level of nested brackets.
_INLINE_LINK_RE = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*[\"')])?\s*\)")
# [label]: target   (reference-style definitions)
_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def markdown_files(root: Path) -> List[Path]:
    """Tracked (and new, not ignored) .md files under root; falls back to a directory walk without git."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
            cwd=str(root), capture_output=True, timeout=30,
        )
        if out.returncode == 0:
            names = [n for n in out.stdout.decode("utf-8").split("\0") if n]
            return sorted({root / n for n in names if (root / n).is_file()})
    except (OSError, subprocess.SubprocessError):
        pass
    skip = {".git", ".venv", "node_modules", "__pycache__"}
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        found += [Path(dirpath) / f for f in filenames if f.endswith(".md")]
    return sorted(found)


def extract_links(text: str) -> List[Tuple[int, str]]:
    """(line number, target) for every link outside code."""
    links: List[Tuple[int, str]] = []
    in_fence: Optional[str] = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if in_fence is None:
                in_fence = marker
            elif marker == in_fence:
                in_fence = None
            continue
        if in_fence is not None:
            continue
        ref = _REF_DEF_RE.match(line)
        if ref:
            links.append((lineno, ref.group(1)))
            continue
        clean = _INLINE_CODE_RE.sub("", line)
        for link in _INLINE_LINK_RE.finditer(clean):
            links.append((lineno, link.group(1)))
    return links


def _abs(path: Path) -> Path:
    # Not Path.resolve(): on Windows/macOS that returns the on-disk letter case and would hide case errors.
    return Path(os.path.normpath(os.path.abspath(str(path))))


def _exists_exact_case(path: Path, root: Path) -> bool:
    """True if `path` exists and every component below `root` matches the on-disk name exactly."""
    path, root = _abs(path), _abs(root)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.exists()  # outside the repo: no case check possible
    current = root
    for part in rel.parts:
        try:
            entries = os.listdir(current)
        except (NotADirectoryError, FileNotFoundError):
            return False
        if part not in entries:
            return False
        current = current / part
    return True


def check_file(md: Path, root: Path) -> List[str]:
    problems = []
    text = md.read_text(encoding="utf-8")
    for lineno, target in extract_links(text):
        if _SCHEME_RE.match(target) or target.startswith("//"):
            continue  # http:, https:, mailto:, ...
        path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not path_part:
            continue  # pure anchor
        base = root if path_part.startswith("/") else md.parent
        if not _exists_exact_case(base / path_part.lstrip("/"), root):
            problems.append(f"{md.relative_to(root).as_posix()}:{lineno}: broken link -> {target}")
    return problems


def check_all(root: Path = ROOT) -> List[str]:
    problems: List[str] = []
    for md in markdown_files(root):
        problems += check_file(md, root)
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check relative links in markdown files (case-sensitive).")
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args(argv)
    root = _abs(Path(args.root))
    files = markdown_files(root)
    problems = check_all(root)
    for line in problems:
        print(line)
    print(f"checked {len(files)} markdown files: {len(problems)} broken link(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
