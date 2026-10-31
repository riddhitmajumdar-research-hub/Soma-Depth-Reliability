"""Check that documentation links and audit_fixes path strings resolve.

Two checks, both run from anywhere (the repo root is found from this file):

1. Every relative markdown link in docs/methodology/*.md, Source/README.md,
   audit_fixes/README.md and audit_fixes/**/README.md must point to a file or
   directory that exists. A #fragment on a .md target must match a heading.
2. Every string of the form audit_fixes/... in a .py or .sh file under
   audit_fixes/ or Source/ must resolve. If the string has a wildcard or a
   format placeholder (*, {, %), only the directory before it is checked.

Paths built at run time (os.path.join with variables) are not checked, and
neither are paths on cloud branches that are not in this checkout. Uses only
the standard library. Exit status is 1 if anything fails.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
PATH_RE = re.compile(r"audit_fixes/[A-Za-z0-9_./*{}%\-]*")
SKIP_DIRS = {"__pycache__", "seed_weights", "controls", "confusion_matrices"}


def slug(heading):
    """GitHub-style anchor for a heading."""
    h = heading.strip().lower()
    h = re.sub(r"[`*_]", "", h)
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(md_path):
    out = set()
    fence = False
    for line in md_path.read_text(errors="replace").splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("#"):
            out.add(slug(line.lstrip("#")))
    return out


def doc_files():
    files = sorted((ROOT / "docs" / "methodology").glob("*.md"))
    files += [ROOT / "Source" / "README.md", ROOT / "audit_fixes" / "README.md"]
    files += sorted((ROOT / "audit_fixes").rglob("README.md"))
    seen, out = set(), []
    for f in files:
        if f.exists() and f not in seen and not (SKIP_DIRS & set(f.parts)):
            seen.add(f)
            out.append(f)
    return out


def check_links():
    bad, n = [], 0
    for f in doc_files():
        in_fence = False
        for lineno, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if line.startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            for target in LINK_RE.findall(line):
                if re.match(r"[a-z]+:", target):
                    continue
                n += 1
                path_part, _, frag = target.partition("#")
                dest = f if not path_part else (f.parent / path_part)
                if not dest.exists():
                    bad.append((f, lineno, target, "target not found"))
                elif frag and dest.suffix == ".md" and slug(frag) not in anchors(dest):
                    bad.append((f, lineno, target, "no such heading"))
    return n, bad


def check_paths():
    bad, n = [], 0
    files = []
    for base in ("audit_fixes", "Source"):
        for ext in ("*.py", "*.sh"):
            files += [p for p in (ROOT / base).rglob(ext) if not (SKIP_DIRS & set(p.parts))]
    for f in sorted(files):
        for lineno, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for m in PATH_RE.findall(line):
                m = m.rstrip(".,;:)")
                cut = re.search(r"[*{%]", m)
                check = m[: cut.start()].rsplit("/", 1)[0] if cut else m
                if check in ("audit_fixes/", "audit_fixes"):
                    continue
                n += 1
                if not (ROOT / check.rstrip("/")).exists():
                    bad.append((f, lineno, m, "path not found"))
    return n, bad


def main():
    n_links, bad_links = check_links()
    n_paths, bad_paths = check_paths()
    for f, lineno, target, why in bad_links + bad_paths:
        print(f"{f.relative_to(ROOT)}:{lineno}: {target}  ({why})")
    print(f"links checked: {n_links}, failures: {len(bad_links)}")
    print(f"path strings checked: {n_paths}, failures: {len(bad_paths)}")
    return 1 if (bad_links or bad_paths) else 0


if __name__ == "__main__":
    sys.exit(main())
