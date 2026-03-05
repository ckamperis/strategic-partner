"""
Codebase Statistics Generator for Chapter 4.

Computes lines of code, file counts, and dependency information
for the AI Strategic Partner prototype.

Lines of code (LoC) counts exclude:
  - Blank lines
  - Comment-only lines (lines starting with # after stripping whitespace)
  - Docstring lines are INCLUDED (they are part of the code)

Thesis reference: Section 4.1 — Implementation Overview
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "results" / "codebase_stats.json"

# Directories to exclude from all scans
EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".git", ".pytest_cache",
                ".ruff_cache", "node_modules", ".claude", "chroma_data",
                "embedded"}

# Source directories with descriptions
SOURCE_DIRS = [
    ("picp/", "PICP coordination protocol (core thesis contribution)"),
    ("pillars/knowledge/", "Knowledge Pillar (RAG, hybrid search, ChromaDB)"),
    ("pillars/reasoning/", "Reasoning Pillar (heuristic policy, skills)"),
    ("pillars/simulation/", "Simulation Pillar (Monte Carlo engine)"),
    ("pillars/trust/", "Trust Pillar (scoring, SHAP, audit)"),
    ("orchestrator/", "Central orchestrator (K->R->S->T pipeline)"),
    ("utils/", "Utilities (LLM abstraction layer)"),
    ("data/pipeline/", "Data pipeline (ERP transformer, classifier)"),
    ("config/", "Configuration (Pydantic settings)"),
    ("tests/unit/", "Unit tests"),
    ("tests/integration/", "Integration tests"),
    ("experiments/", "Experiment scripts (Chapter 5)"),
    ("scripts/", "CLI scripts (ingestion, query runner)"),
]

# Known dependency purposes
DEPENDENCY_PURPOSES = {
    "fastapi": "REST API framework",
    "uvicorn": "ASGI server for FastAPI",
    "pydantic": "Data validation and settings",
    "pydantic-settings": "Environment-based configuration",
    "redis": "PICP message bus and distributed locking",
    "chromadb": "Vector database for RAG embeddings",
    "openai": "LLM and embedding API client",
    "rank-bm25": "BM25 lexical search for hybrid retrieval",
    "openpyxl": "Excel file parsing for ERP data",
    "pandas": "DataFrame operations for ERP transformation",
    "numpy": "Vectorized Monte Carlo simulation engine",
    "scipy": "Statistical tests (Wilcoxon, distributions)",
    "structlog": "Structured logging with correlation IDs",
    "python-dotenv": "Environment variable loading from .env",
    "httpx": "Async HTTP client",
    "pyyaml": "YAML skill definition parsing",
    "pytest": "Test framework",
    "pytest-asyncio": "Async test support",
    "pytest-cov": "Test coverage reporting",
    "ruff": "Python linter and formatter",
    "fakeredis": "In-memory Redis for testing",
    "matplotlib": "Plot generation for thesis figures",
    "seaborn": "Statistical visualization",
    "scikit-learn": "ML utilities for experiments",
}


def count_loc(filepath: Path) -> int:
    """Count non-blank, non-comment lines in a Python file."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0

    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        count += 1
    return count


def count_lines_total(filepath: Path) -> int:
    """Count total lines in a file (including blank/comments)."""
    try:
        return len(filepath.read_text(encoding="utf-8").splitlines())
    except (UnicodeDecodeError, OSError):
        return 0


def should_exclude(path: Path) -> bool:
    """Check if a path is inside an excluded directory."""
    return any(part in EXCLUDE_DIRS for part in path.parts)


def find_python_files(directory: Path) -> list[Path]:
    """Find all .py files in a directory, excluding venv/__pycache__."""
    if not directory.exists():
        return []
    return [
        f for f in directory.rglob("*.py")
        if not should_exclude(f)
    ]


def find_yaml_files(directory: Path) -> list[Path]:
    """Find all .yaml files in a directory."""
    if not directory.exists():
        return []
    return [f for f in directory.rglob("*.yaml") if not should_exclude(f)]


def parse_dependencies(pyproject_path: Path) -> list[dict]:
    """Parse dependencies from pyproject.toml."""
    text = pyproject_path.read_text(encoding="utf-8")
    deps = []

    # Parse main dependencies
    in_deps = False
    in_dev = False
    in_exp = False
    current_group = "runtime"

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == 'dependencies = [':
            in_deps = True
            current_group = "runtime"
            continue
        elif stripped.startswith('dev = ['):
            in_deps = True
            current_group = "dev"
            continue
        elif stripped.startswith('experiments = ['):
            in_deps = True
            current_group = "experiments"
            continue

        if in_deps and stripped == ']':
            in_deps = False
            continue

        if in_deps and stripped.startswith('"'):
            # Parse: "package>=version"
            match = re.match(r'"([a-zA-Z0-9_-]+)(?:\[.*?\])?>=([0-9.]+)"', stripped)
            if match:
                name = match.group(1)
                version = match.group(2)
                deps.append({
                    "name": name,
                    "min_version": version,
                    "group": current_group,
                    "purpose": DEPENDENCY_PURPOSES.get(name, ""),
                })

    return deps


def get_installed_version(package_name: str) -> str | None:
    """Get installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def main() -> None:
    print("=" * 70)
    print("Codebase Statistics Generator")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Per-directory stats
    # ------------------------------------------------------------------
    per_directory = []
    total_files = 0
    total_loc = 0
    total_loc_no_tests = 0
    test_files_count = 0
    test_loc = 0

    print(f"\n{'Directory':<30} {'Files':>5} {'LoC':>6}")
    print("-" * 45)

    for dir_rel, description in SOURCE_DIRS:
        dir_path = BASE_DIR / dir_rel
        py_files = find_python_files(dir_path)
        loc = sum(count_loc(f) for f in py_files)

        entry = {
            "directory": dir_rel,
            "python_files": len(py_files),
            "loc": loc,
            "description": description,
        }
        per_directory.append(entry)
        total_files += len(py_files)
        total_loc += loc

        is_test = dir_rel.startswith("tests/")
        if is_test:
            test_files_count += len(py_files)
            test_loc += loc
        else:
            total_loc_no_tests += loc

        print(f"  {dir_rel:<28} {len(py_files):>5} {loc:>6}")

    # Check for root-level files (conftest, __init__ in tests/)
    root_conftest = BASE_DIR / "tests" / "conftest.py"
    if root_conftest.exists():
        loc_conf = count_loc(root_conftest)
        per_directory.append({
            "directory": "tests/",
            "python_files": 1,
            "loc": loc_conf,
            "description": "Test fixtures (conftest.py)",
        })
        total_files += 1
        total_loc += loc_conf
        test_files_count += 1
        test_loc += loc_conf
        print(f"  {'tests/ (conftest)':<28} {1:>5} {loc_conf:>6}")

    # Also count pillars/__init__.py and pillars/base.py if not already
    for extra in ["pillars/__init__.py", "pillars/base.py"]:
        ep = BASE_DIR / extra
        if ep.exists():
            loc_e = count_loc(ep)
            dir_name = str(Path(extra).parent) + "/"
            # Check not already counted
            already = any(d["directory"] == dir_name for d in per_directory
                         if "base" in extra and "base" in d.get("description", ""))
            if not already:
                per_directory.append({
                    "directory": extra,
                    "python_files": 1,
                    "loc": loc_e,
                    "description": "Pillar base class" if "base" in extra else "Pillars package init",
                })
                total_files += 1
                total_loc += loc_e
                total_loc_no_tests += loc_e

    print("-" * 45)
    print(f"  {'TOTAL':<28} {total_files:>5} {total_loc:>6}")
    print(f"  {'(excluding tests)':<28} {'':>5} {total_loc_no_tests:>6}")

    # ------------------------------------------------------------------
    # YAML skill files
    # ------------------------------------------------------------------
    yaml_dir = BASE_DIR / "pillars" / "reasoning" / "skills"
    yaml_files = find_yaml_files(yaml_dir)
    yaml_total_lines = sum(count_lines_total(f) for f in yaml_files)

    print(f"\nYAML skill files: {len(yaml_files)} files, {yaml_total_lines} lines")
    for yf in sorted(yaml_files):
        print(f"  {yf.name}: {count_lines_total(yf)} lines")

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------
    pyproject = BASE_DIR / "pyproject.toml"
    deps = parse_dependencies(pyproject)

    print(f"\nDependencies: {len(deps)} total")
    print(f"  Fetching installed versions...")

    for dep in deps:
        installed = get_installed_version(dep["name"])
        dep["installed_version"] = installed or "not installed"

    runtime_deps = [d for d in deps if d["group"] == "runtime"]
    dev_deps = [d for d in deps if d["group"] == "dev"]
    exp_deps = [d for d in deps if d["group"] == "experiments"]

    print(f"  Runtime: {len(runtime_deps)}, Dev: {len(dev_deps)}, Experiments: {len(exp_deps)}")

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    output = {
        "description": "Codebase statistics for AI Strategic Partner prototype (Chapter 4)",
        "total_python_files": total_files,
        "total_loc": total_loc,
        "total_loc_excluding_tests": total_loc_no_tests,
        "test_loc": test_loc,
        "per_directory": per_directory,
        "dependencies": deps,
        "dependency_counts": {
            "runtime": len(runtime_deps),
            "dev": len(dev_deps),
            "experiments": len(exp_deps),
            "total": len(deps),
        },
        "skill_files": {
            "count": len(yaml_files),
            "total_lines": yaml_total_lines,
            "files": [f.name for f in sorted(yaml_files)],
        },
        "test_files": {
            "count": test_files_count,
            "total_lines": test_loc,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
