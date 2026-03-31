#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "uv.lock"
ENV_PATH = ROOT / "environment.yml"
TARGET_PYTHON = "3.12"

CONDA_PACKAGES = [
    "numpy",
    "matplotlib",
    "scipy",
    "tqdm",
    "statsmodels",
    "pandas",
    "pandas-datareader",
    "seaborn",
    "scikit-learn",
    # Keep pkg_resources available for RISE in the classic notebook stack.
    "setuptools",
    "notebook",
    "rise",
    "dtaidistance",
    "plotly",
    "ipywidgets",
]

PIP_PACKAGES = [
    "opencv-python-headless",
    "pmdarima",
    "prophet",
    "reservoir-computing",
    "tck",
    "yfinance",
]

MARKER_RE = re.compile(r"python_full_version\s*([<>=!]+)\s*'([^']+)'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate the user-facing Conda environment."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if environment.yml does not match the generated content.",
    )
    return parser.parse_args()


def parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def marker_matches(marker: str | None, target_python: str) -> bool:
    if not marker:
        return True

    match = MARKER_RE.fullmatch(marker.strip())
    if not match:
        raise ValueError(f"Unsupported marker in uv.lock: {marker}")

    operator, raw_version = match.groups()
    current = parse_version(target_python)
    target = parse_version(raw_version)

    if operator == "<":
        return current < target
    if operator == "<=":
        return current <= target
    if operator == ">":
        return current > target
    if operator == ">=":
        return current >= target
    if operator == "==":
        return current == target
    if operator == "!=":
        return current != target

    raise ValueError(f"Unsupported operator in marker: {marker}")


def load_lock_data() -> dict:
    with LOCK_PATH.open("rb") as lock_file:
        return tomllib.load(lock_file)


def get_package_record(lock_data: dict, name: str) -> dict:
    matches = [package for package in lock_data["package"] if package["name"] == name]
    if not matches:
        raise KeyError(f"Package {name!r} not found in {LOCK_PATH.name}")
    if len(matches) > 1:
        versions = sorted({package["version"] for package in matches})
        raise ValueError(
            f"Package {name!r} has multiple versions in {LOCK_PATH.name}: {versions}. "
            "Add explicit handling in scripts/sync_user_environment.py."
        )
    return matches[0]


def get_tsa_course_record(lock_data: dict) -> dict:
    for package in lock_data["package"]:
        if package["name"] == "tsa-course":
            return package
    raise KeyError("Package 'tsa-course' not found in uv.lock")


def select_dependency_entries(entries: list[dict], target_python: str) -> list[dict]:
    selected: dict[str, dict] = {}
    for entry in entries:
        if not marker_matches(entry.get("marker"), target_python):
            continue
        selected[entry["name"]] = entry
    return list(selected.values())


def resolve_locked_version(lock_data: dict, dependency: dict, target_python: str) -> str:
    if "version" in dependency and marker_matches(dependency.get("marker"), target_python):
        return dependency["version"]
    return get_package_record(lock_data, dependency["name"])["version"]


def collect_expected_versions(lock_data: dict, target_python: str) -> dict[str, str]:
    tsa_course = get_tsa_course_record(lock_data)
    base_dependencies = select_dependency_entries(tsa_course["dependencies"], target_python)
    notebook_dependencies = select_dependency_entries(
        tsa_course["optional-dependencies"]["notebooks"], target_python
    )

    direct_dependencies = {entry["name"]: entry for entry in base_dependencies + notebook_dependencies}
    expected_names = set(direct_dependencies)
    configured_names = set(CONDA_PACKAGES) | set(PIP_PACKAGES)
    if expected_names != configured_names:
        missing = sorted(expected_names - configured_names)
        extra = sorted(configured_names - expected_names)
        raise ValueError(
            "The configured user environment package split is out of date.\n"
            f"Missing from script: {missing}\n"
            f"Unexpected in script: {extra}"
        )

    return {
        name: resolve_locked_version(lock_data, dependency, target_python)
        for name, dependency in direct_dependencies.items()
    }


def render_environment(expected_versions: dict[str, str]) -> str:
    lines = [
        "# User notebook environment for the course.",
        "name: tsa-course",
        "channels:",
        "  - conda-forge",
        "  - nodefaults",
        "dependencies:",
        f"  - python={TARGET_PYTHON}",
        "  - pip",
    ]

    for package_name in CONDA_PACKAGES:
        lines.append(f"  - {package_name}={expected_versions[package_name]}")

    lines.extend(
        [
            "  - pip:",
        ]
    )
    for package_name in PIP_PACKAGES:
        lines.append(f"      - {package_name}=={expected_versions[package_name]}")
    lines.append("      - -e .")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    expected_versions = collect_expected_versions(load_lock_data(), TARGET_PYTHON)
    rendered = render_environment(expected_versions)

    if args.check:
        current = ENV_PATH.read_text() if ENV_PATH.exists() else ""
        if current != rendered:
            sys.stderr.write(
                "environment.yml is out of sync with uv.lock.\n"
                "Run `uv run --no-sync python scripts/sync_user_environment.py`.\n"
            )
            return 1
        return 0

    ENV_PATH.write_text(rendered)
    print(f"Updated {ENV_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
