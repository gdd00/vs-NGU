#!/usr/bin/env python3
"""Generate a PEP 503 simple index from this repository's GitHub release wheels."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import urllib.request


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch_releases(repository: str, token: str | None) -> list[dict]:
    releases, page = [], 1
    while True:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "vs-ngu-index",
                     **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(request) as response:
            batch = json.load(response)
        if not batch:
            return releases
        releases += batch
        page += 1


def collect(releases: list[dict]) -> dict[str, list[tuple[str, str, str]]]:
    projects: dict[str, list[tuple[str, str, str]]] = {}
    for release in releases:
        if release.get("draft"):
            continue
        for asset in release.get("assets", []):
            name, digest = asset["name"], asset.get("digest") or ""
            if name.endswith(".whl") and digest.startswith("sha256:"):
                projects.setdefault(normalize(name.split("-", 1)[0]), []).append(
                    (name, asset["browser_download_url"], digest.removeprefix("sha256:")))
    return projects


def page(title: str, body: str) -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{html.escape(title)}</title></head><body>\n{body}\n</body></html>\n")


def write(projects: dict[str, list[tuple[str, str, str]]], output: Path) -> None:
    simple = output / "simple"
    simple.mkdir(parents=True, exist_ok=True)
    for project, wheels in projects.items():
        links = [f'<a href="{html.escape(f"{url}#sha256={sha}", quote=True)}">{html.escape(name)}</a><br>'
                 for name, url, sha in sorted(wheels)]
        (simple / project).mkdir(exist_ok=True)
        (simple / project / "index.html").write_text(page(f"Links for {project}", "\n".join(links)), encoding="utf-8")
    (simple / "index.html").write_text(page("Simple index", "\n".join(
        f'<a href="{p}/">{p}</a><br>' for p in sorted(projects))), encoding="utf-8")
    (output / "index.html").write_text(page("vs-NGU", '<a href="simple/">PEP 503 package index</a>'), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("_site"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args()
    if not args.repository:
        raise SystemExit("--repository or GITHUB_REPOSITORY is required")
    write(collect(fetch_releases(args.repository, os.environ.get("GITHUB_TOKEN"))), args.output)


if __name__ == "__main__":
    main()
