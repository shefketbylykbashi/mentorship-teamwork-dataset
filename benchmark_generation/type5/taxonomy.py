"""Deterministic file-path taxonomy for Type 5 empirical skills."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Iterable, List, Mapping, Sequence


EXTENSION_LANGUAGE = {
    ".py": "Python",
    ".pyx": "Cython",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".c": "C",
    ".h": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".cs": "CSharp",
    ".rs": "Rust",
    ".go": "Go",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".sql": "SQL",
    ".sh": "Shell",
    ".ps1": "PowerShell",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "CSS",
    ".sass": "CSS",
    ".vue": "Vue",
    ".r": "R",
}

EXCLUDED_PARTS = {
    "node_modules",
    "vendor",
    "third_party",
    "dist",
    "build",
    "target",
    "generated",
    "coverage",
    "cache",
    "__pycache__",
}

GENERIC_SUBSYSTEM_PARTS = {
    "src",
    "source",
    "lib",
    "libs",
    "packages",
    "modules",
    "components",
    "app",
    "apps",
    "pkg",
    "internal",
}

# Small documented root-wrapper table from the Type 5 specification examples.
REPOSITORY_ROOT_PARTS = {
    "django",
    "spring-boot-project",
}

CI_PARTS = {".github", "github", "workflows", "ci", "cd", "deploy", "deployment", "docker"}

LOCK_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "gemfile.lock",
    "composer.lock",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".jar",
    ".war",
    ".class",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
}

DOC_PARTS = {"docs", "doc", "documentation", "examples"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}


def normalize_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/").strip("/"))


def should_exclude_path(path: str) -> bool:
    p = normalize_path(path)
    parts = [part.lower() for part in p.parts]
    name = p.name.lower()
    suffix = p.suffix.lower()
    if any(part in EXCLUDED_PARTS for part in parts):
        return True
    if name in LOCK_FILENAMES:
        return True
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return True
    if suffix in BINARY_EXTENSIONS:
        return True
    return False


def is_docs_only_path(path: str) -> bool:
    p = normalize_path(path)
    parts = [part.lower() for part in p.parts]
    return p.suffix.lower() in DOC_EXTENSIONS or any(part in DOC_PARTS for part in parts)


def language_for_path(path: str) -> str | None:
    p = normalize_path(path)
    name = p.name
    suffix = p.suffix.lower()
    parts = [part.lower() for part in p.parts]
    if name == "Dockerfile" or name.startswith("Dockerfile."):
        return "DevOps"
    if suffix in {".yml", ".yaml"} and any(part in CI_PARTS for part in parts):
        return "DevOps"
    return EXTENSION_LANGUAGE.get(suffix)


def subsystem_for_path(path: str) -> str:
    p = normalize_path(path)
    for part in p.parts[:-1]:
        lower = part.lower()
        if lower in GENERIC_SUBSYSTEM_PARTS or lower in REPOSITORY_ROOT_PARTS:
            continue
        if lower.startswith("."):
            continue
        return _clean_subsystem(part)
    return "Core"


def skill_token_for_path(path: str) -> str | None:
    if should_exclude_path(path):
        return None
    language = language_for_path(path)
    if language is None:
        return None
    return f"{language}::{subsystem_for_path(path)}"


def parent_language_token(skill_token: str) -> str:
    language = skill_token.split("::", 1)[0]
    return f"{language}::General"


def is_bot_identity(identity: str | None, patterns: Sequence[str]) -> bool:
    if not identity:
        return False
    normalized = identity.lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def compile_bot_patterns(config: Mapping[str, object]) -> List[str]:
    return [str(pattern) for pattern in config.get("bot_patterns", [])]


def _clean_subsystem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned or "Core"
