"""Acquire and verify the pinned official MSLIB release asset for Type 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from benchmark_generation.core.config import load_config


DEFAULT_CONFIG: Dict[str, Any] = {
    "source_release": {
        "observed_release_tag": "v2.2",
        "release_asset": "MSLIB_v2_2.zip",
        "download_url": "https://github.com/MarioVanhoucke/MSLIB-Multi-Skilled-Resource-Constrained-Project-Library/releases/download/v2.2/MSLIB_v2_2.zip",
        "sha256": "f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c",
    },
    "v2_1_tag_archive_evidence": {
        "filename": "MSLIB_v2.1_tag.zip",
        "sha256": "197406df4991314cc7879a3cb65eda450b855c96fa8118963332a6f23d6e0833",
    },
    "expected_counts": {
        "MSLIB1": 6600,
        "MSLIB2": 9000,
        "MSLIB3": 11880,
        "MSLIB4": 5000,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_or_download_mslib(
    source_dir: str | Path,
    metadata_dir: str | Path,
    config: Mapping[str, Any],
    download: bool = True,
    extract: bool = True,
) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    metadata_dir = Path(metadata_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    release = dict(config["source_release"])
    archive_path = source_dir / str(release["release_asset"])
    if not archive_path.exists():
        if not download:
            raise FileNotFoundError(f"Missing pinned MSLIB release asset: {archive_path}")
        logging.info("Downloading pinned MSLIB release asset: %s", release["download_url"])
        request = urllib.request.Request(
            str(release["download_url"]),
            headers={"User-Agent": "mtp-type6-benchmark-generator"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            archive_path.write_bytes(response.read())

    observed_sha = sha256_file(archive_path)
    expected_sha = str(release["sha256"]).lower()
    if observed_sha.lower() != expected_sha:
        raise RuntimeError(
            f"MSLIB archive checksum mismatch for {archive_path}: "
            f"observed {observed_sha}, expected {expected_sha}"
        )

    archive_report = inspect_release_archive(archive_path, config)
    extract_dir = source_dir / archive_path.stem
    if extract and not (extract_dir / archive_path.stem).exists():
        logging.info("Extracting %s to %s", archive_path, extract_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)

    tag_evidence = inspect_v2_1_tag_archive(source_dir, config)
    provenance = {
        "type6_source_artifact": "MSLIB official release asset MSLIB_v2_2.zip, retrieved from GitHub release v2.2, SHA-256 f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c, with upstream documentation still describing the dataset release as v2.1 (June 2026).",
        "verified_at": utc_now(),
        "archive_path": str(archive_path),
        "extract_dir": str(extract_dir),
        "release": release,
        "observed_sha256": observed_sha,
        "archive_report": archive_report,
        "v2_1_tag_archive_evidence": tag_evidence,
        "redistribution_note": (
            "The upstream archive is kept in type6_source_data and ignored by git. "
            "Public benchmark provenance should record deterministic download instructions, "
            "checksums, and selected filenames rather than redistributing the archive unless "
            "upstream terms explicitly permit it."
        ),
    }
    provenance_path = metadata_dir / "type6_data_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return provenance


def inspect_release_archive(archive_path: str | Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    archive_path = Path(archive_path)
    expected_counts = dict(config.get("expected_counts", {}))
    required_outer = {
        "MSLIB_v2_2/MSLIB1.zip",
        "MSLIB_v2_2/MSLIB2.zip",
        "MSLIB_v2_2/MSLIB3.zip",
        "MSLIB_v2_2/MSLIB4.zip",
        "MSLIB_v2_2/MSRCPSP_DataFormat.pdf",
        "MSLIB_v2_2/Readme.rtf",
    }
    with zipfile.ZipFile(archive_path) as outer:
        real_names = [name for name in outer.namelist() if not _is_macos_artifact(name)]
        missing_outer = sorted(required_outer - set(real_names))
        has_workbook = "MSLIB_v2_2/MSLIB (Parameters and BKS).xlsx" in set(real_names)
        if missing_outer:
            raise RuntimeError(f"Pinned MSLIB archive missing required files: {missing_outer}")
        readme_text = rtf_plain_text(outer.read("MSLIB_v2_2/Readme.rtf"))
        dataset_information_files: Dict[str, list[str]] = {}
        for info_dataset in ("MSLIB1", "MSLIB2", "MSLIB3", "MSLIB4", "MSLIB5"):
            nested_name = f"MSLIB_v2_2/{info_dataset}.zip"
            if nested_name not in set(real_names):
                continue
            with zipfile.ZipFile(outer.open(nested_name)) as nested:
                dataset_information_files[info_dataset] = [
                    name
                    for name in nested.namelist()
                    if not _is_macos_artifact(name)
                    and ("INFO" in Path(name).name.upper() or "README" in Path(name).name.upper())
                ]

        dataset_counts = {}
        noncanonical_files = {}
        for dataset, expected in expected_counts.items():
            nested_name = f"MSLIB_v2_2/{dataset}.zip"
            with zipfile.ZipFile(outer.open(nested_name)) as nested:
                canonical, noncanonical = canonical_msrcp_entries(nested, dataset)
                ids = {entry["source_instance_id"] for entry in canonical}
                missing_ids = sorted(set(range(1, int(expected) + 1)) - ids)
                extra_ids = sorted(id_value for id_value in ids if id_value > int(expected))
                if len(canonical) != int(expected) or missing_ids or extra_ids:
                    raise RuntimeError(
                        f"{dataset} canonical count mismatch: observed={len(canonical)} "
                        f"expected={expected} missing={missing_ids[:10]} extra={extra_ids[:10]}"
                    )
                dataset_counts[dataset] = {
                    "canonical_msrcp_count": len(canonical),
                    "expected_count": int(expected),
                    "first_id": min(ids),
                    "last_id": max(ids),
                }
                if noncanonical:
                    noncanonical_files[dataset] = noncanonical
    return {
        "contains_mslib1_4": all(dataset in dataset_counts for dataset in expected_counts),
        "contains_msrcpsp_dataformat_document": True,
        "contains_dataset_information_files": any(dataset_information_files.values()),
        "dataset_information_files": dataset_information_files,
        "internal_readme_version_statement": readme_version_statement(readme_text),
        "contains_parameters_and_bks_workbook": has_workbook,
        "canonical_dataset_counts": dataset_counts,
        "noncanonical_msrcp_files_excluded": noncanonical_files,
    }


def inspect_v2_1_tag_archive(source_dir: str | Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = dict(config.get("v2_1_tag_archive_evidence", {}))
    path = Path(source_dir) / str(evidence.get("filename", "MSLIB_v2.1_tag.zip"))
    result = dict(evidence)
    result["path"] = str(path)
    result["present"] = path.exists()
    if not path.exists():
        return result
    result["observed_sha256"] = sha256_file(path)
    result["size_bytes"] = path.stat().st_size
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if not _is_macos_artifact(name)]
        result["contains_data_assets"] = any(name.endswith(".msrcp") or name.endswith(".zip") for name in names)
        result["contents"] = names
    return result


def canonical_msrcp_entries(zf: zipfile.ZipFile, dataset: str) -> tuple[list[Dict[str, Any]], list[str]]:
    pattern = re.compile(rf"^MSLIB_Set{dataset[-1]}_(\d+)\.msrcp$", re.IGNORECASE)
    canonical: list[Dict[str, Any]] = []
    noncanonical: list[str] = []
    for name in zf.namelist():
        if _is_macos_artifact(name) or not name.lower().endswith(".msrcp"):
            continue
        filename = Path(name).name
        match = pattern.match(filename)
        if not match:
            noncanonical.append(name)
            continue
        canonical.append(
            {
                "zip_entry": name,
                "filename": filename,
                "source_instance_id": int(match.group(1)),
            }
        )
    canonical.sort(key=lambda item: int(item["source_instance_id"]))
    return canonical, sorted(noncanonical)


def _is_macos_artifact(name: str) -> bool:
    return "__MACOSX" in name or Path(name).name.startswith("._")


def rtf_plain_text(payload: bytes) -> str:
    """Return enough plain text from a simple RTF file for provenance checks."""

    text = payload.decode("cp1252", errors="replace")

    def hex_replace(match: re.Match[str]) -> str:
        return bytes.fromhex(match.group(1)).decode("cp1252", errors="replace")

    text = re.sub(r"\\'([0-9a-fA-F]{2})", hex_replace, text)
    text = re.sub(r"\\(?:par|line)\b", "\n", text)
    text = re.sub(r"\{\\\*[^{}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"[ \t]+", " ", text)


def readme_version_statement(text: str) -> str:
    match = re.search(r"Last update:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return "Last update: " + match.group(1).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="type6_source_data")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type6.yaml")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config, DEFAULT_CONFIG)
    provenance = verify_or_download_mslib(
        args.source_dir,
        args.metadata_dir,
        config,
        download=not args.no_download,
        extract=not args.no_extract,
    )
    print(json.dumps(provenance["archive_report"], indent=2, sort_keys=True))
    if args.verify_only:
        return


if __name__ == "__main__":
    main()
