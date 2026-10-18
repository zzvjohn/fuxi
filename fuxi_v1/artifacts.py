"""Immutable run directories: publish once, hashes verified on every consumption."""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import time
import uuid
from .contracts import canonical, canonical_chunks, digest, load_json, require, keys


def code_manifest():
    root = Path(__file__).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*.py"))}


def safe_name(name):
    require(isinstance(name, str) and "\\" not in name, "invalid artifact path")
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and all(x not in ("", ".", "..") for x in name.split("/")),
            "artifact path must remain inside bundle")
    require(all(":" not in part for part in p.parts), "invalid artifact path drive")
    return p


def publish(output_root, files, provenance, *, run_id=None):
    run_id = run_id or "run_" + uuid.uuid4().hex
    require(run_id.startswith("run_") and run_id.replace("_", "").isalnum(), "invalid run id")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    require(not destination.exists(), "run identity already exists; retries need new IDs")
    pending = root / (".pending_" + run_id)
    pending.mkdir(exist_ok=False)
    hashes = {}
    for name, value in files.items():
        safe_name(name)
        require(name != "manifest.json", "manifest is reserved")
        path = pending / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = hashlib.sha256()
        with path.open("xb") as stream:
            chunks = (value,) if isinstance(value, bytes) else canonical_chunks(value)
            for chunk in chunks:
                stream.write(chunk)
                content.update(chunk)
            if not isinstance(value, bytes):
                stream.write(b"\n")
                content.update(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = content.hexdigest()
    manifest = dict(schema_version="fuxi.run.1", run_id=run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version(), files=hashes,
                    provenance=provenance, complete=True)
    manifest["manifest_hash"] = digest(manifest)
    with (pending / "manifest.json").open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Publication is the last operation. Interrupted pending directories are never consumed.
    _rename_into_place(pending, destination)
    return destination


def _rename_into_place(pending, destination):
    """Directory rename into place; bounded retry on Windows transient sharing conflicts.

    P-20261006-004: M2 A/B niche_mix arm hit [WinError 5] here after many successful
    publishes in the same root (antivirus/indexer briefly holding handles on freshly
    written bundle files). Retries do not change atomicity: a failed rename leaves the
    pending directory incomplete and never consumed. Non-Windows keeps single attempt.
    """
    attempts = 5 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            pending.rename(destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.25 * (attempt + 1))


def verify(path):
    root = Path(path)
    require(root.name.startswith("run_") and not root.is_symlink(), "not a published run directory")
    manifest = load_json(root / "manifest.json")
    keys(manifest, ("schema_version", "run_id", "created_at", "python_version", "files",
                    "provenance", "complete", "manifest_hash"))
    require(manifest["schema_version"] == "fuxi.run.1" and manifest["complete"] is True,
            "incomplete/unsupported manifest")
    require(manifest["run_id"] == root.name, "directory/manifest run id mismatch")
    require(digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"],
            "manifest hash mismatch")
    discovered = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}
    require(discovered == set(manifest["files"]) | {"manifest.json"}, "artifact file set mismatch")
    require(not any(p.is_symlink() for p in root.rglob("*")), "artifact symlinks forbidden")
    for name, expected in manifest["files"].items():
        safe_name(name)
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "artifact hash mismatch: " + name)
    return manifest
