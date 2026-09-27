"""Fill missing selected DADA-2000 clips from the Hugging Face mirror.

The mirror is a gzip-compressed tar split into 58 transport chunks. Because gzip
is sequential, the compressed stream may need to be read in full, but only RGB
frames for selected clips that are not already complete are written to disk.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_dada_selected import (
    ClipGroup,
    lines_sha256,
    load_cached_index,
    read_selected,
    resize_frame,
    zip_is_complete,
)


HF_REPO = "JeffreyChou/MM-AU"
HF_REVISION = "540cb1277cb70e91a7022abe852decb3ee9adb0a"
HF_FOLDER = "DADA-2000_chunks"
USER_AGENT = "traffic-checker-dada-hf-fallback/1.0"


def list_hf_chunks() -> list[tuple[str, int]]:
    api = (
        f"https://huggingface.co/api/datasets/{HF_REPO}/tree/"
        f"{HF_REVISION}/{HF_FOLDER}?recursive=false&expand=false&limit=1000"
    )
    request = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        records = json.load(response)
    chunks = sorted(
        (record["path"], int(record["size"]))
        for record in records
        if PurePosixPath(record["path"]).name.startswith("DADA2000.part_")
    )
    if len(chunks) != 58:
        raise RuntimeError(f"Expected 58 Hugging Face chunks, found {len(chunks)}")
    return chunks


def hf_url(path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/datasets/{HF_REPO}/resolve/"
        f"{HF_REVISION}/{encoded}"
    )


class HuggingFaceChunkStream(io.RawIOBase):
    """Expose independently hosted split files as one retrying byte stream."""

    def __init__(self, chunks: list[tuple[str, int]]):
        super().__init__()
        self.chunks = chunks
        self.chunk_index = 0
        self.chunk_offset = 0
        self.response = None
        self.retries = 0
        self.total_read = 0
        self.total_size = sum(size for _, size in chunks)
        self.next_report = 512 * 1024**2

    def readable(self) -> bool:
        return True

    def _close_response(self) -> None:
        if self.response is not None:
            self.response.close()
            self.response = None

    def close(self) -> None:
        self._close_response()
        super().close()

    def _open_current(self) -> None:
        path, size = self.chunks[self.chunk_index]
        while True:
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if self.chunk_offset:
                headers["Range"] = f"bytes={self.chunk_offset}-{size - 1}"
            request = urllib.request.Request(hf_url(path), headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=120)
                if self.chunk_offset and response.status != 206:
                    response.close()
                    raise RuntimeError(
                        f"Mirror ignored resume range for {path}: HTTP {response.status}"
                    )
                if not self.chunk_offset and response.status not in (200, 206):
                    response.close()
                    raise RuntimeError(f"Unexpected HTTP {response.status} for {path}")
                self.response = response
                print(
                    f"HF part {self.chunk_index + 1}/{len(self.chunks)}: "
                    f"{PurePosixPath(path).name} from "
                    f"{self.chunk_offset / 1024**2:.1f} MiB",
                    flush=True,
                )
                return
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                self.retries += 1
                if self.retries > 12:
                    raise RuntimeError(f"Could not read mirror chunk {path}") from error
                delay = min(60, 2**self.retries)
                print(
                    f"HF retry {self.retries}/12 for {PurePosixPath(path).name} "
                    f"in {delay}s: {error}",
                    flush=True,
                )
                time.sleep(delay)

    def readinto(self, buffer) -> int:
        if self.chunk_index >= len(self.chunks):
            return 0
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view) and self.chunk_index < len(self.chunks):
            path, size = self.chunks[self.chunk_index]
            if self.response is None:
                self._open_current()
            wanted = min(len(view) - written, size - self.chunk_offset, 1024 * 1024)
            try:
                data = self.response.read(wanted)
            except (OSError, urllib.error.URLError) as error:
                self._close_response()
                self.retries += 1
                if self.retries > 12:
                    raise RuntimeError(f"Connection repeatedly failed for {path}") from error
                delay = min(60, 2**self.retries)
                print(f"HF connection retry in {delay}s", flush=True)
                time.sleep(delay)
                continue
            if not data:
                self._close_response()
                if self.chunk_offset != size:
                    self.retries += 1
                    if self.retries > 12:
                        raise RuntimeError(
                            f"Unexpected EOF in {path}: {self.chunk_offset}/{size}"
                        )
                    continue
                self.chunk_index += 1
                self.chunk_offset = 0
                self.retries = 0
                continue
            view[written : written + len(data)] = data
            written += len(data)
            self.chunk_offset += len(data)
            self.total_read += len(data)
            self.retries = 0
            if self.total_read >= self.next_report:
                print(
                    f"HF compressed stream: {self.total_read / 1024**3:.1f}/"
                    f"{self.total_size / 1024**3:.1f} GiB "
                    f"({100 * self.total_read / self.total_size:.1f}%)",
                    flush=True,
                )
                self.next_report += 512 * 1024**2
        return written


def parse_hf_member(name: str) -> tuple[str | None, str | None]:
    parts = PurePosixPath(name).parts
    for index in range(len(parts) - 3):
        if parts[index : index + 2] != ("DADA2000", "DADA2000"):
            continue
        if index + 3 >= len(parts):
            return None, None
        category, clip = parts[index + 2], parts[index + 3]
        if not category.isdigit() or not clip.isdigit():
            return None, None
        clip_id = f"{int(category)}/{int(clip):03d}"
        if (
            index + 5 < len(parts)
            and parts[index + 4] == "images"
            and parts[index + 5].lower().endswith(".png")
        ):
            return clip_id, parts[index + 5]
        return clip_id, None
    return None, None


def load_groups(data_root: Path) -> tuple[list[str], list[ClipGroup]]:
    split_dir = data_root / "DADA2K_my_split"
    selected = read_selected(split_dir / "selected_dataset_clips.txt")
    selected_hash = lines_sha256(selected)
    groups = load_cached_index(
        split_dir / "official_drive_selected_index.json", selected_hash
    )
    if groups is None:
        raise RuntimeError(
            "Missing reusable official index; run download_dada_selected.py first"
        )
    return selected, groups


def extract_missing(data_root: Path, groups: list[ClipGroup]) -> dict:
    expected = {group.clip: len(group.entries) for group in groups}
    missing: set[str] = set()
    for group in groups:
        category, clip = group.clip.split("/")
        archive = data_root / "frames" / category / clip / "images.zip"
        if not zip_is_complete(archive, expected[group.clip]):
            missing.add(group.clip)
    if not missing:
        return {"missing_before": 0, "downloaded": 0, "frames": 0}
    initial_missing = len(missing)
    print(
        f"Hugging Face fallback: {len(missing)} clips are missing. "
        "Only these clips will be written.",
        flush=True,
    )

    chunks = list_hf_chunks()
    raw_stream = HuggingFaceChunkStream(chunks)
    buffered = io.BufferedReader(raw_stream, buffer_size=1024 * 1024)
    current_archive_clip: str | None = None
    writer: zipfile.ZipFile | None = None
    writer_clip: str | None = None
    temporary: Path | None = None
    destination: Path | None = None
    written_frames = 0
    downloaded_clips = 0
    total_frames = 0

    def finalize_writer() -> None:
        nonlocal writer, writer_clip, temporary, destination
        nonlocal written_frames, downloaded_clips, total_frames
        if writer is None or writer_clip is None or temporary is None or destination is None:
            return
        writer.close()
        writer = None
        if written_frames != expected[writer_clip] or not zip_is_complete(
            temporary, expected[writer_clip]
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Mirror frame count mismatch for {writer_clip}: "
                f"{written_frames}/{expected[writer_clip]}"
            )
        os.replace(temporary, destination)
        missing.remove(writer_clip)
        downloaded_clips += 1
        total_frames += written_frames
        print(
            f"HF completed {writer_clip}: {written_frames} frames; "
            f"{len(missing)} clips remain",
            flush=True,
        )
        writer_clip = None
        temporary = None
        destination = None
        written_frames = 0

    try:
        with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
            for member in archive:
                member_clip, frame_name = parse_hf_member(member.name)
                if (
                    current_archive_clip is not None
                    and member_clip is not None
                    and member_clip != current_archive_clip
                ):
                    finalize_writer()
                    if not missing:
                        break
                if member_clip is not None:
                    current_archive_clip = member_clip
                if (
                    not member.isfile()
                    or frame_name is None
                    or member_clip not in missing
                ):
                    continue
                if writer is None:
                    category, clip = member_clip.split("/")
                    destination = (
                        data_root / "frames" / category / clip / "images.zip"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".zip.partial")
                    temporary.unlink(missing_ok=True)
                    writer = zipfile.ZipFile(
                        temporary,
                        "w",
                        compression=zipfile.ZIP_STORED,
                        allowZip64=True,
                    )
                    writer_clip = member_clip
                    written_frames = 0
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read tar member {member.name}")
                writer.writestr(frame_name, resize_frame(source.read(), 256))
                written_frames += 1
            finalize_writer()
    finally:
        if writer is not None:
            writer.close()
        buffered.close()

    if missing:
        raise RuntimeError(
            f"Hugging Face stream ended with {len(missing)} clips missing: "
            f"{sorted(missing)[:10]}"
        )
    return {
        "missing_before": initial_missing,
        "downloaded": downloaded_clips,
        "frames": total_frames,
        "compressed_bytes_read": raw_stream.total_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    selected, groups = load_groups(data_root)
    result = extract_missing(data_root, groups)
    complete = 0
    output_bytes = 0
    for group in groups:
        category, clip = group.clip.split("/")
        path = data_root / "frames" / category / clip / "images.zip"
        if zip_is_complete(path, len(group.entries)):
            complete += 1
            output_bytes += path.stat().st_size
    result.update(
        {
            "selected_clips": len(selected),
            "complete_clips": complete,
            "output_bytes": output_bytes,
            "source": f"https://huggingface.co/datasets/{HF_REPO}",
            "source_revision": HF_REVISION,
        }
    )
    manifest = data_root / "DADA2K_my_split" / "frame_download_manifest.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved download manifest: {manifest}")


if __name__ == "__main__":
    main()
