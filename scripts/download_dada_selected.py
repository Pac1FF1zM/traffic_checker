"""Download only selected DADA-2000 clips from the official split ZIP."""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


DRIVE_VOLUMES = (
    ("DADA2000.z01", "1q2wU6zZJekNEjgCeJGb9wS2EQwICswLV", 21_367_462_297),
    ("DADA2000.z02", "1LWWBFqYKiiE_pEEo090G-OjrG4oQAsOL", 21_367_462_297),
    ("DADA2000.z03", "1AydgwavRikuXra_e65eOOVpPTvdafP8u", 21_367_462_297),
    ("DADA2000.z04", "1HUV5Qr2KPXDmxpNeVsCZT9gmHIp4RAUP", 21_367_462_297),
    ("DADA2000.z05", "1GDCuY157lpmSqgCiV4FPSMItkXvEgbJt", 21_367_462_297),
    ("DADA2000.zip", "1QrrSZECLBpBpzhLGa7Lw0YiQ1YRR3P7S", 18_518_305_241),
)
CENTRAL_DIRECTORY_DISK = 5
CENTRAL_DIRECTORY_OFFSET = 18_035_845_548
CENTRAL_DIRECTORY_SIZE = 482_459_595
CENTRAL_DIRECTORY_ENTRIES = 3_909_940
USER_AGENT = "traffic-checker-dada-downloader/1.0"


@dataclass
class FrameEntry:
    name: str
    disk: int
    offset: int
    compressed_size: int
    uncompressed_size: int
    method: int
    flags: int
    crc32: int


@dataclass
class ClipGroup:
    clip: str
    start_disk: int
    start_offset: int
    end_disk: int
    end_offset: int
    entries: list[FrameEntry]


def read_selected(path: Path) -> list[str]:
    clips = [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    invalid = [clip for clip in clips if len(clip.split("/")) != 2]
    if invalid:
        raise ValueError(f"Invalid clip ids in {path}: {invalid[:5]}")
    if len(clips) != len(set(clips)):
        raise ValueError(f"Duplicate clip ids in {path}")
    return sorted(clips)


def lines_sha256(items: list[str]) -> str:
    return hashlib.sha256("".join(f"{item}\n" for item in items).encode()).hexdigest()


def drive_url(disk: int) -> str:
    file_id = DRIVE_VOLUMES[disk][1]
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


class HTTPRangeReader:
    """Sequential, retrying reader for one inclusive HTTP byte range."""

    def __init__(self, url: str, start: int, end_exclusive: int, label: str):
        if not 0 <= start < end_exclusive:
            raise ValueError(f"Invalid range {start}:{end_exclusive} for {label}")
        self.url = url
        self.position = start
        self.end = end_exclusive
        self.label = label
        self.response = None
        self.retries = 0

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
            self.response = None

    def _open(self) -> None:
        while True:
            try:
                # Google Drive can cache a full-file (HTTP 200) response after many
                # requests and then ignore Range for the same URL.  A unique,
                # otherwise ignored query value keeps every byte-range request
                # independent without changing the source object.
                separator = "&" if "?" in self.url else "?"
                request_url = (
                    f"{self.url}{separator}range_request="
                    f"{self.position}-{time.time_ns()}"
                )
                request = urllib.request.Request(
                    request_url,
                    headers={
                        "Range": f"bytes={self.position}-{self.end - 1}",
                        "User-Agent": USER_AGENT,
                        "Accept-Encoding": "identity",
                        "Cache-Control": "no-cache",
                    },
                )
                response = urllib.request.urlopen(request, timeout=90)
                content_range = response.headers.get("Content-Range", "")
                if response.status == 200 and not content_range:
                    response.close()
                    self.retries += 1
                    if self.retries > 12:
                        raise RuntimeError(
                            f"Google Drive kept throttling range requests for {self.label}"
                        )
                    cooldowns = (15, 30, 60, 120, 300, 600)
                    delay = cooldowns[min(self.retries - 1, len(cooldowns) - 1)]
                    print(
                        f"Google Drive temporarily throttled {self.label}; "
                        f"cooling down for {delay}s "
                        f"(attempt {self.retries}/12)",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                if response.status != 206 or not content_range.startswith(
                    f"bytes {self.position}-"
                ):
                    response.close()
                    raise RuntimeError(
                        f"Server ignored byte range for {self.label}: "
                        f"HTTP {response.status}, Content-Range={content_range!r}"
                    )
                self.response = response
                return
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                self.retries += 1
                if self.retries > 8:
                    raise RuntimeError(
                        f"Could not read {self.label} after 8 retries"
                    ) from error
                delay = min(30, 2**self.retries)
                print(
                    f"Network retry {self.retries}/8 for {self.label} in {delay}s: {error}",
                    flush=True,
                )
                time.sleep(delay)

    def read(self, size: int) -> bytes:
        if size <= 0 or self.position >= self.end:
            return b""
        wanted = min(size, self.end - self.position)
        chunks: list[bytes] = []
        remaining = wanted
        while remaining:
            if self.response is None:
                self._open()
            try:
                chunk = self.response.read(min(remaining, 1024 * 1024))
            except (OSError, urllib.error.URLError) as error:
                self.close()
                self.retries += 1
                if self.retries > 8:
                    raise RuntimeError(
                        f"Connection repeatedly failed for {self.label}"
                    ) from error
                delay = min(30, 2**self.retries)
                print(
                    f"Network retry {self.retries}/8 for {self.label} in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                continue
            if not chunk:
                self.close()
                if self.position < self.end:
                    self.retries += 1
                    if self.retries > 8:
                        raise RuntimeError(
                            f"Unexpected EOF for {self.label} at byte {self.position}"
                        )
                    continue
                break
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
            self.retries = 0
        return b"".join(chunks)

    def read_exact(self, size: int) -> bytes:
        data = self.read(size)
        if len(data) != size:
            raise EOFError(
                f"Expected {size} bytes from {self.label}, received {len(data)}"
            )
        return data

    def __enter__(self) -> "HTTPRangeReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def parse_zip64_extra(
    extra: bytes, uncomp: int, comp: int, offset: int, disk: int
) -> tuple[int, int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        field = extra[cursor : cursor + size]
        cursor += size
        if field_id != 0x0001:
            continue
        position = 0
        if uncomp == 0xFFFFFFFF:
            uncomp = struct.unpack_from("<Q", field, position)[0]
            position += 8
        if comp == 0xFFFFFFFF:
            comp = struct.unpack_from("<Q", field, position)[0]
            position += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", field, position)[0]
            position += 8
        if disk == 0xFFFF:
            disk = struct.unpack_from("<I", field, position)[0]
        break
    if 0xFFFFFFFF in (uncomp, comp, offset) or disk == 0xFFFF:
        raise ValueError("Incomplete ZIP64 central-directory metadata")
    return uncomp, comp, offset, disk


def clip_from_archive_name(name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(name).parts
    if len(parts) != 5 or parts[0] != "DADA2000" or parts[3] != "images":
        return None
    if not parts[4].lower().endswith(".png"):
        return None
    return f"{int(parts[1])}/{int(parts[2]):03d}", parts[4]


def load_cached_index(path: Path, selected_hash: str) -> list[ClipGroup] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != 1 or payload.get("selected_sha256") != selected_hash:
            return None
        return [
            ClipGroup(
                clip=item["clip"],
                start_disk=item["start_disk"],
                start_offset=item["start_offset"],
                end_disk=item["end_disk"],
                end_offset=item["end_offset"],
                entries=[FrameEntry(**entry) for entry in item["entries"]],
            )
            for item in payload["groups"]
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_index(path: Path, selected_hash: str, groups: list[ClipGroup]) -> None:
    payload = {
        "format": 1,
        "selected_sha256": selected_hash,
        "official_drive_folder": "1l1_xOMWfs2eSoh0771ZJOcS2tcKwhh-C",
        "groups": [asdict(group) for group in groups],
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, path)


def build_remote_index(selected: set[str]) -> list[ClipGroup]:
    reader = HTTPRangeReader(
        drive_url(CENTRAL_DIRECTORY_DISK),
        CENTRAL_DIRECTORY_OFFSET,
        CENTRAL_DIRECTORY_OFFSET + CENTRAL_DIRECTORY_SIZE,
        "official DADA-2000 ZIP index",
    )
    groups: dict[str, ClipGroup] = {}
    active_clip: str | None = None
    last_report = 0
    with reader:
        for entry_number in range(CENTRAL_DIRECTORY_ENTRIES):
            header = reader.read_exact(46)
            values = struct.unpack("<4s6H3I5H2I", header)
            if values[0] != b"PK\x01\x02":
                raise ValueError(
                    f"Invalid central-directory record at entry {entry_number}"
                )
            flags, method = values[3], values[4]
            crc32, compressed, uncompressed = values[7], values[8], values[9]
            name_length, extra_length, comment_length = (
                values[10],
                values[11],
                values[12],
            )
            disk, offset = values[13], values[16]
            name_bytes = reader.read_exact(name_length)
            extra = reader.read_exact(extra_length)
            if comment_length:
                reader.read_exact(comment_length)
            uncompressed, compressed, offset, disk = parse_zip64_extra(
                extra, uncompressed, compressed, offset, disk
            )
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding)
            parsed = clip_from_archive_name(name)
            clip = parsed[0] if parsed and parsed[0] in selected else None

            if active_clip is not None and clip != active_clip:
                group = groups[active_clip]
                group.end_disk = disk
                group.end_offset = offset
                active_clip = None

            if clip is not None:
                if clip not in groups:
                    groups[clip] = ClipGroup(clip, disk, offset, -1, -1, [])
                group = groups[clip]
                if group.end_disk >= 0:
                    raise ValueError(
                        f"Archive entries for {clip} are not contiguous"
                    )
                group.entries.append(
                    FrameEntry(
                        parsed[1],
                        disk,
                        offset,
                        compressed,
                        uncompressed,
                        method,
                        flags,
                        crc32,
                    )
                )
                active_clip = clip

            if entry_number - last_report >= 100_000:
                percent = 100 * entry_number / CENTRAL_DIRECTORY_ENTRIES
                print(
                    f"Index: {entry_number:,}/{CENTRAL_DIRECTORY_ENTRIES:,} ({percent:.1f}%)",
                    flush=True,
                )
                last_report = entry_number

    if active_clip is not None:
        group = groups[active_clip]
        group.end_disk = CENTRAL_DIRECTORY_DISK
        group.end_offset = CENTRAL_DIRECTORY_OFFSET
    missing = sorted(selected.difference(groups))
    if missing:
        raise RuntimeError(
            f"Official archive is missing {len(missing)} selected clips: {missing[:10]}"
        )
    result = sorted(groups.values(), key=lambda group: (group.start_disk, group.start_offset))
    print(
        f"Index complete: {len(result)} clips, "
        f"{sum(len(group.entries) for group in result):,} frames"
    )
    return result


def download_span(group: ClipGroup) -> tuple[bytes, dict[int, int]]:
    chunks: list[bytes] = []
    disk_bases: dict[int, int] = {}
    total = 0
    for disk in range(group.start_disk, group.end_disk + 1):
        start = group.start_offset if disk == group.start_disk else 0
        end = group.end_offset if disk == group.end_disk else DRIVE_VOLUMES[disk][2]
        if end <= start:
            continue
        disk_bases[disk] = total - start
        label = f"{group.clip} from {DRIVE_VOLUMES[disk][0]}"
        with HTTPRangeReader(drive_url(disk), start, end, label) as reader:
            chunk = reader.read_exact(end - start)
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), disk_bases


def decode_entry(blob: bytes, disk_bases: dict[int, int], entry: FrameEntry) -> bytes:
    position = disk_bases[entry.disk] + entry.offset
    header = blob[position : position + 30]
    if len(header) != 30:
        raise ValueError(f"Truncated local header for {entry.name}")
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise ValueError(f"Invalid local ZIP header for {entry.name}")
    method, name_length, extra_length = values[3], values[9], values[10]
    if method != entry.method:
        raise ValueError(f"Compression method mismatch for {entry.name}")
    data_start = position + 30 + name_length + extra_length
    compressed = blob[data_start : data_start + entry.compressed_size]
    if len(compressed) != entry.compressed_size:
        raise ValueError(f"Truncated frame data for {entry.name}")
    if entry.flags & 0x1:
        raise ValueError(f"Encrypted ZIP entry is unsupported: {entry.name}")
    if entry.method == 0:
        data = compressed
    elif entry.method == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"Unsupported ZIP method {entry.method} for {entry.name}"
        )
    if len(data) != entry.uncompressed_size:
        raise ValueError(f"Size check failed for {entry.name}")
    if binascii.crc32(data) & 0xFFFFFFFF != entry.crc32:
        raise ValueError(f"CRC check failed for {entry.name}")
    return data


def resize_frame(data: bytes, short_side: int) -> bytes:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode a PNG frame")
    height, width = image.shape[:2]
    current_short = min(height, width)
    if current_short > short_side:
        scale = short_side / current_short
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    # The official archive already stores JPEG-encoded frames under .png names.
    # Keep those names because Simple-TAD indexes .png, while OpenCV correctly
    # detects the encoded content. JPEG avoids expanding the compact source into
    # much larger PNGs after resizing.
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not ok:
        raise ValueError("OpenCV could not encode a frame")
    return encoded.tobytes()


def zip_is_complete(path: Path, expected_frames: int) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                name for name in archive.namelist() if name.lower().endswith(".png")
            ]
            return len(names) == expected_frames and len(names) == len(set(names))
    except (OSError, zipfile.BadZipFile):
        return False


def write_clip(
    data_root: Path, group: ClipGroup, short_side: int
) -> tuple[int, int, bool]:
    category, clip = group.clip.split("/")
    destination = data_root / "frames" / category / clip / "images.zip"
    if zip_is_complete(destination, len(group.entries)):
        return len(group.entries), destination.stat().st_size, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.partial")
    temporary.unlink(missing_ok=True)
    blob, disk_bases = download_span(group)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for entry in group.entries:
            archive.writestr(
                entry.name,
                resize_frame(decode_entry(blob, disk_bases, entry), short_side),
            )
    if not zip_is_complete(temporary, len(group.entries)):
        raise RuntimeError(f"Created archive failed validation: {temporary}")
    os.replace(temporary, destination)
    return len(group.entries), destination.stat().st_size, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument(
        "--limit-clips", type=int, help="Download only the first N clips for a smoke test"
    )
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    if args.short_side < 224:
        raise ValueError(
            "--short-side must be at least 224 for the VideoMAE input crop"
        )
    if args.limit_clips is not None and args.limit_clips < 1:
        raise ValueError("--limit-clips must be positive")

    data_root = args.data_root.expanduser().resolve()
    split_dir = data_root / "DADA2K_my_split"
    selected_path = split_dir / "selected_dataset_clips.txt"
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Run prepare_dada_half_splits.py first: {selected_path}"
        )
    selected = read_selected(selected_path)
    selected_hash = lines_sha256(selected)
    index_path = split_dir / "official_drive_selected_index.json"
    groups = (
        None
        if args.rebuild_index
        else load_cached_index(index_path, selected_hash)
    )
    if groups is None:
        print(
            "Reading the official ZIP index (about 460 MiB; no full archive is downloaded)..."
        )
        groups = build_remote_index(set(selected))
        save_index(index_path, selected_hash, groups)
        print(f"Saved reusable index: {index_path}")
    else:
        print(f"Using cached official ZIP index: {index_path}")

    work = groups[: args.limit_clips] if args.limit_clips else groups
    minimum_free = 8 * 1024**3
    total_frames = 0
    total_bytes = 0
    skipped = 0
    started = time.monotonic()
    for number, group in enumerate(work, 1):
        if shutil.disk_usage(data_root).free < minimum_free:
            raise RuntimeError("Less than 8 GiB free; stopping before the next clip")
        frames, size, existed = write_clip(data_root, group, args.short_side)
        total_frames += frames
        total_bytes += size
        skipped += int(existed)
        elapsed = max(1, time.monotonic() - started)
        status = "already complete" if existed else "downloaded"
        print(
            f"[{number}/{len(work)}] {group.clip}: {status}, {frames} frames, "
            f"{size / 1024**2:.1f} MiB ({number / elapsed * 60:.1f} clips/min)",
            flush=True,
        )

    result = {
        "selected_sha256": selected_hash,
        "selected_clips": len(selected),
        "processed_clips": len(work),
        "frames": total_frames,
        "output_bytes": total_bytes,
        "short_side": args.short_side,
        "completed_archives_reused": skipped,
        "official_drive_folder": "1l1_xOMWfs2eSoh0771ZJOcS2tcKwhh-C",
    }
    manifest_path = split_dir / "frame_download_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved download manifest: {manifest_path}")


if __name__ == "__main__":
    main()
