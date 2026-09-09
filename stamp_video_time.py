#!/usr/bin/env python
"""Restore the wall-clock stamp on rendered clips so they can be time-synced.

Why this exists
---------------
Every phone/GoPro clip carries an absolute ``creation_time`` in its container
metadata. Downstream tools that line a video up against sensor data (the rippl
wave viewer, for one) never ask the video where it belongs on a timeline — they
work it out from that stamp against the log's clock.

``cv2.VideoWriter`` writes no container metadata at all. So a pose overlay plays
back perfectly but carries no ``creation_time``, and a consumer that needs one
cannot place it. In rippl's case the clip is *silently skipped*: no error, just an
empty video panel and no hint as to why.

This copies each source clip's metadata onto its rendered counterpart using
ffmpeg's ``-c copy``, i.e. a remux. The encoded frames are passed through
byte-for-byte — verified by comparing the video stream's MD5 before and after —
so there is no re-encode, no quality loss, and it takes about a second per clip
rather than re-running the render.

``batch_pose.py`` calls ``stamp()`` itself as each clip finishes, so freshly
rendered output is already stamped. This CLI exists to backfill clips rendered
before that was wired up.

Usage
-----
    python stamp_video_time.py                      # dry run over the default folder
    python stamp_video_time.py --apply              # actually stamp them
    python stamp_video_time.py some/other/out --apply

Pairing is by filename: ``<stem><suffix>.MOV`` in the target folder pairs with
``<stem>.MOV`` among the originals (``--suffix``, default ``_pose``; originals
default to the target's parent folder, which is how ``batch_pose.py`` lays things
out).

Caveat
------
The stamp means "when recording finished", so a consumer derives the start as
``creation_time - duration``. That stays correct only while the render preserves
the original duration — which ``batch_pose.py`` does, writing every
``FRAME_STEP``-th frame at ``FPS/FRAME_STEP``. Trim or pad a clip and its computed
start moves by the same amount.

Requires ffmpeg/ffprobe on PATH (``conda install -c conda-forge ffmpeg``).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

# Same layout batch_pose.py writes to, so the common case needs no arguments.
SRC_DIR = "dataLog00188"
DEFAULT_TARGET = os.path.join(SRC_DIR, "pose_out")

# Containers we treat as video; skip rippl's transcode caches if they're present.
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
_CACHE_MARKER = ".rippl-"


def _tool(name):
    """Path to an ffmpeg-family binary — PATH first, then this interpreter's env."""
    found = shutil.which(name)
    if found:
        return found
    local = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.exists(local):
        return local
    raise SystemExit(
        f"{name} not found. Install it into the active env with:\n"
        f"  conda install -c conda-forge ffmpeg -y")


def creation_time(path):
    """The clip's container ``creation_time`` string, or None if it carries none."""
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "quiet", "-print_format", "json",
         "-show_entries", "format_tags=creation_time", str(path)],
        capture_output=True, text=True).stdout
    tags = (json.loads(out or "{}")).get("format", {}).get("tags", {})
    return tags.get("creation_time")


def needs_stamp(path):
    """True if ``path`` has no ``creation_time``, so it can't be placed on a timeline."""
    return creation_time(path) is None


def stamp(src, dst):
    """Copy ``src``'s container metadata onto ``dst``, in place, without re-encoding.

    Streams come from the render (input 0), global metadata from the original
    (input 1). Writes to a temp file and renames over the target, so an
    interruption leaves the existing clip intact rather than truncated.
    Returns True on success.
    """
    src, dst = str(src), str(dst)
    if creation_time(src) is None:
        print(f"  !! {os.path.basename(src)} has no creation_time to copy", flush=True)
        return False
    tmp = dst + ".stamping" + os.path.splitext(dst)[1]
    proc = subprocess.run(
        [_tool("ffmpeg"), "-y", "-v", "error", "-i", dst, "-i", src,
         "-map", "0", "-map_metadata", "1", "-c", "copy", tmp],
        capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(tmp):
        print(f"  !! could not stamp {os.path.basename(dst)}: "
              f"{proc.stderr.strip()[-200:]}", flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, dst)          # atomic: the clip is never left half-written
    return True


def _clips(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS and _CACHE_MARKER not in f)


def _find_original(stem, originals, like_ext):
    for ext in (like_ext, ".MOV", ".mov", ".mp4", ".MP4"):
        cand = os.path.join(originals, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("target", nargs="?", default=DEFAULT_TARGET,
                   help=f"folder of rendered clips to stamp (default {DEFAULT_TARGET})")
    p.add_argument("--originals", default=None,
                   help="folder holding the source clips (default: target's parent)")
    p.add_argument("--suffix", default="_pose",
                   help="suffix the render added to each stem (default '_pose')")
    p.add_argument("--apply", action="store_true",
                   help="rewrite the files in place; without this it's a dry run")
    args = p.parse_args()

    target = args.target
    originals = args.originals or (os.path.dirname(target.rstrip("/")) or ".")
    if not os.path.isdir(target):
        print(f"Not a folder: {target}", file=sys.stderr)
        sys.exit(1)

    clips = _clips(target)
    if not clips:
        print(f"No clips in {target}.")
        return

    todo, already = [], 0
    for clip in clips:
        if not needs_stamp(clip):
            already += 1
            continue
        stem, ext = os.path.splitext(os.path.basename(clip))
        if args.suffix and stem.endswith(args.suffix):
            stem = stem[:-len(args.suffix)]
        src = _find_original(stem, originals, ext)
        if src is None:
            print(f"  ?? {os.path.basename(clip)}: no original named "
                  f"'{stem}' in {originals}")
            continue
        todo.append((src, clip))

    print(f"{len(clips)} clip(s) in {target}: {already} already stamped, "
          f"{len(todo)} to stamp.")
    for src, dst in todo:
        ct = creation_time(src)
        if args.apply:
            ok = stamp(src, dst)
            if ok:
                print(f"  stamped {os.path.basename(dst)}  <- "
                      f"{os.path.basename(src)}  ({ct})")
        else:
            print(f"  would stamp {os.path.basename(dst)}  <- "
                  f"{os.path.basename(src)}  ({ct})")
    if todo and not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to stamp them.")


if __name__ == "__main__":
    main()
