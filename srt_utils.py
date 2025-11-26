"""
SRT Parser and Text Cleaner

Utilities for parsing SRT subtitle files and cleaning text for TTS.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""

    index: int
    start_time: str  # "00:00:01,000"
    end_time: str  # "00:00:04,000"
    text: str  # Cleaned text content


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp to seconds."""
    # "00:01:23,456" -> 83.456
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def clean_text(text: str) -> str:
    """Clean subtitle text for TTS.

    - Remove speaker labels like "[Speaker]:" or "(Speaker):"
    - Remove sound effects like [music], (applause)
    - Remove HTML tags
    - Normalize whitespace
    - Remove empty lines
    - Strip invisible control characters (zero-width, bidi, BOM, C0/C1 except tab/newline)
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove speaker labels: [Speaker]: or (Speaker):
    text = re.sub(r"[\[\(][^\]\)]+[\]\)]:\s*", "", text)

    # Remove sound effects: [music], (applause), etc.
    text = re.sub(r"[\[\(][^\]\)]+[\]\)]", "", text)

    # Remove music notes ♪ ♫
    text = re.sub(r"[♪♫]+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove zero-width and bidi control chars + BOM + C0/C1 controls
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


def normalize_tts_tail_punctuation(text: str) -> str:
    """Replace trailing punctuation with a Chinese period for TTS input (SRT unchanged)."""
    trimmed = text.rstrip()
    if not trimmed:
        return trimmed
    trailing_map = {"。", "！", "？", "!", "?", ";", "；", "，", "、", "…", "﹖", "﹗", "．"}
    last = trimmed[-1]
    if last in trailing_map:
        return trimmed[:-1].rstrip() + "。"
    return trimmed


def parse_srt(srt_path: Path) -> list[SubtitleEntry]:
    """Parse SRT file into list of subtitle entries."""
    content = srt_path.read_text(encoding="utf-8")

    # Split by double newline (subtitle blocks)
    blocks = re.split(r"\n\s*\n", content.strip())

    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # First line: index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Second line: timestamps
        time_match = re.match(
            r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", lines[1].strip()
        )
        if not time_match:
            continue

        start_time, end_time = time_match.groups()

        # Remaining lines: text content (keep newlines for rolling subtitle detection)
        text_lines = lines[2:]
        # Store as newline-joined for rolling detection, will be cleaned later
        raw_text = "\n".join(text_lines)

        if raw_text.strip():  # Only add if there's actual text
            entries.append(
                SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=raw_text,  # Keep raw for now
                )
            )

    return entries


def merge_duplicate_entries(entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
    """Merge consecutive entries with identical text content.

    When the same text appears in consecutive subtitle entries,
    merge them into one entry with combined time range.
    """
    if not entries:
        return entries

    merged = []
    current = entries[0]

    for next_entry in entries[1:]:
        if current.text.strip() == next_entry.text.strip():
            # Same text, extend the time range
            current = SubtitleEntry(
                index=current.index,
                start_time=current.start_time,
                end_time=next_entry.end_time,
                text=current.text,
            )
        else:
            merged.append(current)
            current = next_entry

    merged.append(current)

    # Re-index
    for i, entry in enumerate(merged, 1):
        entry.index = i

    return merged


def merge_short_entries(
    entries: list[SubtitleEntry], min_chars: int = 10
) -> list[SubtitleEntry]:
    """Merge short consecutive entries for better TTS flow."""
    if not entries:
        return entries

    merged = []
    current = entries[0]

    for next_entry in entries[1:]:
        if len(current.text) < min_chars:
            # Merge with next
            current = SubtitleEntry(
                index=current.index,
                start_time=current.start_time,
                end_time=next_entry.end_time,
                text=f"{current.text} {next_entry.text}".strip(),
            )
        else:
            merged.append(current)
            current = next_entry

    merged.append(current)

    # Re-index after merging
    for i, entry in enumerate(merged, 1):
        entry.index = i

    return merged


def entries_to_txt_files(entries: list[SubtitleEntry], output_dir: Path) -> list[Path]:
    """Export each subtitle entry to a numbered text file for TTS batch processing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old txt files to avoid conflicts
    for old_file in output_dir.glob("*.txt"):
        old_file.unlink()

    paths = []
    for entry in entries:
        # Use zero-padded index for proper sorting
        filename = f"{entry.index:04d}.txt"
        filepath = output_dir / filename
        tts_text = normalize_tts_tail_punctuation(entry.text)
        filepath.write_text(tts_text, encoding="utf-8")
        paths.append(filepath)

    return paths


def srt_to_tts_input(
    srt_path: Path, output_dir: Path, min_chars: int | None = None
) -> list[SubtitleEntry]:
    """
    Parse SRT file, clean text, and export to TTS input format.

    Returns list of subtitle entries for reference.
    """
    import os

    print(f"[SRT] Parsing: {srt_path}")

    if min_chars is None:
        min_chars = int(os.getenv("SRT_MIN_CHARS", "10"))

    entries = parse_srt(srt_path)
    original_count = len(entries)

    # First: merge duplicate consecutive entries
    entries = merge_duplicate_entries(entries)
    duplicates_merged = original_count - len(entries)

    # Then: merge short entries
    before_short_merge = len(entries)
    entries = merge_short_entries(entries, min_chars)
    short_merged = before_short_merge - len(entries)

    # Final text cleanup for TTS (removes invisible chars)
    for entry in entries:
        entry.text = clean_text(entry.text)

    entries_to_txt_files(entries, output_dir)

    print(f"[SRT] Parsed {original_count} entries")
    if duplicates_merged > 0:
        print(f"[SRT] Merged {duplicates_merged} duplicate entries")
    if short_merged > 0:
        print(f"[SRT] Merged {short_merged} short entries")
    print(f"[SRT] ✓ Exported {len(entries)} text files to {output_dir}")

    return entries
