"""SRT parser, text cleaner, and ASS converter for TTS and subtitle burning."""

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
    speaker: str | None = None  # Speaker label (e.g., "SPEAKER_00")


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

    Note: Speaker labels like [SPEAKER_00] are extracted separately, not removed here.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove speaker labels: [Speaker]: or (Speaker): (but not [SPEAKER_XX] format)
    text = re.sub(r"[\[\(](?!SPEAKER_)[^\]\)]+[\]\)]:\s*", "", text)

    # Remove sound effects: [music], (applause), etc. (but not [SPEAKER_XX])
    text = re.sub(r"[\[\(](?!SPEAKER_)[^\]\)]+[\]\)]", "", text)

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
    trailing_map = {
        "。",
        "！",
        "？",
        "!",
        "?",
        ";",
        "；",
        "，",
        "、",
        "…",
        "﹖",
        "﹗",
        "．",
    }
    last = trimmed[-1]
    if last in trailing_map:
        return trimmed[:-1].rstrip() + "。"
    return trimmed


def parse_srt(srt_path: Path) -> list[SubtitleEntry]:
    """Parse SRT file into list of subtitle entries.

    Supports multi-speaker format: [SPEAKER_XX] text
    Handles [Unknown] speaker by inheriting from previous entry.
    """
    content = srt_path.read_text(encoding="utf-8")

    # Split by double newline (subtitle blocks)
    blocks = re.split(r"\n\s*\n", content.strip())

    entries = []
    last_known_speaker = None  # Track last valid speaker for Unknown handling
    unknown_count = 0

    print(f"[SRT] 解析文件: {srt_path.name}")

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
        raw_text = "\n".join(text_lines)

        # Extract speaker label if present: [SPEAKER_XX] text
        speaker = None
        speaker_match = re.match(r"\[([A-Za-z_0-9]+)\]\s*(.+)", raw_text, re.DOTALL)
        if speaker_match:
            speaker = speaker_match.group(1)
            raw_text = speaker_match.group(2).strip()

            # Handle Unknown speaker: use last known speaker
            if speaker.lower() == "unknown":
                unknown_count += 1
                if last_known_speaker:
                    print(f"[SRT] ⚠ WARNING: 条目 #{index} 说话人为 [Unknown]，使用上文说话人 [{last_known_speaker}]")
                    speaker = last_known_speaker
                else:
                    print(f"[SRT] ⚠ WARNING: 条目 #{index} 说话人为 [Unknown]，无上文说话人可用，保持为 None")
                    speaker = None
            else:
                # Update last known speaker
                last_known_speaker = speaker

        if raw_text.strip():  # Only add if there's actual text
            entries.append(
                SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=raw_text,  # Keep raw for now
                    speaker=speaker,
                )
            )

    # Summary
    speakers = set(e.speaker for e in entries if e.speaker)
    print(f"[SRT] ✓ 解析完成: {len(entries)} 条字幕")
    if speakers:
        print(f"[SRT]   说话人: {sorted(speakers)}")
    if unknown_count > 0:
        print(f"[SRT]   ⚠ Unknown 说话人数量: {unknown_count}")

    return entries


def merge_duplicate_entries(entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
    """Merge consecutive entries with identical text content.

    When the same text appears in consecutive subtitle entries,
    merge them into one entry with combined time range.
    Only merge if speaker is the same (or both None).
    """
    if not entries:
        return entries

    merged = []
    current = entries[0]

    for next_entry in entries[1:]:
        # Only merge if same text AND same speaker
        same_text = current.text.strip() == next_entry.text.strip()
        same_speaker = current.speaker == next_entry.speaker

        if same_text and same_speaker:
            # Same text and speaker, extend the time range
            current = SubtitleEntry(
                index=current.index,
                start_time=current.start_time,
                end_time=next_entry.end_time,
                text=current.text,
                speaker=current.speaker,
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
    """Merge short consecutive entries for better TTS flow.

    Only merge if speaker is the same (or both None).
    """
    if not entries:
        return entries

    merged = []
    current = entries[0]

    for next_entry in entries[1:]:
        # Only merge if same speaker
        same_speaker = current.speaker == next_entry.speaker

        if len(current.text) < min_chars and same_speaker:
            # Merge with next
            current = SubtitleEntry(
                index=current.index,
                start_time=current.start_time,
                end_time=next_entry.end_time,
                text=f"{current.text} {next_entry.text}".strip(),
                speaker=current.speaker,
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
    """Export each subtitle entry to a numbered text file for TTS batch processing.

    For single speaker mode, outputs to output_dir directly.
    """
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


def entries_to_txt_files_by_speaker(
    entries: list[SubtitleEntry], output_dir: Path
) -> dict[str, list[Path]]:
    """Export subtitle entries to text files, grouped by speaker.

    Output structure:
      output_dir/
        SPEAKER_00/
          0001.txt
          0003.txt
        SPEAKER_01/
          0002.txt
          0004.txt

    Returns:
        Dict mapping speaker name to list of file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group entries by speaker
    speaker_entries: dict[str, list[SubtitleEntry]] = {}
    for entry in entries:
        speaker = entry.speaker or "UNKNOWN"
        if speaker not in speaker_entries:
            speaker_entries[speaker] = []
        speaker_entries[speaker].append(entry)

    result = {}
    for speaker, speaker_entries_list in speaker_entries.items():
        speaker_dir = output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old txt files in speaker dir
        for old_file in speaker_dir.glob("*.txt"):
            old_file.unlink()

        paths = []
        for entry in speaker_entries_list:
            # Use original index to maintain ordering across speakers
            filename = f"{entry.index:04d}.txt"
            filepath = speaker_dir / filename
            tts_text = normalize_tts_tail_punctuation(entry.text)
            filepath.write_text(tts_text, encoding="utf-8")
            paths.append(filepath)

        result[speaker] = paths

    return result


def get_speakers_from_entries(entries: list[SubtitleEntry]) -> list[str]:
    """Get unique speaker names from entries."""
    speakers = set()
    for entry in entries:
        if entry.speaker:
            speakers.add(entry.speaker)
    return sorted(speakers)


def srt_to_tts_input(
    srt_path: Path, output_dir: Path, min_chars: int | None = None
) -> list[SubtitleEntry]:
    """
    Parse SRT file, clean text, and export to TTS input format.

    Returns list of subtitle entries for reference.
    """
    import config as cfg

    print(f"[SRT] Parsing: {srt_path}")

    if min_chars is None:
        min_chars = cfg.SRT_MIN_CHARS

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


# ==================== SRT Preprocessing for Burn-in ====================


def get_display_width(text: str) -> float:
    """Calculate display width of text for subtitle wrapping.

    Chinese/fullwidth chars count as 1.0, ASCII/halfwidth as 0.5.
    This gives approximate visual width for mixed Chinese/English text.
    """
    width = 0.0
    for char in text:
        # CJK unified ideographs and common fullwidth ranges
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            width += 1.0
        elif '\u3000' <= char <= '\u303f':  # CJK Punctuation
            width += 1.0
        elif '\uff00' <= char <= '\uffef':  # Fullwidth forms
            width += 1.0
        elif '\u3400' <= char <= '\u4dbf':  # CJK Extension A
            width += 1.0
        elif '\U00020000' <= char <= '\U0002a6df':  # CJK Extension B
            width += 1.0
        else:
            # ASCII, Latin, etc. - halfwidth
            width += 0.5
    return width


def wrap_subtitle_text(text: str, max_width: float = 22.0, max_lines: int = 2) -> str:
    """Wrap subtitle text to fit within max display width.

    Args:
        text: Subtitle text (may contain newlines)
        max_width: Max display width per line (in CJK char units)
                   Default 22 = ~44 ASCII chars, good for 1080p
        max_lines: Maximum number of lines allowed (default 2, 0=unlimited)

    Returns:
        Text with newlines inserted for wrapping (limited to max_lines)
    """
    # If already has newlines, process each line separately
    if '\n' in text:
        lines = text.split('\n')
        sub_max = 1 if max_lines == 0 else max_lines
        wrapped_lines = [wrap_subtitle_text(line, max_width, max_lines=sub_max) for line in lines]
        result = '\n'.join(wrapped_lines)
        # Limit total lines (unless max_lines=0 means unlimited)
        if max_lines > 0:
            final_lines = result.split('\n')
            if len(final_lines) > max_lines:
                return '\n'.join(final_lines[:max_lines])
        return result

    # Check if wrapping is needed
    if get_display_width(text) <= max_width:
        return text

    # Find best split point
    # Try to split at punctuation or space
    best_split = -1
    target_width = max_width * 0.5  # Aim for balanced lines

    # Preferred split characters (in order of preference)
    split_chars = '，。！？、；：,;: '

    current_width = 0.0
    for i, char in enumerate(text):
        char_width = 1.0 if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\uffef' else 0.5
        current_width += char_width

        # Look for split points
        if char in split_chars and current_width >= target_width:
            best_split = i + 1  # Split after punctuation
            break
        elif current_width >= max_width:
            # Force split at max width if no good split point found
            if best_split == -1:
                best_split = i
            break

    if best_split == -1 or best_split >= len(text) - 1:
        return text  # Can't split or split would be at end

    # Split text
    first_part = text[:best_split].rstrip()
    second_part = text[best_split:].lstrip()

    # Only recursively wrap if we haven't reached max_lines yet (or unlimited when max_lines=0)
    if get_display_width(second_part) > max_width:
        if max_lines == 0:
            # Unlimited - keep recursing
            second_part = wrap_subtitle_text(second_part, max_width, 0)
        elif max_lines > 1:
            # Limited - decrease remaining
            second_part = wrap_subtitle_text(second_part, max_width, max_lines - 1)

    result = f"{first_part}\n{second_part}"

    # Final check: limit to max_lines (only if max_lines > 0)
    if max_lines > 0:
        final_lines = result.split('\n')
        if len(final_lines) > max_lines:
            return '\n'.join(final_lines[:max_lines])

    return result


def remove_speaker_labels(text: str) -> str:
    """Remove speaker labels like [SPEAKER_XX] from subtitle text."""
    # Remove [SPEAKER_XX] or [Unknown] at the beginning of text
    return re.sub(r'^\[(?:SPEAKER_\d+|Unknown)\]\s*', '', text, flags=re.IGNORECASE)


# ==================== ASS Subtitle Style Configuration ====================

@dataclass
class SubtitleStyle:
    """ASS subtitle style configuration.

    Colors use ASS format: &HBBGGRR (BGR order, not RGB!)
    Example: &H00FFFFFF = white, &H00000000 = black
    """
    # Font settings
    font_name: str = "Microsoft YaHei"  # 微软雅黑
    font_size: int = 45  # 字号
    bold: bool = False  # 粗体
    italic: bool = False  # 斜体

    # Colors (ASS BGR format: &HBBGGRR)
    primary_color: str = "&H00FFFFFF"  # 主颜色（白色）
    secondary_color: str = "&H000000FF"  # 次颜色（红色，用于卡拉OK效果）
    outline_color: str = "&H00000000"  # 描边颜色（黑色）
    back_color: str = "&H80000000"  # 阴影/背景颜色（半透明黑色）

    # Border and shadow
    outline: float = 2.0  # 描边宽度
    shadow: float = 1.0  # 阴影距离

    # Position and alignment
    alignment: int = 2  # 对齐方式: 1-3下排左中右, 4-6中排, 7-9上排
    margin_l: int = 30  # 左边距
    margin_r: int = 30  # 右边距
    margin_v: int = 25  # 垂直边距（底部）

    # Advanced
    scale_x: int = 100  # 水平缩放 %
    scale_y: int = 100  # 垂直缩放 %
    spacing: float = 0  # 字间距
    angle: float = 0  # 旋转角度
    border_style: int = 1  # 边框样式: 1=描边+阴影, 3=不透明框

    def to_ass_style_line(self, name: str = "Default") -> str:
        """Generate ASS Style line."""
        bold_val = -1 if self.bold else 0
        italic_val = -1 if self.italic else 0

        return (
            f"Style: {name},{self.font_name},{self.font_size},"
            f"{self.primary_color},{self.secondary_color},"
            f"{self.outline_color},{self.back_color},"
            f"{bold_val},{italic_val},0,0,"  # Bold, Italic, Underline, StrikeOut
            f"{self.scale_x},{self.scale_y},{self.spacing},{self.angle},"
            f"{self.border_style},{self.outline},{self.shadow},"
            f"{self.alignment},{self.margin_l},{self.margin_r},{self.margin_v},1"  # Encoding=1
        )


def get_default_subtitle_style() -> SubtitleStyle:
    """Get default subtitle style from config."""
    import config as cfg

    return SubtitleStyle(
        font_name=cfg.SUB_FONT_NAME,
        font_size=cfg.SUB_FONT_SIZE,
        bold=cfg.SUB_BOLD,
        italic=cfg.SUB_ITALIC,
        primary_color=cfg.SUB_PRIMARY_COLOR,
        secondary_color=cfg.SUB_SECONDARY_COLOR,
        outline_color=cfg.SUB_OUTLINE_COLOR,
        back_color=cfg.SUB_BACK_COLOR,
        outline=cfg.SUB_OUTLINE,
        shadow=cfg.SUB_SHADOW,
        alignment=cfg.SUB_ALIGNMENT,
        margin_l=cfg.SUB_MARGIN_L,
        margin_r=cfg.SUB_MARGIN_R,
        margin_v=cfg.SUB_MARGIN_V,
        border_style=cfg.SUB_BORDER_STYLE,
    )


def srt_timestamp_to_ass(ts: str) -> str:
    """Convert SRT timestamp to ASS format.

    SRT: 00:01:23,456
    ASS: 0:01:23.46
    """
    # Replace comma with dot
    ts = ts.replace(',', '.')
    # Remove leading zero from hours if present
    if ts.startswith('0'):
        ts = ts[1:]
    # Truncate milliseconds to centiseconds (2 digits)
    parts = ts.rsplit('.', 1)
    if len(parts) == 2:
        ts = parts[0] + '.' + parts[1][:2]
    return ts


def seconds_to_ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp format.

    Args:
        seconds: Time in seconds (e.g., 83.456)

    Returns:
        ASS format timestamp (e.g., "0:01:23.45")
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def split_long_subtitle(
    text: str,
    start_seconds: float,
    end_seconds: float,
    max_line_width: float = 22.0,
    max_lines: int = 2,
) -> list[tuple[str, str, str]]:
    """Split a long subtitle into multiple entries with proportional timing.

    If wrapping would exceed max_lines, split at punctuation and distribute time
    proportionally by character count.

    Args:
        text: Subtitle text
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        max_line_width: Max display width per line (CJK char units)
        max_lines: Maximum lines per subtitle entry

    Returns:
        List of (text, start_time_ass, end_time_ass) tuples
    """
    # First try normal wrapping
    wrapped = wrap_subtitle_text(text, max_line_width, max_lines=0)  # No limit to see actual lines
    line_count = len(wrapped.split('\n'))

    # If it fits within max_lines, no need to split
    if line_count <= max_lines:
        wrapped_limited = wrap_subtitle_text(text, max_line_width, max_lines)
        return [(wrapped_limited, seconds_to_ass_timestamp(start_seconds), seconds_to_ass_timestamp(end_seconds))]

    # Need to split into multiple subtitle entries
    # Split at major punctuation points
    split_punctuation = '。！？!?；;'
    secondary_punctuation = '，,、：:'

    segments = []
    current_segment = ""

    for char in text:
        current_segment += char
        if char in split_punctuation:
            segments.append(current_segment.strip())
            current_segment = ""

    # Add remaining text
    if current_segment.strip():
        segments.append(current_segment.strip())

    # If no splits found, try secondary punctuation
    if len(segments) <= 1:
        segments = []
        current_segment = ""
        for char in text:
            current_segment += char
            if char in secondary_punctuation:
                segments.append(current_segment.strip())
                current_segment = ""
        if current_segment.strip():
            segments.append(current_segment.strip())

    # If still no splits, fall back to single entry with truncated lines
    if len(segments) <= 1:
        wrapped_limited = wrap_subtitle_text(text, max_line_width, max_lines)
        return [(wrapped_limited, seconds_to_ass_timestamp(start_seconds), seconds_to_ass_timestamp(end_seconds))]

    # Merge segments into groups that fit within max_lines when wrapped
    merged_segments = []
    current_group = ""

    for segment in segments:
        # Don't add space for CJK text
        test_text = (current_group + segment).strip() if current_group else segment
        wrapped_test = wrap_subtitle_text(test_text, max_line_width, max_lines=0)

        if len(wrapped_test.split('\n')) <= max_lines:
            current_group = test_text
        else:
            if current_group:
                merged_segments.append(current_group)
            current_group = segment

    if current_group:
        merged_segments.append(current_group)

    # If we only got one segment after merging, return as-is
    if len(merged_segments) <= 1:
        wrapped_limited = wrap_subtitle_text(text, max_line_width, max_lines)
        return [(wrapped_limited, seconds_to_ass_timestamp(start_seconds), seconds_to_ass_timestamp(end_seconds))]

    # Calculate proportional timing based on character count
    total_chars = sum(len(seg) for seg in merged_segments)
    total_duration = end_seconds - start_seconds

    results = []
    current_time = start_seconds

    for i, segment in enumerate(merged_segments):
        # Calculate duration based on character proportion
        char_ratio = len(segment) / total_chars
        segment_duration = total_duration * char_ratio

        segment_start = current_time
        segment_end = current_time + segment_duration

        # Ensure last segment ends exactly at end_seconds
        if i == len(merged_segments) - 1:
            segment_end = end_seconds

        # Wrap the segment text
        wrapped_segment = wrap_subtitle_text(segment, max_line_width, max_lines)

        results.append((
            wrapped_segment,
            seconds_to_ass_timestamp(segment_start),
            seconds_to_ass_timestamp(segment_end)
        ))

        current_time = segment_end

    return results


def convert_srt_to_ass(
    srt_path: Path,
    output_path: Path | None = None,
    style: SubtitleStyle | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    max_line_width: float = 22.0,
    max_lines: int = 2,
    remove_speakers: bool = True,
) -> Path:
    """Convert SRT file to ASS format with custom styling.

    Args:
        srt_path: Input SRT file
        output_path: Output ASS file (default: input.ass)
        style: Subtitle style configuration
        video_width: Video width for PlayResX
        video_height: Video height for PlayResY
        max_line_width: Max display width per line (CJK char units)
        max_lines: Maximum lines per subtitle (default 2)
        remove_speakers: Whether to remove speaker labels

    Returns:
        Path to generated ASS file
    """
    if output_path is None:
        output_path = srt_path.with_suffix('.ass')

    if style is None:
        style = get_default_subtitle_style()

    # Parse SRT content
    content = srt_path.read_text(encoding='utf-8')
    blocks = re.split(r'\n\s*\n', content.strip())

    # Build ASS header
    ass_header = f"""[Script Info]
Title: Converted from {srt_path.name}
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style.to_ass_style_line("Default")}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Process each subtitle block
    dialogue_lines = []
    split_count = 0  # Track how many entries were split

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Parse index
        try:
            int(lines[0].strip())
        except ValueError:
            continue

        # Parse timestamps
        time_match = re.match(
            r'(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)',
            lines[1].strip()
        )
        if not time_match:
            continue

        start_time_srt = time_match.group(1)
        end_time_srt = time_match.group(2)
        start_seconds = parse_timestamp(start_time_srt)
        end_seconds = parse_timestamp(end_time_srt)

        # Get text content
        text = '\n'.join(lines[2:])

        # Remove speaker labels
        if remove_speakers:
            text = remove_speaker_labels(text)

        # Split long subtitles into multiple entries with proportional timing
        subtitle_parts = split_long_subtitle(
            text, start_seconds, end_seconds, max_line_width, max_lines
        )

        if len(subtitle_parts) > 1:
            split_count += 1

        # Create dialogue lines for each part
        for part_text, part_start, part_end in subtitle_parts:
            # Convert newlines to ASS format (\N)
            ass_text = part_text.replace('\n', '\\N')
            dialogue_line = f"Dialogue: 0,{part_start},{part_end},Default,,0,0,0,,{ass_text}"
            dialogue_lines.append(dialogue_line)

    # Write ASS file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_content = ass_header + '\n'.join(dialogue_lines) + '\n'
    output_path.write_text(ass_content, encoding='utf-8-sig')  # BOM for better compatibility

    print(f"[SRT→ASS] ✓ 转换完成: {output_path}")
    print(f"[SRT→ASS]   - 字体: {style.font_name} {style.font_size}pt")
    print(f"[SRT→ASS]   - 描边: {style.outline}px")
    print(f"[SRT→ASS]   - 条目数: {len(dialogue_lines)}")
    if split_count > 0:
        print(f"[SRT→ASS]   - 长字幕分割: {split_count} 条")

    return output_path


def preprocess_srt_for_burn(
    srt_path: Path,
    output_path: Path | None = None,
    max_line_width: float = 22.0,
    remove_speakers: bool = True,
) -> Path:
    """Preprocess SRT file for subtitle burn-in.

    Operations:
    1. Remove speaker labels [SPEAKER_XX]
    2. Wrap long lines for better display

    Args:
        srt_path: Input SRT file
        output_path: Output SRT file (default: input_burn.srt)
        max_line_width: Max display width per line (CJK char units)
                        Default 22 ≈ 44 ASCII chars, suitable for 1080p
        remove_speakers: Whether to remove speaker labels

    Returns:
        Path to preprocessed SRT file
    """
    if output_path is None:
        output_path = srt_path.with_suffix('.burn.srt')

    content = srt_path.read_text(encoding='utf-8')

    # Split by double newline (subtitle blocks)
    blocks = re.split(r'\n\s*\n', content.strip())

    processed_blocks = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            processed_blocks.append(block)
            continue

        # First line: index
        index_line = lines[0]

        # Second line: timestamps
        time_line = lines[1]

        # Remaining lines: text content
        text_lines = lines[2:]
        text = '\n'.join(text_lines)

        # Remove speaker labels
        if remove_speakers:
            text = remove_speaker_labels(text)

        # Wrap long lines
        text = wrap_subtitle_text(text, max_line_width)

        # Reconstruct block
        processed_block = f"{index_line}\n{time_line}\n{text}"
        processed_blocks.append(processed_block)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n\n'.join(processed_blocks) + '\n', encoding='utf-8')

    print(f"[SRT] ✓ 预处理完成: {output_path}")
    print(f"[SRT]   - 移除说话人标签: {remove_speakers}")
    print(f"[SRT]   - 最大行宽: {max_line_width} (中文字符单位)")

    return output_path
