"""
YouTube download wrapper using yt-dlp.
"""

import os
import subprocess
import sys
from pathlib import Path


def load_env(env_path: Path = Path(".env")) -> None:
    """Load environment variables from .env file."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def download_media(
    url: str,
    output_dir: Path,
    sub_langs: str,
    proxy: str | None,
    cookies_from_browser: str | None,
    cookies_file: str | None,
    extractor_args: str | None,
    format_selector: str | None,
    allow_auto_subs: bool,
    skip_download: bool,
) -> None:
    print(f"[Download] URL: {url}")
    print(f"[Download] Output: {output_dir}")
    print(f"[Download] Subtitles: {sub_langs}, Auto-subs: {allow_auto_subs}")
    if skip_download:
        print("[Download] Mode: Subtitles only")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "%(title)s.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-o",
        out_template,
        "--write-subs",
        "--sub-langs",
        sub_langs,
        "--convert-subs",
        "srt",
        "--merge-output-format",
        "mp4",
        url,
    ]
    if skip_download:
        cmd.append("--skip-download")
    if allow_auto_subs:
        cmd.append("--write-auto-subs")
    if proxy:
        cmd.extend(["--proxy", proxy])
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    if format_selector:
        cmd.extend(["-f", format_selector])
    print(f"[Download] Running: yt-dlp ...")
    try:
        subprocess.run(cmd, check=True)
        print("[Download] ✓ Download completed successfully")
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "yt-dlp failed; try updating yt-dlp or supplying cookies for age/region restrictions.\n"
            "Update: python -m pip install -U yt-dlp\n"
            'Cookies (optional): yt-dlp --cookies-from-browser chrome "<url>"'
        ) from exc
