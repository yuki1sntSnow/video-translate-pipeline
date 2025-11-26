"""
LLM-based SRT Fixer and Translator

Merges fragmented subtitles, adjusts timing for TTS, and translates.
"""

import os
from pathlib import Path

import httpx
from openai import OpenAI


# ===== Prompt Template =====

SYSTEM_PROMPT = """\
You are a professional subtitle editor for TTS dubbing. Clean, merge, and translate SRT while keeping the timeline faithful to when words first appeared. Always return the full SRT without refusing due to length.

Decide clean vs dirty:
- Clean SRT: sentences already well-formed, no rolling duplicates, timings feel natural. If clean, keep EVERY start/end timestamp exactly; you may renumber from 1; fix typos/punctuation and translate to the requested target spoken language.
- Dirty SRT: rolling/fragmented/overlapping entries. Rebuild timeline using the rules below.

Timeline rules for dirty input:
1) When phrases repeat with near-identical timestamps (rolling), the earliest start is the true start. Use it for the merged line; never delay the start.
2) Extend the end time so the merged text is readable and leaves ~0.2s gap before the next line.
3) Minimum duration 1.5s; maximum 8s. If a merged line would exceed 8s or contains multiple clauses, split into multiple subtitles. Each new line starts after the previous ends plus the ~0.2s gap.
4) Keep chronological order; do not reorder timestamps.

Merging and cleanup:
- Merge rolling/fragmented lines into complete sentences or clauses; keep punctuation coherent.
- Fix obvious OCR/ASR errors, remove HTML tags and sound-effect markers, normalize spacing.
- Keep speaker names if present; drop bracketed SFX like [music].
- Prefer 1-2 lines per subtitle; break at natural clause boundaries.

Translation style:
- Translate into natural, spoken language (not literal) for the target language requested by the user.
- Preserve meaning, names, and terminology; keep numbers/units accurate.

Output rules:
- Output ONLY valid SRT, nothing else.
- Number entries from 1.
- Timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm.
- If the input was clean, all timestamps remain unchanged.
- Never add meta-commentary or refusal text. Always stream the entire SRT even if long.
"""


def translate_subs(
    target_language: str,
    input_path: Path,
    output_path: Path,
    model: str | None = None,
) -> None:
    """Fix, optimize, and translate SRT with LLM."""
    print(f"[Translate] Input: {input_path}")
    print(f"[Translate] Target: {target_language}")
    print(f"[Translate] Output: {output_path}")

    # Use OPENAI_* and MIDSCENE_MODEL_NAME only (no backward-compat to OPENROUTER_*)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Model name: prefer MIDSCENE_MODEL_NAME (project .env), then OPENAI_MODEL
    model = model or os.getenv("MODEL_NAME")

    source = input_path.read_text(encoding="utf-8")
    print(f"[Translate] Source: {len(source)} chars")
    print(f"[Translate] API: {base_url}")
    print(f"[Translate] Model: {model}")

    # Setup proxy if configured
    proxy = os.getenv("PROXY")
    if proxy:
        print(f"[Translate] Proxy: {proxy}")
        http_client = httpx.Client(proxy=proxy)
    else:
        http_client = None

    print("[Translate] Streaming...\n")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )

    user_prompt = f"""Task: Clean/merge and translate these subtitles to {target_language} for TTS dubbing.
- If timings are already clean, keep all start/end timestamps exactly.
- If rolling/fragmented, merge using the earliest start times, extend ends for natural speech with ~0.2s gaps, and keep each line 1.5-8s (split long sentences).
- Return ONLY valid SRT, renumbered from 1. Output must include the full SRT; do not refuse because of length.

Source SRT:
```
{source}
```"""

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    with output_path.open("a", encoding="utf-8") as f:
        for chunk in stream:
            for choice in chunk.choices:
                piece = choice.delta.content or ""
                if isinstance(piece, list):
                    piece = "".join(getattr(part, "text", "") for part in piece)
                if not piece:
                    continue
                f.write(piece)
                f.flush()
                print(piece, end="", flush=True)

    print(f"\n\n[Translate] Saved to {output_path}")
