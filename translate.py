"""LLM-based SRT translator with async chunked processing."""

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


# ===== Configuration =====
DEFAULT_CHUNK_SIZE = 200  # Lines per chunk
DEFAULT_MAX_WAIT = 300  # Max wait time for async task (seconds)
DEFAULT_POLL_INTERVAL = 2  # Poll interval (seconds)
DEFAULT_MAX_CONCURRENT = 5  # Max concurrent tasks


@dataclass
class SrtEntry:
    """A single SRT subtitle entry."""

    index: int
    start_time: str
    end_time: str
    text: str
    speaker: str | None = None  # Speaker label (e.g., "SPEAKER_00")


@dataclass
class TranslationTask:
    """A translation task for async processing."""

    chunk_index: int
    text_lines: list[str]
    line_nums: list[int]
    temp_dir: Path | None = None  # For saving output immediately
    task_id: str | None = None
    result: list[str] | None = None
    missing_nums: list[int] | None = None
    error: str | None = None


# ===== Prompt Template =====

SYSTEM_PROMPT = """\
你是一位专业的字幕翻译专家，专门负责将视频字幕翻译成自然流畅的目标语言。

翻译原则：
1. **意译优先**：翻译成自然的口语表达，不要逐字直译。译文要像是目标语言母语者会说的话。
2. **保持对应**：每行输入对应一行输出，不要合并或拆分行。
3. **保留行号**：输入带有 [序号] 前缀，输出也必须保留相同的 [序号] 前缀。
4. **语境连贯**：虽然分行显示，但要理解上下文语境，确保翻译连贯自然。
5. **专业术语**：技术术语、产品名、人名等专有名词保持准确，可保留英文或使用通用译法。
6. **口语化**：这是视频配音字幕，翻译要适合朗读，避免书面语和冗长表达。
7. **简洁有力**：字幕需要简短易读，在保持原意的前提下尽量精简。

输出格式：
- 每行必须以 [序号] 开头，序号与输入对应
- 序号后面是翻译后的文本
- 不要输出任何解释说明
- 示例输入: [1] Hello world
- 示例输出: [1] 你好世界
"""


def parse_srt(srt_content: str) -> list[SrtEntry]:
    """Parse SRT content into list of entries.

    Extracts speaker labels from text (e.g., [SPEAKER_00]) and stores separately.
    """
    entries = []
    blocks = re.split(r"\n\s*\n", srt_content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Parse index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Parse timestamps
        time_match = re.match(
            r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", lines[1].strip()
        )
        if not time_match:
            continue

        start_time, end_time = time_match.groups()

        # Text content (may span multiple lines within entry)
        text = " ".join(lines[2:]).strip()

        # Extract speaker label if present: [SPEAKER_XX] text
        speaker = None
        speaker_match = re.match(r"\[([A-Za-z_0-9]+)\]\s*(.+)", text, re.DOTALL)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = speaker_match.group(2).strip()

        entries.append(
            SrtEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
                speaker=speaker,
            )
        )

    return entries


def entries_to_srt(entries: list[SrtEntry]) -> str:
    """Convert entries back to SRT format.

    Re-adds speaker labels to text if present.
    """
    srt_lines = []
    for i, entry in enumerate(entries, 1):
        srt_lines.append(str(i))
        srt_lines.append(f"{entry.start_time} --> {entry.end_time}")
        # Re-add speaker label if present
        if entry.speaker:
            srt_lines.append(f"[{entry.speaker}] {entry.text}")
        else:
            srt_lines.append(entry.text)
        srt_lines.append("")
    return "\n".join(srt_lines)


def chunk_entries(
    entries: list[SrtEntry], chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[list[SrtEntry]]:
    """Split entries into chunks for batch translation."""
    chunks = []
    for i in range(0, len(entries), chunk_size):
        chunks.append(entries[i : i + chunk_size])
    return chunks


def entries_to_text_lines(entries: list[SrtEntry]) -> list[str]:
    """Extract just the text from entries, one per line."""
    return [entry.text for entry in entries]


def get_api_config() -> tuple[str, str, str, str | None]:
    """Get API configuration from config module.

    Returns:
        Tuple of (api_key, base_url, model, proxy)
    """
    import config as cfg

    if not cfg.OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY in the environment.")

    return cfg.OPENAI_API_KEY, cfg.OPENAI_BASE_URL, cfg.MODEL_NAME, cfg.PROXY or None


def _build_translation_prompt(
    text_lines: list[str],
    target_language: str,
    line_nums: list[int],
) -> str:
    """Build translation prompt with line numbers.

    Returns:
        User prompt string
    """
    # Format input with line numbers: [1] text, [2] text, ...
    numbered_lines = []
    for i, line in enumerate(text_lines):
        line_num = line_nums[i]
        numbered_lines.append(f"[{line_num}] {line}")

    input_text = "\n".join(numbered_lines)

    user_prompt = f"""将以下字幕文本翻译成{target_language}。

要求：
- 每行以 [序号] 开头，翻译后必须保留相同的 [序号]
- 输入共 {len(text_lines)} 行，输出必须也是 {len(text_lines)} 行
- 只输出翻译结果，不要额外解释

原文：
{input_text}"""

    return user_prompt


def parse_numbered_output(
    output_text: str,
    original_lines: list[str],
    line_nums: list[int],
) -> tuple[list[str], list[int]]:
    """
    Parse LLM output with line numbers and align with original lines.

    Expected format: [N] translated text
    Falls back to original text for missing/invalid lines.

    Args:
        output_text: Raw LLM output
        original_lines: Original text lines for fallback
        line_nums: Expected line numbers

    Returns:
        Tuple of (translated lines aligned with original, list of missing line numbers)
    """
    # Pattern to match [number] text
    line_pattern = re.compile(r"^\[(\d+)\]\s*(.*)$")

    # Build a dict: line_number -> translated_text
    translations = {}

    for line in output_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = line_pattern.match(line)
        if match:
            line_num = int(match.group(1))
            text = match.group(2).strip()
            # Only accept if not already seen (keep first occurrence)
            if line_num not in translations:
                translations[line_num] = text
        else:
            # Line without number prefix - try to handle gracefully
            if len(line) > 2:  # Skip very short garbage
                print(f"[Translate] ⚠ 无法解析行 (缺少序号): {line[:50]}...")

    # Build result array aligned with original
    result = []
    missing_line_nums = []

    for i, orig_line in enumerate(original_lines):
        line_num = line_nums[i]
        if line_num in translations:
            result.append(translations[line_num])
        else:
            # Fallback to original text, record missing
            result.append(orig_line)
            missing_line_nums.append(line_num)

    if missing_line_nums:
        print(
            f"[Translate] ⚠ 缺失 {len(missing_line_nums)} 行翻译: "
            f"{missing_line_nums[:20]}{'...' if len(missing_line_nums) > 20 else ''}"
        )

    return result, missing_line_nums


# ===== Async API Functions =====


async def submit_async_task(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
) -> str:
    """Submit async translation task and return task_id.

    Args:
        client: httpx async client
        api_key: API key
        base_url: Base URL for API
        model: Model name
        messages: Chat messages

    Returns:
        task_id for polling result
    """
    url = f"{base_url}/chat/completions?async=true"

    payload = {
        "model": model,
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = await client.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    task_id = result.get("task_id")

    if not task_id:
        raise RuntimeError(f"No task_id in response: {result}")

    return task_id


async def poll_async_result(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    task_id: str,
    chunk_idx: int = 0,
    max_wait: int = DEFAULT_MAX_WAIT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> str:
    """Poll for async task result.

    Args:
        client: httpx async client
        api_key: API key
        base_url: Base URL for API
        task_id: Task ID to poll
        chunk_idx: Chunk index for logging
        max_wait: Maximum wait time in seconds
        poll_interval: Polling interval in seconds

    Returns:
        Translated text content
    """
    url = f"{base_url}/async_result?task_id={task_id}"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    elapsed = 0
    log_prefix = f"[Translate] 分块 {chunk_idx}" if chunk_idx > 0 else "[Translate]"

    while elapsed < max_wait:
        response = await client.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            result = response.json()

            # Check if task is still pending
            err = result.get("err", "")
            if err == "result pending":
                # Task still processing, wait and retry
                if elapsed > 0 and elapsed % 30 == 0:
                    print(f"{log_prefix} 等待中... ({elapsed}s)")
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                continue

            # Check if task is complete
            status_code = result.get("status_code")

            if status_code == 200:
                # Task completed successfully
                # Extract content from OpenAI-style response
                data = result.get("data", {})

                # Check for corrupted/compressed data (API error)
                if isinstance(data, str) and (data.startswith("\x1f\x8b") or not data):
                    raise RuntimeError(f"API returned invalid data, need retry")

                if isinstance(data, dict):
                    choices = data.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            return content

                # Fallback: try extracted_content
                content = result.get("extracted_content", "")
                if content:
                    return content

                # No content yet, might still be processing
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                continue

            elif status_code and status_code >= 400:
                # Task failed
                err = result.get("err", "Unknown error")
                raise RuntimeError(f"Async task failed: {err}")

            # Task still processing, wait and retry

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    raise RuntimeError(f"Async task {task_id} timed out after {max_wait} seconds")


async def translate_chunk_async(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    task: TranslationTask,
    target_language: str,
    total_chunks: int,
    semaphore: asyncio.Semaphore,
) -> TranslationTask:
    """
    Translate a single chunk asynchronously.

    Args:
        client: httpx async client
        api_key: API key
        base_url: Base URL for API
        model: Model name
        task: Translation task with text_lines and line_nums
        target_language: Target language
        total_chunks: Total number of chunks (for logging)
        semaphore: Semaphore for concurrency control

    Returns:
        Updated TranslationTask with result or error
    """
    async with semaphore:
        chunk_idx = task.chunk_index
        text_lines = task.text_lines
        line_nums = task.line_nums

        print(
            f"[Translate] 分块 {chunk_idx}/{total_chunks} "
            f"({len(text_lines)} 行, 行号 {line_nums[0]}-{line_nums[-1]}) 提交中..."
        )

        user_prompt = _build_translation_prompt(text_lines, target_language, line_nums)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Submit async task
                task_id = await submit_async_task(client, api_key, base_url, model, messages)
                task.task_id = task_id
                print(f"[Translate] 分块 {chunk_idx} 任务ID: {task_id}, 轮询中...")

                # Poll for result
                translated_text = await poll_async_result(
                    client, api_key, base_url, task_id, chunk_idx=chunk_idx
                )

                if translated_text:
                    # Parse output
                    translated_lines, missing_nums = parse_numbered_output(
                        translated_text, text_lines, line_nums
                    )
                    task.result = translated_lines
                    task.missing_nums = missing_nums

                    # Save to temp immediately (防止程序中断丢失结果)
                    if task.temp_dir:
                        output_file = task.temp_dir / f"04_chunk_{chunk_idx:03d}_output.txt"
                        numbered_output = [
                            f"[{num}] {line}"
                            for num, line in zip(line_nums, translated_lines)
                        ]
                        output_file.write_text("\n".join(numbered_output), encoding="utf-8")

                    print(f"[Translate] ✓ 分块 {chunk_idx} 完成")
                    return task
                else:
                    print(f"[Translate] ⚠ 分块 {chunk_idx} API 返回空响应，重试 {attempt + 1}/{max_retries}...")

            except Exception as e:
                print(f"[Translate] ⚠ 分块 {chunk_idx} 失败: {e}")
                if attempt < max_retries - 1:
                    print(f"[Translate] 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    task.error = str(e)
                    task.result = text_lines  # Fallback to original
                    task.missing_nums = line_nums
                    print(f"[Translate] ✗ 分块 {chunk_idx} 达到最大重试次数，使用原文")
                    return task

        # Should not reach here, but just in case
        task.result = text_lines
        task.missing_nums = line_nums
        return task


async def translate_all_chunks_async(
    tasks: list[TranslationTask],
    api_key: str,
    base_url: str,
    model: str,
    target_language: str,
    proxy: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> list[TranslationTask]:
    """
    Translate all chunks concurrently.

    Args:
        tasks: List of translation tasks
        api_key: API key
        base_url: Base URL for API
        model: Model name
        target_language: Target language
        proxy: Optional proxy URL
        max_concurrent: Max concurrent tasks

    Returns:
        List of completed translation tasks
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(proxy=proxy) as client:
        coros = [
            translate_chunk_async(
                client, api_key, base_url, model, task,
                target_language, len(tasks), semaphore
            )
            for task in tasks
        ]

        # Run all tasks concurrently
        completed = await asyncio.gather(*coros, return_exceptions=True)

        # Handle exceptions
        results = []
        for i, result in enumerate(completed):
            if isinstance(result, Exception):
                print(f"[Translate] ✗ 分块 {i + 1} 异常: {result}")
                tasks[i].error = str(result)
                tasks[i].result = tasks[i].text_lines
                tasks[i].missing_nums = tasks[i].line_nums
                results.append(tasks[i])
            else:
                results.append(result)

        return results


def translate_subs(
    target_language: str,
    input_path: Path,
    output_path: Path,
    model: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    temp_dir: Path | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> None:
    """
    Translate SRT file with async chunked processing.

    Pipeline:
      1. Parse SRT → extract text only (preserve timestamps separately)
      2. Split text into chunks by line count
      3. Submit all chunks async → poll results concurrently
      4. Retry missing lines
      5. Reassemble translated text with original timestamps
      6. Output new SRT

    All intermediate results are saved to temp_dir for debugging/recovery.

    Args:
        target_language: Target language for translation
        input_path: Input SRT file path
        output_path: Output SRT file path
        model: Override model name (uses env var if None)
        chunk_size: Lines per chunk (default 200)
        temp_dir: Directory for intermediate files (optional)
        max_concurrent: Max concurrent translation tasks (default 5)
    """
    print(f"[Translate] ===== 开始翻译 (异步模式) =====")
    print(f"[Translate] 输入: {input_path}")
    print(f"[Translate] 目标语言: {target_language}")
    print(f"[Translate] 输出: {output_path}")

    # Setup temp directory for intermediate files
    if temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Translate] 中间文件: {temp_dir}")

    # Parse SRT - extract entries with timestamps
    srt_content = input_path.read_text(encoding="utf-8")
    entries = parse_srt(srt_content)
    print(f"[Translate] 解析到 {len(entries)} 条字幕")

    # Save parsed entries
    if temp_dir:
        entries_file = temp_dir / "01_parsed_entries.json"
        entries_data = [asdict(e) for e in entries]
        entries_file.write_text(
            json.dumps(entries_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[Translate] 保存解析结果: {entries_file}")

    # Report speaker info
    speakers = set(e.speaker for e in entries if e.speaker)
    if speakers:
        print(f"[Translate] 检测到说话人标签: {sorted(speakers)}")

    if not entries:
        raise ValueError("SRT 文件为空或格式错误")

    # Get API config
    api_key, base_url, default_model, proxy = get_api_config()
    model = model or default_model

    print(f"[Translate] 模型: {model}")
    print(f"[Translate] 分块大小: {chunk_size} 行")
    print(f"[Translate] 最大并发: {max_concurrent}")

    # Chunk entries by line count
    chunks = chunk_entries(entries, chunk_size)
    print(f"[Translate] 共 {len(entries)} 条字幕, {len(chunks)} 个分块")

    # Save chunk info
    if temp_dir:
        chunks_info = []
        for i, chunk in enumerate(chunks):
            chunks_info.append({
                "chunk_index": i + 1,
                "num_entries": len(chunk),
                "start_index": chunk[0].index if chunk else 0,
                "end_index": chunk[-1].index if chunk else 0,
            })
        chunks_file = temp_dir / "02_chunks_info.json"
        chunks_file.write_text(
            json.dumps(chunks_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Prepare translation tasks
    tasks: list[TranslationTask] = []
    current_line_number = 1

    for i, chunk in enumerate(chunks, 1):
        text_lines = entries_to_text_lines(chunk)
        line_nums = list(range(current_line_number, current_line_number + len(text_lines)))

        # Save chunk input
        if temp_dir:
            chunk_input_file = temp_dir / f"03_chunk_{i:03d}_input.txt"
            numbered_input = [f"[{num}] {line}" for num, line in zip(line_nums, text_lines)]
            chunk_input_file.write_text("\n".join(numbered_input), encoding="utf-8")

        tasks.append(TranslationTask(
            chunk_index=i,
            text_lines=text_lines,
            line_nums=line_nums,
            temp_dir=temp_dir,
        ))

        current_line_number += len(text_lines)

    # Run async translation
    print(f"[Translate] 开始异步翻译 {len(tasks)} 个分块...")
    completed_tasks = asyncio.run(
        translate_all_chunks_async(
            tasks, api_key, base_url, model, target_language, proxy, max_concurrent
        )
    )

    # Collect results
    all_translated_lines: list[str] = []
    all_missing: dict[int, str] = {}  # line_num -> original_text

    for task in completed_tasks:
        if task.result:
            all_translated_lines.extend(task.result)

            # Note: chunk output already saved in translate_chunk_async

            # Record missing lines
            if task.missing_nums:
                for missing_num in task.missing_nums:
                    idx = missing_num - task.line_nums[0]
                    if 0 <= idx < len(task.text_lines):
                        all_missing[missing_num] = task.text_lines[idx]
        else:
            # Task failed completely, use original text
            all_translated_lines.extend(task.text_lines)
            for num, text in zip(task.line_nums, task.text_lines):
                all_missing[num] = text

    # Save all translations before retry
    if temp_dir:
        all_trans_file = temp_dir / "05_all_translations_before_retry.txt"
        numbered_all = [f"[{i+1}] {line}" for i, line in enumerate(all_translated_lines)]
        all_trans_file.write_text("\n".join(numbered_all), encoding="utf-8")

    # Retry missing lines if any
    if all_missing:
        print(f"[Translate] ===== 重试缺失的 {len(all_missing)} 行 =====")

        # Save missing lines info
        if temp_dir:
            missing_file = temp_dir / "06_missing_lines.json"
            missing_file.write_text(
                json.dumps(all_missing, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # Sort missing lines by line number
        missing_sorted = sorted(all_missing.items())
        missing_line_nums = [num for num, _ in missing_sorted]
        missing_texts = [text for _, text in missing_sorted]

        # Create retry task
        retry_task = TranslationTask(
            chunk_index=1,
            text_lines=missing_texts,
            line_nums=missing_line_nums,
        )

        # Run retry
        retry_results = asyncio.run(
            translate_all_chunks_async(
                [retry_task], api_key, base_url, model, target_language, proxy, 1
            )
        )

        retry_result = retry_results[0]

        # Save retry result
        if temp_dir and retry_result.result:
            retry_file = temp_dir / "07_retry_result.txt"
            retry_lines = [
                f"[{num}] {line}"
                for num, line in zip(missing_line_nums, retry_result.result)
            ]
            retry_file.write_text("\n".join(retry_lines), encoding="utf-8")

        # Update all_translated_lines with successful retries
        retry_success = 0
        still_missing = retry_result.missing_nums or []

        if retry_result.result:
            for i, line_num in enumerate(missing_line_nums):
                if line_num not in still_missing:
                    # Successfully translated in retry
                    all_translated_lines[line_num - 1] = retry_result.result[i]
                    retry_success += 1

        if still_missing:
            print(
                f"[Translate] ✗ 重试后仍有 {len(still_missing)} 行缺失，使用原文: "
                f"{still_missing[:20]}{'...' if len(still_missing) > 20 else ''}"
            )
            # Save still missing
            if temp_dir:
                still_missing_file = temp_dir / "08_still_missing.json"
                still_missing_data = {num: all_missing.get(num, "") for num in still_missing}
                still_missing_file.write_text(
                    json.dumps(still_missing_data, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        if retry_success > 0:
            print(f"[Translate] ✓ 重试成功 {retry_success} 行")

    # Save final translations
    if temp_dir:
        final_trans_file = temp_dir / "09_final_translations.txt"
        numbered_final = [f"[{i+1}] {line}" for i, line in enumerate(all_translated_lines)]
        final_trans_file.write_text("\n".join(numbered_final), encoding="utf-8")

    # Reassemble: update entries with translated text (keep original timestamps)
    if len(all_translated_lines) != len(entries):
        print(
            f"[Translate] ⚠ 警告: 翻译后总行数不匹配 "
            f"({len(all_translated_lines)} vs {len(entries)})"
        )

    for i, entry in enumerate(entries):
        if i < len(all_translated_lines):
            entry.text = all_translated_lines[i].strip()

    # Save output SRT with original timestamps + translated text
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_srt = entries_to_srt(entries)
    output_path.write_text(output_srt, encoding="utf-8")

    print(f"[Translate] ✓ 已保存: {output_path}")
    print(f"[Translate] ===== 翻译完成 =====")
