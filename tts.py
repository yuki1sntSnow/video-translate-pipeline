"""Fish Speech TTS wrapper for batch processing."""

import os
import platform
import sys
import time
from pathlib import Path

# Add parent directory to path for relative imports when run directly
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import torchaudio

# Conditional import for Windows
if platform.system() == "Windows":
    import soundfile as sf


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac"}


log_info = lambda msg: print(f"[INFO] {msg}")
log_warning = lambda msg: print(f"[WARNING] {msg}")
log_error = lambda msg: print(f"[ERROR] {msg}")


def log_step(step: int, total: int, msg: str):
    print(f"\n{'='*60}\n[Step {step}/{total}] {msg}\n{'='*60}")


def log_task(idx: int, total: int, msg: str):
    print(f"\n{'-'*60}\n[Task {idx}/{total}] {msg}\n{'-'*60}")


# ============= Audio I/O =============


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    if platform.system() == "Windows":
        audio_data, sr = sf.read(str(path), dtype="float32")
        audio = torch.from_numpy(
            audio_data.T if audio_data.ndim > 1 else audio_data[None, :]
        )
    else:
        audio, sr = torchaudio.load(str(path))

    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    return audio


def save_audio(audio: torch.Tensor, path: Path, sample_rate: int):
    """Save audio to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        sf.write(str(path), audio.numpy(), sample_rate)
    else:
        torchaudio.save(
            str(path), audio.unsqueeze(0) if audio.dim() == 1 else audio, sample_rate
        )


# ============= Lazy Model Loading =============

_dac_model = None
_t2s_model = None
_t2s_decode_fn = None


def get_dac_model(checkpoint_path: str, device: str):
    """Lazy load DAC model."""
    global _dac_model
    if _dac_model is None:
        # Import here to avoid circular imports and speed up --help

        from dac_inference import load_model

        _dac_model = load_model(checkpoint_path, device=device)
    return _dac_model


def get_t2s_model(checkpoint_path: str, device: str, precision, compile: bool):
    """Lazy load Text2Semantic model."""
    global _t2s_model, _t2s_decode_fn
    if _t2s_model is None:
        from text2semantic_inference import init_model

        _t2s_model, _t2s_decode_fn = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            _t2s_model.setup_caches(
                max_batch_size=1,
                max_seq_len=_t2s_model.config.max_seq_len,
                dtype=next(_t2s_model.parameters()).dtype,
            )
    return _t2s_model, _t2s_decode_fn


# ============= Pipeline Steps =============


def find_reference(ref_dir: Path, speaker: str | None = None):
    """
    Find reference audio, text and optional codes in ref folder.

    Folder structure: ref/{speaker_name}/{speaker_name}.wav, .txt, .npy
    Example: ref/cxk/cxk.wav, ref/cxk/cxk.txt, ref/cxk/cxk.npy

    Args:
        ref_dir: Reference directory (e.g., "ref")
        speaker: Speaker name (e.g., "cxk", "SPEAKER_00")

    Returns:
        (audio_path, text, codes_path or None) tuple
    """
    ref_dir = Path(ref_dir)

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference folder not found: {ref_dir}")

    audio_path = None
    ref_text = None
    codes_path = None

    # Structure: ref/{speaker}/{speaker}.wav
    if speaker:
        speaker_dir = ref_dir / speaker
        if speaker_dir.exists() and speaker_dir.is_dir():
            # Look for audio files in speaker dir
            audio_files = []
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(speaker_dir.glob(f"*{ext}"))

            if audio_files:
                audio_path = audio_files[0]
                text_path = audio_path.with_suffix(".txt")
                npy_path = audio_path.with_suffix(".npy")

                if text_path.exists():
                    ref_text = text_path.read_text(encoding="utf-8").strip()
                if npy_path.exists():
                    codes_path = npy_path

    # If speaker not found, try to find any available speaker
    if audio_path is None:
        available = get_available_speakers(ref_dir)
        if not available:
            raise FileNotFoundError(f"No speaker folders with audio found in {ref_dir}")

        if speaker:
            log_warning(f"Speaker '{speaker}' not found, using first available: {available[0]}")

        # Use first available speaker
        first_speaker = available[0]
        speaker_dir = ref_dir / first_speaker

        for ext in AUDIO_EXTENSIONS:
            audio_files = list(speaker_dir.glob(f"*{ext}"))
            if audio_files:
                audio_path = audio_files[0]
                text_path = audio_path.with_suffix(".txt")
                npy_path = audio_path.with_suffix(".npy")

                if text_path.exists():
                    ref_text = text_path.read_text(encoding="utf-8").strip()
                if npy_path.exists():
                    codes_path = npy_path
                break

    if audio_path is None:
        raise FileNotFoundError(f"No audio files found in {ref_dir}")

    if ref_text is None:
        log_warning(f"No text file found for {audio_path.name}, using filename as text")
        ref_text = audio_path.stem

    return audio_path, ref_text, codes_path
def get_available_speakers(ref_dir: Path) -> list[str]:
    """
    Get list of available speaker names from ref folder.

    Checks for subdirectories containing audio files.
    """
    ref_dir = Path(ref_dir)
    speakers = []

    if not ref_dir.exists():
        return speakers

    for subdir in ref_dir.iterdir():
        if subdir.is_dir():
            # Check if subdir contains any audio files
            has_audio = any(subdir.glob(f"*{ext}") for ext in AUDIO_EXTENSIONS)
            if has_audio:
                speakers.append(subdir.name)

    return sorted(speakers)


def find_input_texts(input_dir: Path):
    """
    Find all text files in input folder.

    Returns:
        List of (filename_stem, text) tuples
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    texts = []
    for txt_file in sorted(input_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            texts.append((txt_file.stem, text))

    return texts


@torch.no_grad()
def encode_audio(
    audio_path: Path, output_path: Path, dac_checkpoint: str, device: str
) -> Path:
    """Step 1: Encode audio to VQ codes."""
    model = get_dac_model(dac_checkpoint, device)

    audio = load_audio(audio_path, model.sample_rate)
    audios = audio[None].to(device)

    log_info(f"Loaded audio: {audio_path} ({audios.shape[2] / model.sample_rate:.2f}s)")

    audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
    indices, _ = model.encode(audios, audio_lengths)

    if indices.ndim == 3:
        indices = indices[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, indices.cpu().numpy())
    log_info(f"Saved codes to: {npy_path} (shape: {indices.shape})")

    return npy_path


@torch.no_grad()
def generate_codes(
    text: str,
    prompt_text: str,
    prompt_codes_path: Path,
    output_path: Path,
    t2s_checkpoint: str,
    device: str,
    precision,
    compile: bool,
    temperature: float = 0.3,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    seed: int = 42,
) -> Path:
    """Step 2: Generate VQ codes from text."""
    from text2semantic_inference import generate_long

    model, decode_one_token = get_t2s_model(t2s_checkpoint, device, precision, compile)

    prompt_codes = torch.from_numpy(np.load(prompt_codes_path))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    log_info(f"Generating codes for: {text[:50]}{'...' if len(text) > 50 else ''}")

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        prompt_text=[prompt_text],
        prompt_tokens=[prompt_codes],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    codes_path = output_path.with_suffix(".npy")

    for response in generator:
        if response.action == "sample":
            np.save(codes_path, response.codes.cpu().numpy())
            log_info(f"Saved codes to: {codes_path}")

    return codes_path


@torch.no_grad()
def decode_codes(
    codes_path: Path, output_path: Path, dac_checkpoint: str, device: str
) -> Path:
    """Step 3: Decode VQ codes to audio."""
    model = get_dac_model(dac_checkpoint, device)

    indices = np.load(codes_path)
    indices = torch.from_numpy(indices).to(device).long()

    if indices.ndim == 2:
        indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)
    else:
        raise ValueError(f"Expected 2D indices, got {indices.ndim}D")

    fake_audios, _ = model.decode(indices, indices_lens)
    audio_time = fake_audios.shape[-1] / model.sample_rate

    log_info(f"Generated audio: {audio_time:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fake_audio = fake_audios[0, 0].float().cpu()
    save_audio(fake_audio, output_path, model.sample_rate)
    log_info(f"Saved audio to: {output_path}")

    return output_path


# ============= Single TTS =============


def run_single_tts(
    text: str,
    ref_audio: Path,
    ref_text: str,
    output_path: Path,
    temp_dir: Path,
    dac_checkpoint: str,
    t2s_checkpoint: str,
    device: str,
    precision,
    compile: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    ref_codes_path: Path | None = None,
):
    """Run TTS for a single text."""
    # Step 1: Encode reference audio (or use cached codes)
    if ref_codes_path and ref_codes_path.exists():
        log_step(1, 3, "Using cached reference codes")
        log_info(f"Found cached codes: {ref_codes_path}")
    else:
        log_step(1, 3, "Encoding reference audio")
        ref_codes_path = encode_audio(
            ref_audio, temp_dir / "ref_codes", dac_checkpoint, device
        )

    # Step 2: Generate codes from text
    log_step(2, 3, "Generating VQ codes from text")
    generated_codes_path = generate_codes(
        text=text,
        prompt_text=ref_text,
        prompt_codes_path=ref_codes_path,
        output_path=temp_dir / "generated",
        t2s_checkpoint=t2s_checkpoint,
        device=device,
        precision=precision,
        compile=compile,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )

    # Step 3: Decode to audio
    log_step(3, 3, "Decoding codes to audio")
    result = decode_codes(generated_codes_path, output_path, dac_checkpoint, device)

    return result


# ============= Batch Processing =============


def run_batch_tts(args):
    """Run batch TTS processing with folder structure."""
    input_dir = Path(args.input_dir)
    ref_dir = Path(args.ref_dir)
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)

    device = args.device
    dac_checkpoint = args.dac_checkpoint
    t2s_checkpoint = args.t2s_checkpoint
    precision = torch.half if args.half else torch.bfloat16

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Find reference
    log_info(f"Looking for reference in: {ref_dir}")
    ref_audio, ref_text, cached_codes = find_reference(ref_dir, args.speaker)
    log_info(f"Reference audio: {ref_audio}")
    log_info(f"Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")

    # Check if reference codes already exist
    if cached_codes and cached_codes.exists():
        log_step(1, 2, "Loading cached reference codes")
        log_info(f"Found cached codes: {cached_codes}")
        ref_codes_path = cached_codes
    else:
        # Check legacy location (same name as audio)
        ref_codes_path = ref_audio.with_suffix(".npy")
        if ref_codes_path.exists():
            log_step(1, 2, "Loading cached reference codes")
            log_info(f"Found cached codes: {ref_codes_path}")
        else:
            log_step(1, 2, "Encoding reference audio")
            ref_codes_path = encode_audio(
                ref_audio,
                ref_audio.with_suffix(
                    ""
                ),  # Output path without extension, .npy will be added
                dac_checkpoint,
                device,
            )

    # Find all input texts
    log_step(2, 2, "Processing input texts")
    input_texts = find_input_texts(input_dir)

    if not input_texts:
        log_error(f"No .txt files found in {input_dir}")
        return

    log_info(f"Found {len(input_texts)} text file(s) to process")

    # Process each text
    total_time = 0
    success_count = 0

    for idx, (name, text) in enumerate(input_texts, 1):
        log_task(idx, len(input_texts), f"{name}.txt")

        try:
            t0 = time.time()

            # Generate codes
            generated_codes_path = generate_codes(
                text=text,
                prompt_text=ref_text,
                prompt_codes_path=ref_codes_path,
                output_path=temp_dir / f"{name}_codes",
                t2s_checkpoint=t2s_checkpoint,
                device=device,
                precision=precision,
                compile=args.compile,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed + idx,  # Different seed for each
            )

            # Decode to audio
            output_path = decode_codes(
                generated_codes_path, output_dir / f"{name}.wav", dac_checkpoint, device
            )

            elapsed = time.time() - t0
            total_time += elapsed
            success_count += 1
            log_info(f"Completed in {elapsed:.2f}s")

        except Exception as e:
            log_error(f"Failed to process {name}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch TTS Complete!")
    print(f"   Processed: {success_count}/{len(input_texts)} files")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Output folder: {output_dir}")
    print(f"{'='*60}\n")


def run_multi_speaker_tts(
    input_dir: Path,
    ref_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    dac_checkpoint: str,
    t2s_checkpoint: str,
    device: str = "cuda",
    half: bool = False,
    compile: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    seed: int = 42,
    speaker_mapping: dict[str, str] | None = None,
    fallback_speaker: str | None = None,
):
    """
    Run multi-speaker TTS processing.

    Input structure (from entries_to_txt_files_by_speaker):
        input_dir/
            SPEAKER_00/
                0001.txt
                0003.txt
            SPEAKER_01/
                0002.txt
                0004.txt

    Ref structure:
        ref_dir/
            {speaker_name}/
                {speaker_name}.wav
                {speaker_name}.txt
                {speaker_name}.npy (optional, will be generated)

    Args:
        speaker_mapping: Optional dict to map ASR speaker names to ref folder names.
                        e.g. {"SPEAKER_00": "host", "SPEAKER_01": "guest"}
        fallback_speaker: Deprecated, will use first successfully used speaker as fallback.
    """
    input_dir = Path(input_dir)
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)
    temp_dir = Path(temp_dir)

    precision = torch.half if half else torch.bfloat16

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Find all speaker subdirs in input
    speaker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not speaker_dirs:
        log_error(f"No speaker subdirectories found in {input_dir}")
        return

    input_speakers = [d.name for d in speaker_dirs]
    log_info(f"Found {len(speaker_dirs)} speaker(s) in input: {input_speakers}")

    # Find available ref speakers (only folder structure: ref/{speaker}/)
    available_refs = get_available_speakers(ref_dir)
    if not available_refs:
        log_error(f"No speaker folders found in {ref_dir}")
        log_error(f"Expected structure: {ref_dir}/{{speaker_name}}/{{speaker_name}}.wav")
        return

    log_info(f"Available ref speakers: {available_refs}")

    # Check which speakers have matching refs
    matched_speakers = []
    missing_speakers = []
    for speaker_name in input_speakers:
        ref_speaker = (
            speaker_mapping.get(speaker_name, speaker_name)
            if speaker_mapping
            else speaker_name
        )
        if ref_speaker in available_refs:
            matched_speakers.append((speaker_name, ref_speaker))
        else:
            missing_speakers.append((speaker_name, ref_speaker))

    # Fallback will be the first successfully matched speaker (determined during processing)
    # For now, use first available ref as initial fallback
    fallback_ref = available_refs[0] if available_refs else None
    first_used_ref = None  # Will be set to first successfully used ref

    if missing_speakers:
        log_warning(f"⚠ {len(missing_speakers)} speaker(s) 缺少对应的 ref 音频")
        for input_name, ref_name in missing_speakers:
            log_warning(f"  - {input_name} -> {ref_name} (missing)")
        log_info(f"缺失的 speaker 将使用第一个成功使用的 ref 音频")

    # Cache for reference codes per speaker
    speaker_ref_cache: dict[str, tuple[str, Path]] = {}  # {ref_speaker: (ref_text, codes_path)}

    total_time = 0
    total_success = 0
    total_files = 0

    for speaker_dir in sorted(speaker_dirs):
        speaker_name = speaker_dir.name

        # Map speaker name to ref folder if mapping provided
        ref_speaker = (
            speaker_mapping.get(speaker_name, speaker_name)
            if speaker_mapping
            else speaker_name
        )

        # Check if ref exists, otherwise use fallback (first used ref)
        use_fallback = False
        if ref_speaker not in available_refs:
            if first_used_ref:
                log_warning(f"Speaker [{speaker_name}] ref [{ref_speaker}] not found, using first used ref [{first_used_ref}]")
                ref_speaker = first_used_ref
                use_fallback = True
            else:
                # No ref used yet, use first available
                log_warning(f"Speaker [{speaker_name}] ref [{ref_speaker}] not found, using [{fallback_ref}]")
                ref_speaker = fallback_ref
                use_fallback = True

        print(f"\n{'#'*60}")
        if use_fallback:
            print(f"# Processing Speaker: {speaker_name} (ref: {ref_speaker} [FALLBACK])")
        else:
            print(f"# Processing Speaker: {speaker_name} (ref: {ref_speaker})")
        print(f"{'#'*60}")

        # Find reference for this speaker
        try:
            ref_audio, ref_text, cached_codes = find_reference(ref_dir, ref_speaker)
        except FileNotFoundError as e:
            log_error(f"Reference not found for speaker {ref_speaker}: {e}")
            continue

        # Track first successfully used ref for fallback
        if first_used_ref is None:
            first_used_ref = ref_speaker
            log_info(f"First used ref speaker set to: {first_used_ref}")

        log_info(f"Reference audio: {ref_audio}")
        log_info(
            f"Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}"
        )

        # Get or create reference codes (cache by ref_speaker, not input speaker_name)
        if ref_speaker in speaker_ref_cache:
            ref_text, ref_codes_path = speaker_ref_cache[ref_speaker]
        elif cached_codes and cached_codes.exists():
            log_info(f"Using cached reference codes: {cached_codes}")
            ref_codes_path = cached_codes
            speaker_ref_cache[ref_speaker] = (ref_text, ref_codes_path)
        else:
            # Generate codes
            ref_codes_path = ref_audio.with_suffix(".npy")
            if ref_codes_path.exists():
                log_info(f"Using cached reference codes: {ref_codes_path}")
            else:
                log_info("Encoding reference audio...")
                ref_codes_path = encode_audio(
                    ref_audio,
                    ref_audio.with_suffix(""),
                    dac_checkpoint,
                    device,
                )
            speaker_ref_cache[ref_speaker] = (ref_text, ref_codes_path)

        # Find all input texts for this speaker
        input_texts = find_input_texts(speaker_dir)
        if not input_texts:
            log_warning(f"No .txt files found for speaker {speaker_name}")
            continue

        log_info(f"Found {len(input_texts)} text file(s) for {speaker_name}")
        total_files += len(input_texts)

        # Process each text
        for idx, (name, text) in enumerate(input_texts, 1):
            log_task(idx, len(input_texts), f"{speaker_name}/{name}.txt")

            try:
                t0 = time.time()

                # Generate codes
                generated_codes_path = generate_codes(
                    text=text,
                    prompt_text=ref_text,
                    prompt_codes_path=ref_codes_path,
                    output_path=temp_dir / speaker_name / f"{name}_codes",
                    t2s_checkpoint=t2s_checkpoint,
                    device=device,
                    precision=precision,
                    compile=compile,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=seed + int(name),  # Use file index as seed offset
                )

                # Decode to audio - output maintains original file numbering for merging
                output_path = decode_codes(
                    generated_codes_path,
                    output_dir / f"{name}.wav",  # Same filename, flat output
                    dac_checkpoint,
                    device,
                )

                elapsed = time.time() - t0
                total_time += elapsed
                total_success += 1
                log_info(f"Completed in {elapsed:.2f}s")

            except Exception as e:
                log_error(f"Failed to process {speaker_name}/{name}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Multi-Speaker TTS Complete!")
    print(f"   Speakers: {len(speaker_dirs)}")
    print(f"   Processed: {total_success}/{total_files} files")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Output folder: {output_dir}")
    print(f"{'='*60}\n")
