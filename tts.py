"""
Fish Speech V2 - Text-to-Speech Pipeline
TTS wrapper for batch processing with folder-based input/output.
"""

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
    Find reference audio and text in ref folder.

    Returns:
        (audio_path, text) tuple
    """
    ref_dir = Path(ref_dir)

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference folder not found: {ref_dir}")

    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(ref_dir.glob(f"*{ext}"))

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {ref_dir}")

    # If speaker specified, find matching file
    if speaker:
        matching = [f for f in audio_files if f.stem == speaker]
        if matching:
            audio_path = matching[0]
        else:
            log_warning(f"Speaker '{speaker}' not found, using first available")
            audio_path = audio_files[0]
    else:
        audio_path = audio_files[0]

    # Find corresponding text file
    text_path = audio_path.with_suffix(".txt")
    if text_path.exists():
        ref_text = text_path.read_text(encoding="utf-8").strip()
    else:
        log_warning(f"No text file found for {audio_path.name}, using filename as text")
        ref_text = audio_path.stem

    return audio_path, ref_text


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
    temperature: float = 0.8,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
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
):
    """Run TTS for a single text."""
    # Step 1: Encode reference audio
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
    ref_audio, ref_text = find_reference(ref_dir, args.speaker)
    log_info(f"Reference audio: {ref_audio}")
    log_info(f"Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")

    # Check if reference codes already exist (cache in ref folder with same name as audio)
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
