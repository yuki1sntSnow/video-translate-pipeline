# Video Translation Pipeline

端到端的字幕翻译配音流水线：下载 → ASR → 翻译 → TTS → 合成，支持单/多说话人，可从任意阶段恢复。

## 功能亮点
- YouTube/本地视频下载，支持 cookies、代理、仅字幕模式
- ASR，支持说话人分离；可跳过下载直接识别本地文件
- LLM 异步分块翻译（OpenAI 兼容接口），保留行号，失败可重跑
- TTS，单/多说话人模式；参考音频统一放在 `ref/`
- FFmpeg 合成，可烧录字幕；音频模式 replace/mix/bgm，重叠可截断/变速/混合
- 数据、日志按时间戳落盘，`--start-from/--stop-after` 可自由组合流水

## 环境要求
- Python 3.10+
- FFmpeg 已在 PATH
- 推荐 NVIDIA GPU + 对应 CUDA 的 PyTorch（CPU 也可但速度慢）
- OpenAI 兼容 API Key（翻译）
- (可选) HuggingFace token（多说话人分离）
- TTS 模型检查点：`checkpoints/openaudio-s1-mini`

## 安装
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
# 先安装 whisperX, 否则有奇怪的依赖问题了, 以及 protobuf 版本问题不影响, 待解决
pip install -U whisperx
# CUDA 版本覆盖
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# 基础依赖
pip install -r requirements.txt
```

## 模型与资源
- checkpoints/openaudio-s1-mini
- Whisper 模型: 首次运行 ASR 时自动缓存到 `checkpoints/whisper/`
- 参考音频: `ref/{speaker}/{speaker}.wav` 和对应 `ref/{speaker}/{speaker}.txt`（5-15s 清晰语音）。多说话人时放置 `ref/SPEAKER_00/` 等或用映射

## 配置
```bash
cp config/env.example config/.env
```
关键项：
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME`
- 下载: `VIDEO_URL`、`SUB_LANGS`、`YTDLP_FORMAT`、`PROXY`、`COOKIES_FILE`
- ASR: `ASR_MODEL`、`ASR_DEVICE`、`ASR_BATCH_SIZE`、`MULTI_SPEAKER`、`HF_TOKEN`
- 翻译: `TARGET_LANGUAGE`、`TRANSLATE_CHUNK_SIZE`、`TRANSLATE_MAX_CONCURRENT`
- TTS: `TTS_SPEAKER`、`TTS_DAC_CHECKPOINT`、`TTS_T2S_CHECKPOINT`
- 合成: `AUDIO_MODE`、`BURN_SUBS`、`OVERLAP_STRATEGY`、`MAX_TTS_SPEED`
- 路径: `DATA_BASE_DIR`、`LOG_DIR`、`REF_DIR`、`CHECKPOINTS_DIR`

## 快速开始
下载+翻译+配音+合成 (单说话人):
```bash
python pipeline.py --url "https://www.youtube.com/watch?v=XXXX" --speaker cxk --target-language Chinese
```

多说话人 (ASR 分离 + speaker 映射):
```bash
python pipeline.py \
  --url "https://www.youtube.com/watch?v=XXXX" \
  --multi-speaker \
  --speaker-mapping "SPEAKER_00:host,SPEAKER_01:guest" \
  --speaker host
```

从中断处恢复 / 跳过阶段:
```bash
# 使用现有数据目录，从翻译开始
python pipeline.py --start-from translate --data-dir data/20251205_120000

# 只跑到某一阶段
python pipeline.py --url "..." --stop-after asr
```

使用本地文件跳过下载:
```bash
python pipeline.py --start-from asr --video /path/video.mp4 --srt /path/subs.srt
```

## 数据与日志
每次运行会生成时间戳目录 (默认 `data/`)，日志写到 `logs/`:
```
data/{timestamp}/
  {name}.mp4
  {name}.srt
  {name}_translated.srt
  {name}_translated.ass
  {name}_final.mp4
  temp/
    tts_input/*.txt
    tts_output/*.wav
    translate/*           # debug 时保留
    compose/*             # 合成中间产物
logs/pipeline_{timestamp}.log
```

## 常用 CLI 选项
- `--url`/`--video`/`--srt`：输入来源，支持跳过下载
- `--start-from`/`--stop-after`：控制阶段流水
- `--target-language`、`--chunk-size`、`--max-concurrent`：翻译配置
- `--multi-speaker`、`--speaker-mapping`、`--speaker`：TTS 说话人控制
- `--audio-mode` replace|mix|bgm、`--overlap-strategy` truncate|speed|hybrid|none、`--burn-subs`：合成策略
- `--subtitle-font`、`--subtitle-font-size` 等：ASS 样式
- `--debug`：保留翻译中间文件方便排查

完整参数参考 `python pipeline.py --help`。

## 目录结构
```
video-translate-pipeline/
├── pipeline.py               # 流水线入口
├── download.py               # yt-dlp 下载封装
├── asr.py                    # WhisperX ASR + 说话人分离
├── translate.py              # LLM 异步翻译
├── tts.py                    # Fish Speech TTS
├── compose.py                # FFmpeg 合成与字幕烧录
├── srt_utils.py              # 字幕解析/样式工具
├── dac_inference.py          # DAC 解码器封装
├── text2semantic_inference.py# 文本转语义模型封装
├── config/
│   ├── env.example
│   └── ...
├── checkpoints/              # 模型权重 (自备)
├── ref/                      # 参考音频
├── requirements.txt
└── LICENSE
```

## 许可
本项目遵循 GPL-3.0 License，详见 LICENSE。
