# CinemaDTSFileProcessor

Merge decoded APT-X100 WAV files from Cinema DTS CDs into a single continuous multi-channel audio file.

Cinema DTS CDs store the film soundtrack as APT-X100 encoded data split across multiple disc tracks. After decoding, the resulting WAV files must be reassembled into a single continuous multi-channel audio stream. This script automates that process.

## Features

### 1. Merge

The decoded WAV segments are joined in playback order. Each segment boundary is handled by:

- Detecting and removing the 1 kHz marker tone at the end of each part.
- Cutting a 2 s leader from appended files.
- Aligning consecutive segments via cross-correlation to find the exact overlap.

### 2. LFE Extraction

The original 5-channel mix has no discrete LFE channel; low-frequency content is carried in the surround channels. The script extracts the LFE using a Linkwitz-Riley 4th-order crossover filter at 80 Hz and remaps the result to a standard 5.1 channel layout:

| Channel | Source |
|---------|--------|
| 1-3     | Unchanged (L, C, R) |
| 4 (LFE) | Low-pass of surround channels 5+6 |
| 5 (Ls)  | Old channel 4, high-passed, -3 dB |
| 6 (Rs)  | Old channel 5, high-passed, -3 dB |

### 3. Speed Adjustment

Cinema DTS CDs are mastered at 24 FPS. For playback at the NTSC-standard 23.976 FPS the audio must be slowed down by the ratio 23.976/24 (factor 0.999). This is done via polyphase resampling, which preserves pitch. If the source sample rate is 44.1 kHz, it is simultaneously resampled to 48 kHz.

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [SoundFile](https://python-soundfile.readthedocs.io/)

```bash
pip install numpy scipy soundfile
```

## Usage

```bash
python ProcessCinemaDTSFiles.py -o merged.wav "part*.wav"
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output` | Output WAV file path (required) | - |
| `--max-overlap` | Maximum expected overlap in seconds | 4 |
| `--no-lfe` | Disable LFE channel extraction (ch 4-6 remapping) | off |
| `--lfe-gain` | Gain for the extracted LFE channel in dB | 6 |
| `--slowdown` | Slow down output by factor 0.999 (24 -> 23.976 FPS) | off |
| `--write-tmp` | Write trimmed intermediate files for debugging | off |
| `--signal-cut-length` | Extra margin before detected signal tone in ms | 200 |

### Examples

Merge all parts with LFE extraction and speed adjustment:

```bash
python ProcessCinemaDTSFiles.py -o movie_soundtrack.wav --slowdown "part*.wav"
```

Merge without LFE extraction:

```bash
python ProcessCinemaDTSFiles.py -o merged.wav --no-lfe "part*.wav"
```

Merge with custom LFE gain and debug output:

```bash
python ProcessCinemaDTSFiles.py -o merged.wav --lfe-gain 10 --write-tmp "part*.wav"
```

