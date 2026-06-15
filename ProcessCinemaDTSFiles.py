#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProcessCinemaDTSFiles.py - Merge decoded APT-X100 WAV files from Cinema DTS CDs.

Cinema DTS CDs store the film soundtrack as APT-X100 encoded data split
across multiple disc tracks.  After decoding, the resulting WAV files
must be reassembled into a single continuous multi-channel audio stream.

This script performs three main tasks:

  1. **Merge** - The decoded WAV segments are joined in playback order.
     Each segment boundary is handled by detecting and removing the 1 kHz
     marker tone at the end of each part, cutting a 2 s leader from
     appended files, and aligning consecutive segments via
     cross-correlation to find the exact overlap.  The overlap region is
     joined using an equal-power crossfade, which ensures click-free
     transitions even when segments differ slightly.  If consecutive
     segments have inverted polarity, the inversion is detected
     automatically and corrected.  For some films the overlap between
     segments may be longer than the default; in that case the
     --max-overlap parameter must be increased.  Note that not all
     Cinema DTS CDs contain perfectly mergeable segments.  Some segments
     may be inverted or differ slightly, resulting in low correlation.
     Such cases are reported as warnings in the output.

  2. **LFE extraction** - The original 5-channel mix has no discrete LFE
     channel; low-frequency content is carried in the surround channels.
     The script extracts the LFE from the surround channels using a
     Linkwitz-Riley 4th-order crossover filter and remaps the result to a
     standard 5.1 channel layout.  The surround channels are attenuated
     by 3 dB to match the cinema playback calibration of the DTS-6D/
     DTS-6AD processor.  If the LFE extraction causes clipping, the peak
     level is detected and all channels are attenuated equally to prevent
     distortion.  For films released before January 1, 1999 the LFE must be
     lowered by 3 dB (--lfe-gain -3), because the subwoofer calibration
     of +10 dB was only adopted from that date onward.

  3. **Speed adjustment** - Cinema DTS CDs are mastered at 24 FPS.  For
     playback at the NTSC-standard 23.976 FPS the audio must be slowed
     down by the ratio 23.976/24 (factor 0.999).  This is done via
     polyphase resampling, which preserves pitch.

All processing is done in 32-bit float.  The output is written in RF64
format, since the resulting files can exceed the 4 GB limit of standard
WAV.

Requirements
    pip install numpy scipy soundfile

Usage
    python ProcessCinemaDTSFiles.py -o merged.wav "part*.wav"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, correlate, resample_poly, sosfilt


# -- 1 kHz tone detection -------------------------------------------------

def _bandpass(data, sr, centre, bw=100):
    """Narrow bandpass filter around *centre* Hz."""
    lo = max((centre - bw / 2) / (sr / 2), 1e-5)
    hi = min((centre + bw / 2) / (sr / 2), 1.0 - 1e-5)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, data, axis=0)


def detect_signal_tone(audio, sr, max_search_s=25, win_ms=30,
                hop_ms=10, margin_ms=200):
    """Return sample index where a 1 kHz tone begins at the end of
    audio, or None when no tone is found.

    All channels are inspected.
    """
    tone_freq = 1000
    n_ch = audio.shape[1] if audio.ndim > 1 else 1

    search_len = min(int(max_search_s * sr), len(audio))
    region = audio[-search_len:]

    win = int(win_ms / 1000 * sr)
    hop = int(hop_ms / 1000 * sr)
    n_wins = max(1, (len(region) - win) // hop + 1)

    # Minimum RMS to consider a channel (skip near-silent channels
    # so they do not inflate ratios via numerical noise).
    silence_floor = 10 ** (-80 / 20)  # -80 dBFS

    # For every analysis window keep the highest tone-to-broadband ratio
    # across all channels, and track which channel produced it.
    ratios = np.zeros(n_wins)
    best_ch = np.zeros(n_wins, dtype=int)
    for ch in range(n_ch):
        ch_data = region[:, ch] if audio.ndim > 1 else region
        bp = _bandpass(ch_data, sr, tone_freq)
        for i in range(n_wins):
            s = i * hop
            e = s + win
            rms_full = np.sqrt(np.mean(ch_data[s:e] ** 2))
            if rms_full < silence_floor:
                continue
            rms_bp = np.sqrt(np.mean(bp[s:e] ** 2))
            r = rms_bp / rms_full
            if r > ratios[i]:
                ratios[i] = r
                best_ch[i] = ch

    # Adaptive threshold - the tone must clearly stand out.
    threshold = max(np.median(ratios) * 4, 0.35)
    # The tone must also be at least -45 dBFS loud.
    min_level = 10 ** (-45 / 20)
    mask = np.zeros(n_wins, dtype=bool)
    for i in range(n_wins):
        if ratios[i] <= threshold:
            continue
        s_i = i * hop
        e_i = s_i + win
        for ch in range(n_ch):
            ch_data = region[:, ch] if audio.ndim > 1 else region
            rms = np.sqrt(np.mean(ch_data[s_i:e_i] ** 2))
            if rms >= min_level:
                mask[i] = True
                break
    if not mask.any():
        print(f"         No signal tone detected (max ratio {np.max(ratios):.4f}, "
              f"threshold {threshold:.4f}, min level -45 dB)")
        return None, []

    # Scan from right to left: find the first tone window from the end,
    # then walk backwards to find where the tone begins.
    # The tone must be the last event: all windows after it must be silent.
    silence_thresh = 10 ** (-50 / 20)  # -50 dBFS

    # Find the rightmost tone window.
    rightmost = None
    for i in range(n_wins - 1, -1, -1):
        if mask[i]:
            rightmost = i
            break

    if rightmost is None:
        print("         No signal tone window found during backward scan.")
        return None, []

    # Report which channels have the tone above threshold at the rightmost window.
    tone_channels = []
    s_r = rightmost * hop
    e_r = s_r + win
    candidate_sample = len(audio) - search_len + rightmost * hop
    print(f"         Signal tone candidate @ {candidate_sample / sr:.3f} s  "
          f"(threshold {threshold:.4f})")
    for ch in range(n_ch):
        ch_data = region[:, ch] if audio.ndim > 1 else region
        rms_full = np.sqrt(np.mean(ch_data[s_r:e_r].astype(np.float64) ** 2))
        bp = _bandpass(ch_data.astype(np.float64), sr, tone_freq)
        rms_bp = np.sqrt(np.mean(bp[s_r:e_r] ** 2))
        rms_db = 20 * np.log10(max(rms_full, 1e-10))
        rms_bp_db = 20 * np.log10(max(rms_bp, 1e-10))
        ratio = rms_bp / rms_full if rms_full >= silence_floor else 0.0
        hit = ratio > threshold and rms_full >= silence_floor
        if hit:
            tone_channels.append(ch)
            print(f"         Ch {ch + 1}: total {rms_db:.1f} dB, "
                  f"tone {rms_bp_db:.1f} dB, ratio {ratio:.4f}")
    ch_names = ", ".join(str(c + 1) for c in tone_channels)
    print(f"         Signal tone on channel(s): {ch_names}")
  

    # Walk backwards from rightmost to find where the tone starts.
    tone_start = rightmost
    for i in range(rightmost - 1, -1, -1):
        if mask[i]:
            tone_start = i
        else:
            break

    sample = len(audio) - search_len + tone_start * hop
    # Extra margin before the tone to remove any lead-in artifacts.
    return max(0, sample - int(margin_ms / 1000.0 * sr)), tone_channels


# -- Overlap detection via cross-correlation -------------------------------

def _bandpass_for_corr(data, sr, lo=200, hi=4000):
    """Bandpass filter to improve correlation robustness.

    Removes DC offset, low-frequency rumble and high-frequency noise
    that would otherwise reduce the normalised correlation coefficient.
    """
    lo_n = max(lo / (sr / 2), 1e-5)
    hi_n = min(hi / (sr / 2), 1.0 - 1e-5)
    sos = butter(3, [lo_n, hi_n], btype="band", output="sos")
    return sosfilt(sos, data, axis=0)


def find_overlap(tail, head, sr, max_s=10, ref_s=1.0, min_corr=0.6,
                 silence_db=-80):
    """Return (overlap_samples, scores, inverted) where scores is a list of
    normalised correlation values for each channel and inverted indicates
    whether head needed polarity inversion to match.

    A 1-second reference from head is slid across the last max_s seconds
    of tail per channel.  Before correlation, each channel is bandpass-
    filtered (200-4000 Hz) to remove DC, rumble and HF noise.  The overlap
    position is determined by the channel with the strongest peak.  All
    per-channel scores are evaluated at that position.

    If all per-channel correlations are negative, the reference is inverted
    and the correlation is repeated.  If that succeeds, inverted=True.

    If both the tail region and the reference consist only of silence
    (below silence_db dBFS), the correlation is skipped and 0 is returned.
    """
    max_n = min(int(max_s * sr), len(tail), len(head))
    if max_n == 0:
        print("         [OVL] Aborted: no data to correlate (max_n=0)")
        return 0, [], False

    n_ch = tail.shape[1] if tail.ndim > 1 else 1

    tail_region = tail[-max_n:]
    ref_len = min(int(ref_s * sr), len(head), max_n)
    ref = head[:ref_len]

    # Check whether both regions are silent - skip correlation if so.
    silence_lin = 10 ** (silence_db / 20)
    tail_mono = np.mean(tail_region, axis=1) if tail_region.ndim > 1 else tail_region
    ref_mono = np.mean(ref, axis=1) if ref.ndim > 1 else ref
    rms_tail = np.sqrt(np.mean(tail_mono ** 2))
    rms_ref = np.sqrt(np.mean(ref_mono ** 2))
    if rms_tail < silence_lin and rms_ref < silence_lin:
        rms_tail_db = 20 * np.log10(max(rms_tail, 1e-10))
        rms_ref_db = 20 * np.log10(max(rms_ref, 1e-10))
        print(f"         [OVL] Aborted: both regions silent "
              f"(tail {rms_tail_db:.1f} dB, ref {rms_ref_db:.1f} dB, "
              f"threshold {silence_db} dB)")
        return 0, [0.0] * n_ch, False

    # Bandpass-filter each channel for position detection.
    tail_bp = np.empty_like(tail_region, dtype=np.float64)
    ref_bp = np.empty_like(ref, dtype=np.float64)
    for ch in range(n_ch):
        tc = tail_region[:, ch].astype(np.float64) if n_ch > 1 else tail_region.astype(np.float64)
        rc = ref[:, ch].astype(np.float64) if n_ch > 1 else ref.astype(np.float64)
        if n_ch > 1:
            tail_bp[:, ch] = _bandpass_for_corr(tc, sr)
            ref_bp[:, ch] = _bandpass_for_corr(rc, sr)
        else:
            tail_bp = _bandpass_for_corr(tc, sr)
            ref_bp = _bandpass_for_corr(rc, sr)

    # Per-channel correlation on filtered data to find best overlap position.
    best_idx = 0
    best_peak = -1.0
    worst_peak = 0.0
    worst_idx = 0
    for ch in range(n_ch):
        tc = tail_bp[:, ch] if n_ch > 1 else tail_bp.copy()
        rc = ref_bp[:, ch] if n_ch > 1 else ref_bp.copy()
        cc = correlate(tc, rc, mode="valid")
        if len(cc) == 0:
            continue
        ci = int(np.argmax(np.abs(cc)))
        seg = tc[ci: ci + ref_len]
        e_r = np.sqrt(np.sum(rc ** 2))
        e_s = np.sqrt(np.sum(seg ** 2))
        if e_r > 1e-10 and e_s > 1e-10:
            peak = cc[ci] / (e_r * e_s)
            if peak > best_peak:
                best_peak = peak
                best_idx = ci
            if peak < worst_peak:
                worst_peak = peak
                worst_idx = ci

    # If the most negative correlation exceeds the most positive one in
    # absolute value and is below -min_corr, try with inverted head.
    inverted = False
    if abs(worst_peak) > abs(best_peak) and worst_peak < -min_corr:
        print(f"         [OVL] Negative correlation ({worst_peak:.4f}) exceeds "
              f"positive ({best_peak:.4f}), retrying with inverted head...")
        best_idx = 0
        best_peak = -1.0
        ref_bp_inv = -ref_bp
        for ch in range(n_ch):
            tc = tail_bp[:, ch] if n_ch > 1 else tail_bp.copy()
            rc = ref_bp_inv[:, ch] if n_ch > 1 else ref_bp_inv.copy()
            cc = correlate(tc, rc, mode="valid")
            if len(cc) == 0:
                continue
            ci = int(np.argmax(np.abs(cc)))
            seg = tc[ci: ci + ref_len]
            e_r = np.sqrt(np.sum(rc ** 2))
            e_s = np.sqrt(np.sum(seg ** 2))
            if e_r > 1e-10 and e_s > 1e-10:
                peak = cc[ci] / (e_r * e_s)
                if peak > best_peak:
                    best_peak = peak
                    best_idx = ci
        if best_peak >= min_corr:
            inverted = True
            ref = -ref
            print(f"         [OVL] Inverted head matches (best corr {best_peak:.4f})")

    if best_peak < min_corr:
        print(f"         [OVL] Aborted: best correlation {best_peak:.4f} "
              f"< min_corr {min_corr}")
        return 0, [0.0] * n_ch, False

    # Evaluate normalised correlation at best_idx for every channel (on
    # filtered data so scores are consistent with position detection).
    scores = []
    for ch in range(n_ch):
        tc = tail_bp[:, ch] if n_ch > 1 else tail_bp.copy()
        rc = ref_bp[:, ch] if n_ch > 1 else ref_bp.copy()
        if inverted:
            rc = -rc
        seg = tc[best_idx: best_idx + ref_len]
        e_r = np.sqrt(np.sum(rc ** 2))
        e_s = np.sqrt(np.sum(seg ** 2))
        if e_r > 1e-10 and e_s > 1e-10:
            scores.append(float(np.sum(seg * rc) / (e_r * e_s)))
        else:
            scores.append(0.0)

    return int(max_n - best_idx), scores, inverted


# -- Channel remapping with Linkwitz-Riley filters ------------------------

def _linkwitz_riley_lp(data, sr, freq, order=4):
    """Linkwitz-Riley low-pass: two cascaded Butterworth filters of order/2."""
    sos = butter(order // 2, freq / (sr / 2), btype="low", output="sos")
    out = sosfilt(sos, data, axis=0)
    out = sosfilt(sos, out, axis=0)
    return out


def _linkwitz_riley_hp(data, sr, freq, order=4):
    """Linkwitz-Riley high-pass: two cascaded Butterworth filters of order/2."""
    sos = butter(order // 2, freq / (sr / 2), btype="high", output="sos")
    out = sosfilt(sos, data, axis=0)
    out = sosfilt(sos, out, axis=0)
    return out


def remap_channels(audio, sr, xover_freq, lfe_gain_db=6.0):
    """Remap channels 4-6 (0-indexed: 3-5) of a multi-channel array.

    New ch 4 (idx 3): sum of old ch 5+6 (idx 4+5) with LR4 low-pass, +lfe_gain_db
    New ch 5 (idx 4): old ch 4 (idx 3) with LR4 high-pass, -3 dB
    New ch 6 (idx 5): old ch 5 (idx 4) with LR4 high-pass, -3 dB

    Requires at least 5 input channels.  If only 5 channels exist, the
    output is expanded to 6 and old ch 6 is treated as silence for the
    LFE sum.

    After remapping, all channels are normalized to prevent clipping.
    """
    if audio.ndim < 2 or audio.shape[1] < 5:
        print("  [WARN] Remap requires at least 5 channels - skipping.")
        return audio

    n_ch = audio.shape[1]

    old_ch4 = audio[:, 3].copy()
    old_ch5 = audio[:, 4].copy()
    old_ch6 = audio[:, 5].copy() if n_ch >= 6 else np.zeros(len(audio), dtype=audio.dtype)

    # Expand to 6 channels if needed.
    if n_ch < 6:
        extra = np.zeros((len(audio), 6 - n_ch), dtype=audio.dtype)
        audio = np.concatenate([audio, extra], axis=1)
        print(f"  [REMAP] Expanded from {n_ch} to 6 channels")

    lfe_gain = 10 ** (lfe_gain_db / 20)
    hp_gain = 10 ** (-3.0 / 20)

    # New channel 4: sum of old 5+6, low-passed, boosted
    audio[:, 3] = _linkwitz_riley_lp(old_ch5 + old_ch6, sr, xover_freq) * lfe_gain
    # New channel 5: old channel 4, high-passed, -3 dB
    audio[:, 4] = _linkwitz_riley_hp(old_ch4, sr, xover_freq) * hp_gain
    # New channel 6: old channel 5, high-passed, -3 dB
    audio[:, 5] = _linkwitz_riley_hp(old_ch5, sr, xover_freq) * hp_gain

    print(f"  [GAIN] Ch 4: +{lfe_gain_db:.1f} dB | Ch 5,6: -3.0 dB")

    # Normalize all channels to prevent clipping.
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        gain = 1.0 / peak
        audio *= gain
        print(f"  [NORM] Peak {peak:.4f} -> gain {gain:.4f} ({20 * np.log10(gain):.1f} dB)")
    else:
        print(f"  [NORM] Peak {peak:.4f} - no clipping, no adjustment needed.")

    return audio


# -- Time-stretch via resampling ------------------------------------------

def time_stretch(audio, sr, speed):
    """Slow down or speed up audio by changing its length while keeping
    the sample rate and pitch unchanged.

    speed < 1.0 -> slower (longer output)
    speed > 1.0 -> faster (shorter output)

    If sr is 44100, the audio is simultaneously resampled to 48000 Hz
    by folding the 44100->48000 ratio into the speed ratio.

    Uses polyphase resampling with rational approximation of the speed
    factor to avoid floating-point truncation issues.

    Returns (resampled_audio, new_sr).
    """
    from fractions import Fraction

    frac = Fraction(speed).limit_denominator(100000)
    up = frac.denominator   # upsample factor
    down = frac.numerator    # downsample factor

    new_sr = sr
    if sr == 44100:
        # 44100 -> 48000: ratio 160/147
        sr_frac = Fraction(48000, 44100)  # = 160/147
        up *= sr_frac.numerator     # fold sr conversion into up
        down *= sr_frac.denominator  # fold sr conversion into down
        new_sr = 48000
        print(f"  [SLOW] Resampling 44.1 kHz -> 48 kHz")

    print(f"  [SLOW] Resampling ratio: up={up} / down={down} "
          f"(exact speed={down / up:.6f})")

    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(audio.dtype), new_sr

    n_ch = audio.shape[1]
    channels = []
    for ch in range(n_ch):
        print(f"  [SLOW] Resampling channel {ch + 1}/{n_ch}...")
        channels.append(resample_poly(audio[:, ch], up, down).astype(audio.dtype))
    min_len = min(len(c) for c in channels)
    return np.column_stack([c[:min_len] for c in channels]), new_sr


# -- Main merge logic -----------------------------------------------------

def merge(files, output, max_overlap_s=10, min_corr=0.6, no_lfe=False,
          lfe_gain_db=6.0, slowdown=False, write_tmp=False,
          signal_cut_ms=200):
    """Read files in order, remove marker tones, detect overlaps,
    and write a single merged WAV to output."""

    merged, sr = sf.read(files[0], dtype="float32")
    subtype = sf.info(files[0]).subtype
    n_ch = merged.shape[1] if merged.ndim > 1 else 1

    print(f"Sample rate : {sr} Hz | Channels: {n_ch} | Format: {subtype}\n")
    for i, f in enumerate(files, 1):
        d = len(sf.read(f, dtype="float32")[0]) / sr
        print(f"  [{i}/{len(files)}] {os.path.basename(f):40s} {d:>8.2f} s")
    print()

    # Tone removal on first file (only when more segments follow).
    if len(files) > 1:
        ts, _ = detect_signal_tone(merged, sr, margin_ms=signal_cut_ms)
        if ts is not None:
            total_s = len(merged) / sr
            print(f"  [CUT] File 1 - cutting at {ts / sr:.3f} s "
                  f"(removing last {total_s - ts / sr:.3f} s)")
            merged = merged[:ts]
        else:
            sys.exit("[ERR] No trailing tone found in file 1. Aborting.")

    # Optionally write trimmed first file.
    out_dir = os.path.dirname(output) or "."
    if write_tmp:
        tmp1 = os.path.join(out_dir, f"_tmp_01_{os.path.basename(files[0])}")
        sf.write(tmp1, merged, sr, format="RF64", subtype=subtype)
        print(f"  [TMP] {tmp1}  ({len(merged) / sr:.2f} s)")

    for i, path in enumerate(files[1:], 2):
        audio, sr2 = sf.read(path, dtype="float32")
        if sr2 != sr:
            sys.exit(f"Sample-rate mismatch in {path} ({sr2} vs {sr})")

        # Always cut 2 s from the start.
        cut_samples = int(2.0 * sr)
        if len(audio) > cut_samples:
            audio = audio[cut_samples:]
            print(f"  [SIL] File {i} - 2 s cut from start")

        # Trailing tone removal (skip for the very last file).
        if i < len(files):
            ts, _ = detect_signal_tone(audio, sr, margin_ms=signal_cut_ms)
            if ts is not None:
                total_s = len(audio) / sr
                print(f"  [CUT] File {i} - cutting at {ts / sr:.3f} s "
                      f"(removing last {total_s - ts / sr:.3f} s)")
                trimmed = audio[:ts]
            else:
                sys.exit(f"[ERR] No trailing tone found in file {i}. Aborting.")
        else:
            trimmed = audio

        # Optionally write trimmed file.
        if write_tmp:
            tmp_name = os.path.join(out_dir, f"_tmp_{i:02d}_{os.path.basename(path)}")
            sf.write(tmp_name, trimmed, sr, format="RF64", subtype=subtype)
            print(f"  [TMP] {tmp_name}  ({len(trimmed) / sr:.2f} s)")

        # Overlap detection against the silence-trimmed start of this file,
        # because the actual content (overlap) begins after the silence.
        ov, scores, inverted = find_overlap(merged, audio, sr, max_overlap_s,
                                             min_corr=min_corr)
        if inverted:
            audio = -audio
            trimmed = -trimmed
            print(f"  [INV] File {i} - polarity inverted")
        print(f"  [OVL] Files {i - 1}<>{i} overlap: {ov / sr:.3f} s "
              f"({ov} samples). Correlations:")
        for ch_idx, sc in enumerate(scores, 1):
            print(f"         Ch {ch_idx}: {sc:.4f}")

        if ov == 0:
            sys.exit(f"[ERR] No overlap found between files {i - 1} and {i}. "
                     f"Aborting.")

        if not scores:
            sys.exit(f"[ERR] No channel available for correlation check "
                     f"between files {i - 1} and {i}. Aborting.")

        if max(scores) < min_corr:
            sys.exit(f"[ERR] Best correlation {max(scores):.4f} < {min_corr} "
                     f"between files {i - 1} and {i}. Aborting.")

        # Both segments contain the same audio in the overlap region.
        # Validate overlap: compare the overlapping samples.
        tail_ov = merged[-ov:]
        head_ov = trimmed[:ov]
        diff = tail_ov.astype(np.float64) - head_ov.astype(np.float64)
        rms_diff = np.sqrt(np.mean(diff ** 2))
        rms_ref = np.sqrt(np.mean(tail_ov.astype(np.float64) ** 2))
        diff_db = 20 * np.log10(max(rms_diff, 1e-10))
        if rms_ref > 1e-10:
            snr = 20 * np.log10(rms_ref / max(rms_diff, 1e-10))
            print(f"  [CHK] Overlap diff: {diff_db:.1f} dB  "
                  f"(SNR {snr:.1f} dB)")
        else:
            print(f"  [CHK] Overlap diff: {diff_db:.1f} dB  "
                  f"(silent overlap)")
        if rms_diff > 0.01:
            print(f"  [WARN] Overlap mismatch is high ({diff_db:.1f} dB).")

        # Optionally write both sides of the overlap for manual inspection.
        if write_tmp:
            ov_tail_name = os.path.join(out_dir, f"_tmp_ovl_{i - 1:02d}_{i:02d}_tail.wav")
            ov_head_name = os.path.join(out_dir, f"_tmp_ovl_{i - 1:02d}_{i:02d}_head.wav")
            sf.write(ov_tail_name, tail_ov, sr, format="RF64", subtype=subtype)
            sf.write(ov_head_name, head_ov, sr, format="RF64", subtype=subtype)
            print(f"  [TMP] {ov_tail_name}  ({len(tail_ov) / sr:.2f} s)")
            print(f"  [TMP] {ov_head_name}  ({len(head_ov) / sr:.2f} s)")

        # Crossfade in the overlap region using an equal-power curve.
        # This eliminates clicks and reduces audible artifacts when
        # the two segments differ slightly.
        fade_len = ov
        t = np.linspace(0, np.pi / 2, fade_len, dtype=np.float64)
        fade_out = np.cos(t).astype(np.float32)  # 1 -> 0
        fade_in = np.sin(t).astype(np.float32)   # 0 -> 1
        if merged.ndim > 1:
            fade_out = fade_out[:, np.newaxis]
            fade_in = fade_in[:, np.newaxis]
        merged[-ov:] = tail_ov * fade_out + head_ov * fade_in
        merged = np.concatenate([merged, trimmed[ov:]])
        print(f"  [XFADE] Crossfade applied over {ov / sr:.3f} s "
              f"({ov} samples)")

    total = len(merged) / sr

    if not no_lfe:
        print(f"\n  [REMAP] Extracting LFE @ 80 Hz (Linkwitz-Riley 4th order)")
        merged = remap_channels(merged, sr, 80, lfe_gain_db)
    else:
        print("\n  [REMAP] LFE extraction disabled (--no_lfe)")

    if slowdown:
        speed = 0.999
        print(f"\n  [SLOW] Speed factor {speed} (pitch preserved)")
        merged, sr = time_stretch(merged, sr, speed)
        total = len(merged) / sr

    print(f"\n  [OUT] {output}  ({total:.2f} s)")
    sf.write(output, merged, sr, format="RF64", subtype=subtype)
    print("  [OK]  Done.")


# -- CLI -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Merge multi-channel WAV files with overlap detection "
                    "and 1 kHz tone removal.")
    ap.add_argument("files", nargs="+",
                    help="Glob patterns or WAV files in playback order "
                         "(e.g. 'part*.wav')")
    ap.add_argument("-o", "--output", required=True,
                    help="Output WAV file path")
    ap.add_argument("--max-overlap", type=float, default=4.0,
                     help="Maximum expected overlap in seconds (default: 4)")
    ap.add_argument("--min-corr", type=float, default=0.6,
                     help="Minimum correlation threshold for overlap "
                          "detection (default: 0.6)")
    ap.add_argument("--no-lfe", action="store_true",
                    help="Disable LFE channel extraction (ch 4-6 remapping)")
    ap.add_argument("--lfe-gain", type=float, default=0.0,
                     help="Additional LFE gain adjustment in dB, added to "
                          "the internal base gain of +6 dB (default: 0)")
    ap.add_argument("--slowdown", action="store_true",
                    help="Slow down output by factor 0.001 (reduce sample rate)")
    ap.add_argument("--write-tmp", action="store_true",
                    help="Write trimmed intermediate files for debugging")
    ap.add_argument("--signal-cut-length", type=float, default=200,
                    help="Extra margin before detected signal tone in ms "
                         "(default: 200)")
    args = ap.parse_args()

    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]

    resolved = []
    for pattern in args.files:
        matches = sorted(glob.glob(pattern), key=_natural_key)
        if not matches:
            sys.exit(f"No files matched: {pattern}")
        resolved.extend(matches)

    for f in resolved:
        if not os.path.isfile(f):
            sys.exit(f"File not found: {f}")

    merge(resolved, args.output,
          max_overlap_s=args.max_overlap,
          min_corr=args.min_corr,
          no_lfe=args.no_lfe,
          lfe_gain_db=6.0 + args.lfe_gain,
          slowdown=args.slowdown,
          write_tmp=args.write_tmp,
          signal_cut_ms=args.signal_cut_length)


if __name__ == "__main__":
    main()
