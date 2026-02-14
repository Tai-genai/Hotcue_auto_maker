#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio analysis, scoring, and grid calculation for hot cue optimization.

This module handles pure signal processing and has no XML knowledge.
It provides drop detection, bar snapping, and cue candidate scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import unquote, urlparse

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProcessingMode(Enum):
    SHIFT_ONLY = "shift_only"
    HYBRID = "hybrid"
    STANDALONE = "standalone"


class CuePlacementStrategy(Enum):
    RUN_UP_ONLY = "run_up_only"
    STRUCTURAL = "structural"


class CueRole(Enum):
    RUN_UP = "run_up"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for audio analysis."""

    mode: ProcessingMode = ProcessingMode.SHIFT_ONLY
    placement: CuePlacementStrategy = CuePlacementStrategy.RUN_UP_ONLY
    max_cues: int = 8
    min_cue_distance_bars: int = 4
    drop_score_onset_weight: float = 0.4
    drop_score_rms_weight: float = 0.35
    drop_score_spectral_weight: float = 0.25
    score_threshold: float = 0.3
    sr: int = 22050
    hop_length: int = 512
    phrase_length_bars: int = 8


def validate_analysis_config(config: AnalysisConfig) -> None:
    """Raise ValueError when analysis config values are invalid."""
    if not 1 <= config.max_cues <= 8:
        raise ValueError("max_cues は 1〜8 の範囲で指定してください。")
    if config.min_cue_distance_bars < 1:
        raise ValueError("min_cue_distance_bars は 1 以上を指定してください。")
    weights = [
        config.drop_score_onset_weight,
        config.drop_score_rms_weight,
        config.drop_score_spectral_weight,
    ]
    if any(w < 0 for w in weights):
        raise ValueError("重みは 0 以上を指定してください。")
    if sum(weights) <= 0:
        raise ValueError("重みの合計は 0 より大きくしてください。")
    if not 0.0 <= config.score_threshold <= 1.0:
        raise ValueError("score_threshold は 0.0〜1.0 の範囲で指定してください。")
    if config.sr <= 0:
        raise ValueError("sr は 0 より大きい値を指定してください。")
    if config.hop_length <= 0:
        raise ValueError("hop_length は 0 より大きい値を指定してください。")


@dataclass
class CueCandidate:
    """A candidate hot cue position with scoring information."""

    time_seconds: float = 0.0
    bar_snapped_time: float = 0.0
    role: CueRole = CueRole.RUN_UP
    drop_score: float = 0.0
    onset_strength: float = 0.0
    rms_delta: float = 0.0
    spectral_delta: float = 0.0
    source: str = "detected"
    energy_label: Optional[str] = None


@dataclass
class TrackAnalysisResult:
    """Result of analyzing a single track."""

    file_path: str = ""
    bpm: float = 0.0
    first_downbeat: float = 0.0
    bar_duration: float = 0.0
    candidates: list[CueCandidate] = field(default_factory=list)
    selected: list[CueCandidate] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Grid calculation (pure math, no external dependencies)
# ---------------------------------------------------------------------------

def compute_first_downbeat(inizio: float, bpm: float, battito: int) -> float:
    """Compute the time of the first downbeat (bar start).

    Args:
        inizio: TEMPO.Inizio value (seconds).
        bpm: Beats per minute.
        battito: TEMPO.Battito value (1-4, beat position within bar).

    Returns:
        First downbeat time in seconds (may be negative).
    """
    return inizio - (battito - 1) * (60.0 / bpm)


def compute_bar_duration(bpm: float) -> float:
    """Compute duration of one bar in seconds (assumes 4/4 time)."""
    return 4 * 60.0 / bpm


def snap_to_bar(time_seconds: float, first_downbeat: float, bar_duration: float) -> float:
    """Snap a time position to the nearest bar boundary.

    Returns:
        Snapped time, clamped to >= 0.0.
    """
    relative = time_seconds - first_downbeat
    bar_index = round(relative / bar_duration)
    snapped = first_downbeat + bar_index * bar_duration
    return max(0.0, snapped)


def snap_to_phrase(
    time_seconds: float,
    first_downbeat: float,
    bar_duration: float,
    phrase_bars: int = 8,
) -> float:
    """Snap a time position to the nearest phrase boundary.

    Args:
        phrase_bars: Number of bars per phrase (default 8).

    Returns:
        Snapped time, clamped to >= 0.0.
    """
    phrase_duration = bar_duration * phrase_bars
    relative = time_seconds - first_downbeat
    phrase_index = round(relative / phrase_duration)
    snapped = first_downbeat + phrase_index * phrase_duration
    return max(0.0, snapped)


# ---------------------------------------------------------------------------
# File path conversion
# ---------------------------------------------------------------------------

def location_to_filepath(location_url: Optional[str]) -> Optional[str]:
    """Convert a rekordbox Location URL to a local file path.

    Args:
        location_url: e.g. "file://localhost/Users/user/Desktop/music/track.mp3"

    Returns:
        Local file path string, or None if invalid.
    """
    if not location_url:
        return None
    parsed = urlparse(location_url)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    return path


# ---------------------------------------------------------------------------
# Audio feature extraction (librosa lazy import)
# ---------------------------------------------------------------------------

def load_audio(file_path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file using librosa (lazy import).

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If librosa is not installed or decode fails.
    """
    try:
        import librosa  # noqa: E402 — lazy import
    except ImportError:
        raise RuntimeError(
            "librosa がインストールされていません。"
            "pip install librosa soundfile を実行してください。"
        )

    import os
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {file_path}")

    try:
        y, sr_out = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"音声ファイルの読み込みに失敗しました: {e}")

    return y, sr_out


def compute_onset_strength(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Compute onset strength envelope."""
    import librosa
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)


def compute_rms_energy(y: np.ndarray, hop_length: int = 512) -> np.ndarray:
    """Compute RMS energy for each frame."""
    import librosa
    return librosa.feature.rms(y=y, hop_length=hop_length)[0]


def compute_spectral_contrast(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Compute mean spectral contrast across frequency bands."""
    import librosa
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    return np.mean(contrast, axis=0)


def compute_rms_derivative(rms: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame RMS energy change.

    Positive values indicate energy increase (drop candidates).
    """
    return np.concatenate([[0.0], np.diff(rms)])


def compute_spectral_derivative(spectral: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame spectral contrast change."""
    return np.concatenate([[0.0], np.diff(spectral)])


# ---------------------------------------------------------------------------
# Drop detection & scoring
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_drop_scores(
    onset: np.ndarray,
    rms_deriv: np.ndarray,
    spectral_deriv: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    """Compute composite drop score focusing on energy increase.

    Returns:
        1D array of scores in [0, 1] for each frame.
    """
    # Align lengths to shortest array
    min_len = min(len(onset), len(rms_deriv), len(spectral_deriv))
    onset = onset[:min_len]
    rms_deriv = rms_deriv[:min_len]
    spectral_deriv = spectral_deriv[:min_len]

    rms_pos = np.clip(rms_deriv, 0, None)
    spectral_pos = np.clip(spectral_deriv, 0, None)

    score = (
        config.drop_score_onset_weight * _normalize(onset)
        + config.drop_score_rms_weight * _normalize(rms_pos)
        + config.drop_score_spectral_weight * _normalize(spectral_pos)
    )
    return score


def compute_breakdown_scores(
    rms_deriv: np.ndarray,
    spectral_deriv: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    """Compute composite score focusing on energy decrease (breakdowns).

    Returns:
        1D array of scores in [0, 1] for each frame.
    """
    min_len = min(len(rms_deriv), len(spectral_deriv))
    rms_deriv = rms_deriv[:min_len]
    spectral_deriv = spectral_deriv[:min_len]

    rms_neg = np.clip(-rms_deriv, 0, None)
    spectral_neg = np.clip(-spectral_deriv, 0, None)

    weight_sum = config.drop_score_rms_weight + config.drop_score_spectral_weight
    if weight_sum <= 0:
        return np.zeros(min_len)

    score = (
        config.drop_score_rms_weight * _normalize(rms_neg)
        + config.drop_score_spectral_weight * _normalize(spectral_neg)
    ) / weight_sum

    return score


def detect_drop_peaks(
    drop_scores: np.ndarray,
    sr: int,
    hop_length: int,
    min_distance_seconds: float,
    threshold: float,
) -> list[tuple[float, float]]:
    """Detect peaks in the drop score array.

    Returns:
        List of (time_seconds, score) sorted by score descending.
    """
    from scipy.signal import find_peaks

    min_distance_frames = max(1, int(min_distance_seconds * sr / hop_length))

    peaks, properties = find_peaks(
        drop_scores,
        height=threshold,
        distance=min_distance_frames,
    )

    results = [
        (frame * hop_length / sr, float(drop_scores[frame]))
        for frame in peaks
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def detect_outro_start(
    rms: np.ndarray,
    sr: int,
    hop_length: int,
    total_duration: float,
) -> Optional[tuple[float, float]]:
    """Detect the start of the outro (sustained low energy at the end).

    Returns:
        (time_seconds, score) or None if no outro detected.
    """
    if len(rms) == 0:
        return None

    rms_mean = rms.mean()
    if rms_mean < 1e-10:
        return None

    # Look at the last 30% of the track
    start_frame = int(len(rms) * 0.7)
    threshold = rms_mean * 0.5

    for i in range(start_frame, len(rms)):
        if rms[i] <= threshold:
            # Check that energy doesn't recover after this point
            remaining = rms[i:]
            if remaining.max() <= rms_mean * 0.7:
                time_seconds = i * hop_length / sr
                # Score based on how much energy dropped
                score = float(1.0 - (rms[i] / rms_mean))
                return (time_seconds, min(score, 1.0))

    return None


# ---------------------------------------------------------------------------
# Cue candidate generation
# ---------------------------------------------------------------------------

def score_mik_cues(
    cue_times: list[float],
    drop_scores: np.ndarray,
    onset: np.ndarray,
    rms_deriv: np.ndarray,
    spectral_deriv: np.ndarray,
    sr: int,
    hop_length: int,
    config: AnalysisConfig,
) -> list[CueCandidate]:
    """Score existing MIK cues using audio analysis (hybrid mode).

    Args:
        cue_times: Bar-snapped cue positions in seconds.

    Returns:
        List of CueCandidate with scores assigned.
    """
    min_len = min(len(drop_scores), len(onset), len(rms_deriv), len(spectral_deriv))
    norm_onset = _normalize(onset[:min_len])
    norm_rms = _normalize(np.clip(rms_deriv[:min_len], 0, None))
    norm_spectral = _normalize(np.clip(spectral_deriv[:min_len], 0, None))

    candidates = []
    for cue_time in cue_times:
        frame = int(cue_time * sr / hop_length)
        frame = max(0, min(frame, min_len - 1))

        candidates.append(CueCandidate(
            time_seconds=cue_time,
            bar_snapped_time=cue_time,
            drop_score=float(drop_scores[frame]) if frame < len(drop_scores) else 0.0,
            onset_strength=float(norm_onset[frame]),
            rms_delta=float(norm_rms[frame]),
            spectral_delta=float(norm_spectral[frame]),
            source="mik",
        ))

    return candidates


def generate_standalone_candidates(
    drop_peaks: list[tuple[float, float]],
    onset: np.ndarray,
    rms_deriv: np.ndarray,
    spectral_deriv: np.ndarray,
    sr: int,
    hop_length: int,
    config: AnalysisConfig,
) -> list[CueCandidate]:
    """Generate CueCandidates from detected drop peaks (standalone mode).

    Returns:
        List of CueCandidate with source="detected".
    """
    min_len = min(len(onset), len(rms_deriv), len(spectral_deriv))
    norm_onset = _normalize(onset[:min_len])
    norm_rms = _normalize(np.clip(rms_deriv[:min_len], 0, None))
    norm_spectral = _normalize(np.clip(spectral_deriv[:min_len], 0, None))

    candidates = []
    for time_sec, score in drop_peaks:
        frame = int(time_sec * sr / hop_length)
        frame = max(0, min(frame, min_len - 1))

        candidates.append(CueCandidate(
            time_seconds=time_sec,
            bar_snapped_time=time_sec,  # will be snapped by caller
            drop_score=score,
            onset_strength=float(norm_onset[frame]),
            rms_delta=float(norm_rms[frame]),
            spectral_delta=float(norm_spectral[frame]),
            source="detected",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def deduplicate_candidates(
    candidates: list[CueCandidate],
    min_distance_seconds: float,
) -> list[CueCandidate]:
    """Remove candidates that are too close together (greedy, score-first).

    Higher-scored candidates are kept preferentially.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.drop_score, reverse=True)
    kept: list[CueCandidate] = []

    for cand in sorted_cands:
        if all(
            abs(cand.bar_snapped_time - k.bar_snapped_time) >= min_distance_seconds
            for k in kept
        ):
            kept.append(cand)

    return kept


def select_top_candidates(
    candidates: list[CueCandidate],
    max_cues: int,
    score_threshold: float,
) -> list[CueCandidate]:
    """Select top N candidates above threshold, sorted by time."""
    filtered = [c for c in candidates if c.drop_score >= score_threshold]
    filtered.sort(key=lambda c: c.drop_score, reverse=True)
    top = filtered[:max_cues]
    top.sort(key=lambda c: c.bar_snapped_time)
    return top


# ---------------------------------------------------------------------------
# Integrated entry point
# ---------------------------------------------------------------------------

def analyze_track(
    file_path: str,
    bpm: float,
    first_downbeat: float,
    bar_duration: float,
    mik_cue_times: Optional[list[float]],
    config: AnalysisConfig,
) -> TrackAnalysisResult:
    """Run the full analysis pipeline for a single track.

    Args:
        file_path: Path to the audio file.
        bpm: Track BPM.
        first_downbeat: First downbeat position in seconds.
        bar_duration: Duration of one bar in seconds.
        mik_cue_times: MIK cue positions (hybrid) or None (standalone).
        config: Analysis configuration.

    Returns:
        TrackAnalysisResult with candidates and selected cues.
    """
    try:
        y, sr = load_audio(file_path, config.sr)
        total_duration = len(y) / sr

        # Feature extraction
        onset = compute_onset_strength(y, sr, config.hop_length)
        rms = compute_rms_energy(y, config.hop_length)
        spectral = compute_spectral_contrast(y, sr, config.hop_length)
        rms_d = compute_rms_derivative(rms)
        spectral_d = compute_spectral_derivative(spectral)

        # Drop scores (energy increase direction)
        drop_scores = compute_drop_scores(onset, rms_d, spectral_d, config)

        min_dist_sec = bar_duration * config.min_cue_distance_bars

        if mik_cue_times is not None:
            # -- hybrid mode --
            snapped_times = [
                snap_to_bar(t, first_downbeat, bar_duration)
                for t in mik_cue_times
            ]
            candidates = score_mik_cues(
                snapped_times, drop_scores, onset, rms_d, spectral_d,
                sr, config.hop_length, config,
            )
            for c, st in zip(candidates, snapped_times):
                c.bar_snapped_time = st
                c.role = CueRole.RUN_UP
        else:
            # -- standalone mode --
            peaks = detect_drop_peaks(
                drop_scores, sr, config.hop_length,
                min_dist_sec, config.score_threshold,
            )
            candidates = generate_standalone_candidates(
                peaks, onset, rms_d, spectral_d,
                sr, config.hop_length, config,
            )
            for c in candidates:
                c.bar_snapped_time = snap_to_phrase(
                    c.time_seconds, first_downbeat, bar_duration,
                    config.phrase_length_bars,
                )
                c.role = CueRole.RUN_UP

        # -- structural mode: additional structure detection --
        if config.placement == CuePlacementStrategy.STRUCTURAL:
            # Drop cues (unshifted position markers)
            for c in list(candidates):
                if c.drop_score >= 0.5:
                    drop_cue = CueCandidate(
                        time_seconds=c.bar_snapped_time,
                        bar_snapped_time=c.bar_snapped_time,
                        role=CueRole.DROP,
                        drop_score=c.drop_score * 0.9,
                        onset_strength=c.onset_strength,
                        rms_delta=c.rms_delta,
                        spectral_delta=c.spectral_delta,
                        source=c.source,
                        energy_label="Drop",
                    )
                    candidates.append(drop_cue)

            # Breakdown cues (energy decrease points)
            bd_scores = compute_breakdown_scores(rms_d, spectral_d, config)
            bd_peaks = detect_drop_peaks(
                bd_scores, sr, config.hop_length,
                min_dist_sec, config.score_threshold,
            )
            for time_sec, score in bd_peaks[:3]:
                snapped = snap_to_bar(time_sec, first_downbeat, bar_duration)
                candidates.append(CueCandidate(
                    time_seconds=time_sec,
                    bar_snapped_time=snapped,
                    role=CueRole.BREAKDOWN,
                    drop_score=score * 0.8,
                    source="detected",
                    energy_label="Breakdown",
                ))

            # Outro cue
            outro = detect_outro_start(rms, sr, config.hop_length, total_duration)
            if outro:
                time_sec, score = outro
                snapped = snap_to_bar(time_sec, first_downbeat, bar_duration)
                candidates.append(CueCandidate(
                    time_seconds=time_sec,
                    bar_snapped_time=snapped,
                    role=CueRole.OUTRO,
                    drop_score=score * 0.7,
                    source="detected",
                    energy_label="Outro",
                ))

        # Selection
        deduped = deduplicate_candidates(candidates, min_dist_sec)
        selected = select_top_candidates(
            deduped, config.max_cues, config.score_threshold,
        )

        return TrackAnalysisResult(
            file_path=file_path,
            bpm=bpm,
            first_downbeat=first_downbeat,
            bar_duration=bar_duration,
            candidates=candidates,
            selected=selected,
        )

    except Exception as e:
        return TrackAnalysisResult(
            file_path=file_path,
            bpm=bpm,
            first_downbeat=first_downbeat,
            bar_duration=bar_duration,
            error=str(e),
        )
