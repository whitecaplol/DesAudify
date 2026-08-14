import math
from pathlib import Path

import numpy as np
from ssqueezepy import ssq_stft

COLUMN_FREQUENCY = 0
COLUMN_START_TIME = 1
COLUMN_END_TIME = 2
COLUMN_MAGNITUDE = 3

def process_audio(audio_signal, sample_rate, target_frames_per_second=60, maximum_points_per_frame=192, max_notes=2600000, minimum_magnitude=0.0001):
    hop_length = int(sample_rate / target_frames_per_second)
    time_step_duration = hop_length / sample_rate
    dt_actual, fps_actual = time_step_duration * 1000.0, sample_rate / hop_length
    signal_len = len(audio_signal)

    # These bands aren't perfect but they're still relatively good. You can tune them if you want.
    bands = [
        {"fmin": 20.0,   "fmax": 250.0,  "win_len": 4096, "n_fft": 8192},
        {"fmin": 250.0,  "fmax": 2000.0, "win_len": 2048, "n_fft": 4096},
        {"fmin": 2000.0, "fmax": 8000.0, "win_len": 512,  "n_fft": 1024},
        {"fmin": 8000.0, "fmax": 20000.0,"win_len": 256,  "n_fft": 512}
    ]

    all_freqs, all_frames, all_mags = [], [], []
    max_mag = 1e-9

    for b in bands:
        win_len = min(max(b["win_len"], hop_length * 2), signal_len)
        if win_len < 2:
            continue

        n_fft = max(b["n_fft"], win_len)
        n_fft = 1 << (n_fft - 1).bit_length()

        Tx, _, ssq_freqs, *_ = ssq_stft(audio_signal, window="hann", n_fft=n_fft, win_len=win_len, hop_len=hop_length, fs=sample_rate)

        mags = np.abs(Tx) # type: ignore
        min_idx, max_idx = np.searchsorted(ssq_freqs, [b["fmin"], b["fmax"]]) # type: ignore
        if max_idx <= min_idx:
            continue

        is_peak = np.zeros_like(mags, dtype=bool)
        is_peak[1:-1, :] = (mags[1:-1, :] >= mags[:-2, :]) & (mags[1:-1, :] > mags[2:, :])
        is_peak[:max(1, min_idx), :] = is_peak[min(mags.shape[0] - 2, max_idx):, :] = False

        freq_idx, frame_idx = np.where(is_peak)
        if len(freq_idx) == 0:
            continue

        all_freqs.append(ssq_freqs[freq_idx]) # type: ignore
        all_frames.append(frame_idx)

        mags = mags[freq_idx, frame_idx]
        max_mag = max(max_mag, np.max(mags) or 1e-9)
        all_mags.append(mags)

    if not all_freqs:
        return np.zeros((0, 4)), dt_actual, fps_actual

    all_freqs, all_frames, all_mags = map(np.concatenate, (all_freqs, all_frames, all_mags))
    all_mags /= max(max_mag, 1)

    num_frames = all_frames.max() + 1
    orig_frame_sums = np.bincount(all_frames, weights=all_mags, minlength=num_frames)

    sound_threshold = np.where(all_mags > minimum_magnitude)
    all_freqs, all_frames, all_mags = all_freqs[sound_threshold], all_frames[sound_threshold], all_mags[sound_threshold]

    sort_idx = np.lexsort((-all_mags, all_frames))
    all_frames, all_freqs, all_mags = all_frames[sort_idx], all_freqs[sort_idx], all_mags[sort_idx]

    _, group_starts, group_counts = np.unique(all_frames, return_index=True, return_counts=True)
    intra_idx = np.arange(len(all_frames)) - np.repeat(group_starts, group_counts)
    keep = intra_idx < maximum_points_per_frame

    all_freqs, all_frames, all_mags = all_freqs[keep], all_frames[keep], all_mags[keep]

    if len(all_freqs) > max_notes:
        num_frames = all_frames.max() + 1
        frame_maxes = np.zeros(num_frames)
        np.maximum.at(frame_maxes, all_frames, all_mags)

        scores = all_mags / np.maximum(frame_maxes[all_frames], 1e-12)

        top_idx = np.sort(np.argpartition(scores, -max_notes)[-max_notes:])
        all_freqs, all_frames, all_mags = all_freqs[top_idx], all_frames[top_idx], all_mags[top_idx]

    remaining_frame_sums = np.bincount(all_frames, weights=all_mags, minlength=len(orig_frame_sums))
    scale_factors = np.where(remaining_frame_sums > 0, orig_frame_sums / (remaining_frame_sums + 1e-24), 1.0)
    all_mags *= scale_factors[all_frames]

    start_times = all_frames * time_step_duration
    end_times = start_times + time_step_duration

    return np.column_stack((all_freqs, start_times, end_times, all_mags)), dt_actual, fps_actual

def assign_notes_to_frames(mid_times, total_frames, start_ms, dt):
    f_idx = np.floor((mid_times - start_ms) / dt).astype(np.int64)
    valid = (f_idx >= 0) & (f_idx < total_frames)
    return f_idx[valid], np.where(valid)[0]

def pack_two_notes(a, b):
    if not a and not b:
        return 0
    if not a or not b:
        return int(a or b)
    a_int, b_int = int(max(a, b)), int(min(a, b))
    return (a_int - b_int) * 10000000 + b_int

def pack_frame_notes(temp_vals, n_packed):
    vals = np.sort(temp_vals)
    K = n_packed * 3

    if len(vals) % 2 != 0:
        vals = np.append(vals, 0)

    pairs = vals.reshape(-1, 2)

    if len(pairs) > K:
        pairs = pairs[:K]
    elif len(pairs) < K:
        padding = np.zeros((K - len(pairs), 2), dtype=temp_vals.dtype)
        pairs = np.vstack([pairs, padding])

    packed = [pack_two_notes(a, b) for a, b in pairs]
    return packed[0::3], packed[1::3], packed[2::3]

def generate_processing_schema(minmax_calls, maxpoly_calls, mxp_calls, mnp_calls, s_cond_parts, ct_inits, tones_parts):
    tones_def = (
        f"t_{{ones}}=\\left\\{{{','.join(tones_parts)}\\right\\}}"
        if len(tones_parts) > 1
        else f"t_{{ones}}={tones_parts[0]}"
    )

    lines = [
        f"m_{{inmax}}=\\left[{','.join(minmax_calls)}\\right]",
        f"m_{{axpoly}}=6\\max\\left({','.join(maxpoly_calls)}\\right)",
        f"M=\\left\\{{m_{{inmax}}.x\\le t_{0}<m_{{inmax}}.y,0\\right\\}}",
        f"m_{{axpitch}}=\\max\\left({','.join(mxp_calls)}\\right)",
        f"m_{{inpitch}}=\\min\\left({','.join(mnp_calls)}\\right)",
        f"s_{{upercond}}={','.join(s_cond_parts)}",
        *ct_inits,
        tones_def,
        "d_{uration}=\\max\\left(m_{{inmax}}.y\\right)",
    ]
    return "\n".join(lines)

def generate_desmos_schemas(pts, fps_actual, dt_actual, duration, time_range=None):
    if len(pts) == 0:
        return "", ""

    start_sec, end_sec = time_range or (0.0, duration) # time_range is used internally, but i'm too lazy to separate it out rn
    start_ms, end_ms = round(start_sec * 1000), round(end_sec * 1000)
    total_frames = math.ceil((end_ms - start_ms) / dt_actual)

    pool = pts.copy()
    pool[:, COLUMN_START_TIME] = np.round((pool[:, COLUMN_START_TIME] + start_sec) * 1000)
    pool[:, COLUMN_END_TIME] = np.round((pool[:, COLUMN_END_TIME] + start_sec) * 1000)

    flat_f, flat_n = assign_notes_to_frames(0.5 * (pool[:, COLUMN_START_TIME] + pool[:, COLUMN_END_TIME]), total_frames, start_ms, dt_actual)

    f_part = np.round(np.log(np.clip(pool[:, COLUMN_FREQUENCY], 20.0, 20000.0) / 20.0) / np.log(1000.0) * 9999).astype(np.int32)
    g_clip = np.clip(pool[:, COLUMN_MAGNITUDE], 0.0, 1.0)
    g_part = np.where(g_clip >= 0.0001, np.round(998 / 4 * (np.log10(g_clip) + 4) + 1), 0).astype(np.int32)
    pool_vals = f_part * 1000 + g_part

    valid = pool_vals[flat_n] > 0
    flat_f, flat_n = flat_f[valid], flat_n[valid]

    sort_idx = np.argsort(flat_f)
    flat_f, flat_n = flat_f[sort_idx], flat_n[sort_idx]

    unique_f, split_i = np.unique(flat_f, return_index=True)
    grouped_vals = np.split(pool_vals[flat_n], split_i[1:])

    segment_vals = [np.array([], dtype=np.int32) for _ in range(total_frames)]
    for f, g in zip(unique_f, grouped_vals):
        segment_vals[f] = g

    chunks, current_chunk, current_packed = [], [], 0
    for k, notes in enumerate(segment_vals):
        n_packed = (len(notes) + 5) // 6
        if current_chunk and (current_packed + n_packed > 10000 or len(current_chunk) >= 9998):
            chunks.append(current_chunk)
            current_chunk, current_packed = [], 0
        current_chunk.append((round(start_ms + k * dt_actual), notes, n_packed))
        current_packed += n_packed
    if current_chunk:
        chunks.append(current_chunk)

    data_lines, minmax, maxpoly, mxp, mnp, s_cond, ct_inits, tones = [], [], [], [], [], [], [], []
    for i, chunk in enumerate(chunks, 1):
        num_notes = [n_p for _, _, n_p in chunk]
        packed = [pack_frame_notes(notes, n_p) for _, notes, n_p in chunk]
        l1 = [x for p in packed for x in p[0]]
        l2 = [x for p in packed for x in p[1]]
        l3 = [x for p in packed for x in p[2]]

        p_elems = [chunk[0][0] - int(1000 * start_sec), int(fps_actual)] + num_notes
        data_lines.append(f"t_{{{i}}}=\\left(\\left[{','.join(map(str, l1))}\\right],\\left[{','.join(map(str, l2))}\\right],\\left[{','.join(map(str, l3))}\\right]\\right)")
        data_lines.append(f"p_{{{i}}}=\\left[{','.join(map(str, p_elems))}\\right]")

        minmax.append(f"f_{{minmax}}\\left(p_{{{i}}}\\right)")
        maxpoly.append(f"\\max\\left(p_{{{i}}}\\left[3...\\right]\\right)")
        mxp.append(f"g_{{mxp}}\\left(t_{{{i}}}\\right)")
        mnp.append(f"g_{{mnp}}\\left(t_{{{i}}}\\right)")
        s_cond.append(f"\\left\\{{M\\left[{i}\\right]=1:\\left(c_{{t{i}}}\\to t_{0}\\right),\\left\\{{c_{{t{i}}}\\ge 0:c_{{t{i}}}\\to-1\\right\\}}\\right\\}}")
        ct_inits.append(f"c_{{t{i}}}=0")
        tones.append(f"c_{{t{i}}}\\ge 0:t_{{h}}\\left(t_{{{i}}},i_{{i}}\\left(p_{{{i}}}\\right),p_{{{i}}}\\left[1\\right],p_{{{i}}}\\left[2\\right],c_{{t{i}}}\\right)")

    return "\n".join(data_lines), generate_processing_schema(
        minmax_calls=minmax, maxpoly_calls=maxpoly, mxp_calls=mxp, mnp_calls=mnp,
        s_cond_parts=s_cond, ct_inits=ct_inits, tones_parts=tones
    )

if __name__ == "__main__":
    import argparse
    import librosa

    parser = argparse.ArgumentParser(description="A production-ready (technically in beta) audio to Desmos pipeline")
    parser.add_argument("input_file", type=Path, help="Path to audio file")
    parser.add_argument("output_dir", type=Path, help="Path to output directory")
    parser.add_argument("--notes", type=int, help="Maximum note budget", default=2400000)
    parser.add_argument("--poly", type=int, help="Maximum concurrent notes per frame", default=64)
    parser.add_argument("--fps", type=int, help="How many frames per second to target", default=60)
    parser.add_argument("--start", type=float, help="Start timestamp", default=0)
    parser.add_argument("--end", type=float, help="End timestamp", default=-1)
    parser.add_argument("--min_mag", type=float, help="Minimum magnitude (not dB). Range from 0 to 1.", default=0.0001)

    args = parser.parse_args()

    print("Processing...")
    y, sr = librosa.load(args.input_file, sr=48000, offset=args.start, duration=None if args.end < 0 else args.end-args.start)
    pts, dt_actual, fps_actual = process_audio(y, sr, target_frames_per_second=args.fps, maximum_points_per_frame=args.poly, max_notes=args.notes, minimum_magnitude=args.min_mag)
    data, proc = generate_desmos_schemas(pts, fps_actual, dt_actual, len(y)/sr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "data_schema.txt", "w") as f:
        f.write(data)

    with open(args.output_dir / "processing_schema.txt", "w") as f:
        f.write(proc)

    print("Done!")
