import math
from itertools import accumulate
from pathlib import Path
from collections.abc import Sequence

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

def pack_frame_notes(temp_vals):
    vals = np.sort(temp_vals)
    if len(vals) % 2 != 0:
        vals = np.append(vals, 0)
    pairs = vals.reshape(-1, 2)
    return [pack_two_notes(a, b) for a, b in pairs]

def get_sanitized_string(string: str) -> str:
    return f'{string[0]}_{{{string[1:]}}}'

def format_desmos_list(var_name: str, plain_list: Sequence, max_mega_fragments: int | None = None, max_fragments: int | None = None, max_list_size: int = 10000) -> str:
    if (max_mega_fragments is not None and max_mega_fragments < 1): max_mega_fragments = None
    if (max_fragments is not None and max_fragments < 1): max_fragments = None
    elif (max_fragments is not None):
        max_fragments = min(1024, max_fragments)
    output = ''

    current_matrix_position = 0
    current_mega_fragments = 0
    current_global_fragments = 0

    def getTrailingCharacter():
        if (current_matrix_position >= 1024): return ''
        return '&' if current_matrix_position % 32 > 0 else '\\\\'

    def close_matrix():
        nonlocal output, current_matrix_position
        while (0 < current_matrix_position < 1024):
            output += getTrailingCharacter()
            current_matrix_position += 1
        if (output[-14:] != r'\right)\right]'):
            output += r'\end{bmatrix}'
            output += fr'\left[1+\operatorname{{mod}}\left(\operatorname{{floor}}\left(\frac{{\operatorname{{ceil}}\left(\frac{{k}}{{{max_list_size}}}\right)-1}}{{32}}\right),32\right);1+\operatorname{{mod}}\left(\operatorname{{ceil}}\left(\frac{{k}}{{{max_list_size}}}\right)-1,32\right)\right]\left[1+\operatorname{{mod}}\left(k-1,{max_list_size}\right)\right]'

    for i in range(0, len(plain_list), max_list_size):
        if (current_matrix_position == 0 == current_mega_fragments):
            current_global_fragments += 1
        if (current_mega_fragments == 0):
            current_mega_fragments += 1
            output += rf'{get_sanitized_string(f'{var_name}{current_global_fragments}')}\left(k\right)=\left\{{'
        if (current_matrix_position == 0):
            if (current_mega_fragments > 1):
                output += ','
            total_num_fragments = current_mega_fragments if max_mega_fragments is None else (current_global_fragments - 1) * max_mega_fragments + current_mega_fragments
            output += rf'{min(1 + (total_num_fragments - 1) * 1024 * max_list_size, len(plain_list))}\le k \le{min((total_num_fragments) * 1024 * max_list_size, len(plain_list))}:\begin{{bmatrix}}'
        output += rf'\left[{','.join(map(str, plain_list[i:i+max_list_size]))}\right]'
        current_matrix_position += 1
        if (i+max_list_size >= len(plain_list)):
            output += r'.\operatorname{join}\left(\operatorname{repeat}\left(0,k\right)\right)'.replace("k", str(i+max_list_size - len(plain_list)))
            break
        if (current_matrix_position < 1024 and (max_fragments is None or current_matrix_position < max_fragments)):
            output += getTrailingCharacter()
            continue
        close_matrix()
        current_matrix_position = 0
        current_mega_fragments += 1
        if (max_mega_fragments is None or current_mega_fragments < max_mega_fragments): continue
        current_mega_fragments = 0
        output += r'\right\}''\n'

    close_matrix()
    if output[-1] != '\n':
        output += r'\right\}''\n'

    output += f'{get_sanitized_string(var_name)}\\left(l\\right)='

    if max_mega_fragments is not None and current_global_fragments > 1:
        output += '\\left\\{'
        for i in range(current_global_fragments):
            if i > 0:
                output += ','
            output += f'{min(1 + i * max_list_size * (max_fragments if max_fragments is not None else 1024) * max_mega_fragments, len(plain_list))} \\le k \\le {min((1 + i) * max_list_size * (max_fragments if max_fragments is not None else 1024) * max_mega_fragments, len(plain_list))}:{get_sanitized_string(f'{var_name}{i+1}')}\\left(k\\right)'
        output += '\\right\\}'
    else:
        output += f'{get_sanitized_string(f'{var_name}1')}\\left(k\\right)'
    output += r'\operatorname{for}k=l'

    return output

def generate_desmos_schemas(pts, fps_actual, dt_actual, duration, time_range=None, max_lists_per_fragment: int | None = None, max_lists_per_mega_fragment: int | None = None):
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

    everything = []
    for k, notes in enumerate(segment_vals):
        n_packed = (len(notes) + 1) // 2
        everything.append((round(start_ms + k * dt_actual), notes if n_packed > 0 else np.array([0], dtype=np.int32), max(1, n_packed)))

    tones = format_desmos_list("tonedata", ([note for frame in (pack_frame_notes(notes) for _, notes, _ in everything) for note in frame]), max_fragments=max_lists_per_fragment, max_mega_fragments=max_lists_per_mega_fragment)
    timings = format_desmos_list("tonetimings", list(accumulate([1] + [n_p for _, _, n_p in everything])), max_fragments=max_lists_per_fragment, max_mega_fragments=max_lists_per_mega_fragment)

    return f"{tones}\n{timings}", (rf'a_{{udindex}}=\operatorname{{floor}}\left(\left(t_{{0}}-{everything[0][0] - int(1000 * start_sec)}\right)\cdot 0.001\cdot {int(fps_actual)}\right)''\n'f'a_{{udioduration}}={everything[-1][0] - int(1000 * start_sec)}')

if __name__ == "__main__":
    import argparse

    import librosa

    parser = argparse.ArgumentParser(description="A production-ready audio to Desmos pipeline")
    parser.add_argument("input_file", type=Path, help="Path to audio file")
    parser.add_argument("output_dir", type=Path, help="Path to output directory")
    parser.add_argument("--notes", type=int, help="Maximum note budget", default=2400000)
    parser.add_argument("--poly", type=int, help="Maximum concurrent notes per frame", default=64)
    parser.add_argument("--fps", type=int, help="How many frames per second to target", default=60)
    parser.add_argument("--start", type=float, help="Start timestamp", default=0)
    parser.add_argument("--end", type=float, help="End timestamp", default=-1)
    parser.add_argument("--max_lists_per_fragment", type=int, help="Maximum sublists per BigList fragment. Any integer less than 1 means no limit.", default=42)
    parser.add_argument("--max_lists_per_mega_fragment", type=int, help="Maximum sublists per BigList megafragment. Any integer less than 1 means no limit.", default=1)
    parser.add_argument("--min_mag", type=float, help="Minimum magnitude (not dB). Range from 0 to 1.", default=0.0001)

    args = parser.parse_args()

    print("Processing...")
    y, sr = librosa.load(args.input_file, sr=48000, offset=args.start, duration=None if args.end < 0 else args.end-args.start)
    pts, dt_actual, fps_actual = process_audio(y, sr, target_frames_per_second=args.fps, maximum_points_per_frame=args.poly, max_notes=args.notes, minimum_magnitude=args.min_mag)
    data, proc = generate_desmos_schemas(pts, fps_actual, dt_actual, len(y)/sr, max_lists_per_fragment=args.max_lists_per_fragment, max_lists_per_mega_fragment=args.max_lists_per_mega_fragment)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "data_schema.txt", "w") as f:
        f.write(data)

    with open(args.output_dir / "processing_schema.txt", "w") as f:
        f.write(proc)

    print("Done!")
