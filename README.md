# DesAudify
A production-ready pipeline from audio to high-quality Desmos resynthesis

## Usage
`python desaudify_cli.py <args>`
- Add `-h` as an argument to see the exact list (too lazy to write them all down here for now)

## Features
- High quality audio extraction using `ssqueezepy` to do Synchrosqueezed Multi-Resolution Short-Time Fourier Transform.
- $O(1)$ indexing
- Dynamic polyphony
- Supports up to 8.7 million+ notes
    - Includes sharding script for larger files.
    - Exact limit depends on how much RAM your computer/browser can handle.
- Real-time 60+ fps resynthesis (varies by device)
    - Usually, to get better performance, you'll need to downsample.

## As seen in
whitecaplol's channel

### Examples
- Looping the rooms: https://www.desmos.com/calculator/phsdrta4un

## Notes
1. This is in beta, since the export format is still being optimized for faster performance.
    - Some refactoring had to be made in preparation for publishing. Some issues may arise that I have not accounted for; simply open up an issue and I'll look into it.
2. Currently requires a template and some manual work. You can either write your own (for some reason) or just yoink this one: https://www.desmos.com/calculator/dv7amzb7vs
3. `insert.js` and `autoshard.py` help to simplify importing things into Desmos.
