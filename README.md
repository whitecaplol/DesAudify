# DesAudify
A production-ready pipeline from audio to high-quality Desmos resynthesis

## Usage
`python desaudify_cli.py [-h] [--notes NOTES] [--poly POLY] [--fps FPS] [--start START] [--end END] [--min_mag MIN_MAG] input_file output_dir`

Options (yoinked from argparse and slightly modified):
- -h, --help         show help message and exit
- --notes NOTES      Maximum note budget (Depends on hardware. Usually anything above 700,000 notes will likely require sharding if outputted data exceeds 5 MiB)
- --poly POLY        Maximum concurrent notes per frame (Anything even as low as 32 works)
- --fps FPS          How many frames per second to target (Recommended 60, though you can try 120 or higher with diminishing returns)
- --start START      Start timestamp (in seconds as a float)
- --end END          End timestamp (in seconds as a float)
- --min_mag MIN_MAG  Minimum magnitude (not dB). Range from 0 to 1. Default is 0.0001 (which translates to -80 dB)

## Features
- High quality audio extraction using `ssqueezepy` to do Synchrosqueezed Multi-Resolution Short-Time Fourier Transform.
- Runs in vanilla Desmos
  - No need for GodMode, DesModder, or other desmos extensions!
- $O(1)$ indexing*
  - Using the new BigList 2.
- Dynamic polyphony
  - Means quiet/insignificant parts don't take up redundant space.
- Supports up to 8.7 million+ notes
    - Includes sharding script for larger files.
    - Exact limit depends on how much RAM your computer/browser can handle.
- Real-time 120+ fps resynthesis (varies by device)
    - Usually, to get better performance, you'll need to downsample.
    - 60 fps is expected on average, unless you're on a Chromebook.

## As seen in
whitecaplol's channel

### Examples
- Looping the rooms (pre-BigList 2): https://www.desmos.com/calculator/phsdrta4un
- Amethyst (Using BigList 2): https://www.desmos.com/calculator/nonkqd4rpo

## Notes
1. This is in beta, since the export format is still being optimized for faster performance.
    - Some refactoring had to be made in preparation for publishing. Some issues may arise that I have not accounted for; simply open up an issue and I'll look into it.
2. Currently requires a template and a bit of manual work. You can either write your own (for some reason) or just yoink this one: https://www.desmos.com/calculator/p8gb4nnc5y
3. `insert.js` helps to simplify importing things into Desmos.
