import sys
from os import path


def make_path(paths):
    """Returns the corrected path for a file considering whether this is a frozen (PyInstaller) or a regular script"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = path.abspath(path.dirname(__file__))
        path_to_dat = path.join(bundle_dir, paths)
        return path_to_dat
    else:
        return paths


def chord_token_to_human(txt):
    """Transforms chords like c_200 into C Major, am into A Minor, c7 into C Seventh, etc"""
    if txt.startswith('H'):
        parts = txt.split('_')
        return "Harp Note " + parts[0][1:]

    chord_map = {
        'c': 'C',
        'd': 'D',
        'e': 'E',
        'f': 'F',
        'g': 'G',
        'a': 'A',
        'b': 'B',
        'bb': 'B♭',
        'eb': 'E♭'
    }

    parts = txt.split('_')[0]
    root = ''
    quality = ''

    # Extract root note
    for note in chord_map.keys():
        if parts.startswith(note):
            root = chord_map[note]
            qual = parts.replace(note, '')
            if qual == '':
                quality = 'Major'
            elif qual == 'm':
                quality = 'Minor'
            else:
                quality = "Seventh"

    return f"{root} {quality}"


def count_chords(tokens):
    chords = 0

    for token in tokens:
        if not token.startswith("H"):
            chords += 1

    return chords
