# if it starts with "H" it's a harp, else it's a chord
def is_harp(tok):
    if not isinstance(tok, str): return False
    return tok.startswith("H")


def is_chord(tok):
    if not isinstance(tok, str): return False
    return not tok.startswith("H")


def build_allowed_mask(id_to_tok, history):
    mask = [0] * len(id_to_tok)

    # Convert integer history to token strings if necessary
    history_tok = [id_to_tok[i] for i in history]

    if not history_tok:
        # First token must be a chord
        for idx in id_to_tok:
            if is_chord(id_to_tok[idx]):
                mask[idx] = 1
        return mask

    last_tok = history_tok[-1]

    # Count consecutive chords
    num_consecutive_chords = 0
    for t in reversed(history_tok):
        if is_chord(t):
            num_consecutive_chords += 1
        else:
            break

    # Count consecutive harps
    num_consecutive_harps = 0
    for t in reversed(history_tok):
        if is_harp(t):
            num_consecutive_harps += 1
        else:
            break

    if is_chord(last_tok):
        if num_consecutive_chords > 2:
            # must be followed by a harp
            for idx in id_to_tok:
                if is_harp(id_to_tok[idx]):
                    mask[idx] = 1
        else:
            # can be a chord or a harp
            for idx in id_to_tok:
                if is_chord(id_to_tok[idx]) or is_harp(id_to_tok[idx]):
                    mask[idx] = 1

    elif is_harp(last_tok):
        if num_consecutive_harps >= 12:
            # after 12 harps, we force a chord so it doesnt sound too weird or repetitive
            for idx in id_to_tok:
                if is_chord(id_to_tok[idx]):
                    mask[idx] = 1
        else:
            # can be whatever since we havent hit the limit
            for idx in id_to_tok:
                if is_harp(id_to_tok[idx]) or is_chord(id_to_tok[idx]):
                    mask[idx] = 1

    # if there's still nothing we just allow everything so it doesnt get stuck
    if sum(mask) == 0:
        mask = [1] * len(id_to_tok)

    return mask
