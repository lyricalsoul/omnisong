# this file contains important constants used across the project.
# feel free to mess w them

# how many tokens should we use when continuing generation from a given prompt
TOKEN_CUTOFF_FOR_GEN = 20

# default temperature for generation
DEFAULT_TEMPERATURE = 0.9
# default top-p value for nucleus sampling during generation
DEFAULT_TOP_P = 0.9
# maximum length of generated sequences
MAX_GENERATION_LENGTH = 128
# slowdown factor for harp notes
HARP_SLOWDOWN = 1.5
# slowdown factor for chords
CHORD_SLOWDOWN = 1.2
# initial prompt
INITIAL_PROMPT = "c_200"
# amount of tokens to be kept in the history array
HISTORY_TOKEN_CUTOFF = 1024
