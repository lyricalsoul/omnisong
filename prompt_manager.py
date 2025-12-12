from constants import INITIAL_PROMPT, TOKEN_CUTOFF_FOR_GEN, HISTORY_TOKEN_CUTOFF


class PromptManager:
    def __init__(self):
        self.history = []
        self.total_tokens_gen = 0

    def append_to_history(self, new_tokens):
        self.history.extend(new_tokens)
        self.total_tokens_gen += len(new_tokens)

        if len(self.history) > HISTORY_TOKEN_CUTOFF:
            self.history = self.history[-HISTORY_TOKEN_CUTOFF:]

    def get_prompt(self):
        if self.total_tokens_gen == 0:
            return INITIAL_PROMPT
        else:
            return ' '.join(self.history[-TOKEN_CUTOFF_FOR_GEN:])

    def clear(self):
        self.history = []
        self.total_tokens_gen = 0
