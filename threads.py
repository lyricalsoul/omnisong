import threading
import time
from queue import Queue

from infer import generate
from util import chord_token_to_human, count_chords


class GenerationThread(threading.Thread):
    def __init__(self, window):
        super().__init__(daemon=True)
        self.window = window
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.window.generation_queue = Queue()

    def stopped(self):
        return self._stop_event.is_set()

    def push_to_queue(self, item):
        if self.stopped():
            raise StopIteration
        self.window.generation_queue.put(item)

    def run(self):
        while not self.stopped():
            try:
                prompt = self.window.prompt_manager.get_prompt()
                self.window.signals.status_update.emit(f"Generating from {count_chords(prompt)} chords...")

                temperature = self.window.temp_slider.value() / 100.0
                top_p = self.window.top_p_slider.value() / 100.0
                max_len = self.window.max_len_slider.value()

                sequence = generate(prompt, max_len=max_len, temperature=temperature,
                                    top_p=top_p, debug=False)

                # adds the whole seq as a string to the queue
                self.push_to_queue(sequence)
                time.sleep(1)

            except Exception as e:
                self.window.signals.status_update.emit(f"Generation error: {str(e)}")
                time.sleep(1)


class PlaybackThread(threading.Thread):
    def __init__(self, window):
        super().__init__(daemon=True)
        self.window = window
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        # play the initial prompt first
        initial_prompt = self.window.prompt_manager.get_prompt()
        self.window.audio.interpret_sequence(initial_prompt)

        while not self.stopped():
            try:
                # wait for next seq
                if not self.window.generation_queue.empty():
                    sequence = self.window.generation_queue.get()
                    tokens = sequence.split()

                    self.window.currently_playing_tokens = tokens
                    self.window.current_token_index = 0
                    self.window.prompt_manager.append_to_history(tokens)

                    def update_display_progress():
                        for i, token in enumerate(tokens):
                            if not self.window.is_playing or self.stopped():
                                raise StopIteration
                            self.window.signals.token_playing.emit(i)
                            self.window.signals.status_update.emit(f"Playing {chord_token_to_human(token)}")

                            # estimate duration based on token type
                            if '_' in token:
                                parts = token.split('_')
                                duration = int(parts[1])
                                if parts[0].startswith('H'):
                                    time.sleep(duration * self.window.audio.slow_down_harp / 1000.0)
                                else:
                                    time.sleep(duration * self.window.audio.slow_down_chord / 1000.0)

                    # another thread so we can update display progress without blocking audio playback
                    display_thread = threading.Thread(target=update_display_progress, daemon=True)
                    display_thread.start()

                    self.window.audio.interpret_sequence(sequence)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.window.signals.status_update.emit(f"Playback error: {str(e)}")
                time.sleep(1)
