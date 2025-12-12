import sys
from queue import Queue

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QTextEdit, QGroupBox, QStatusBar
)

from audio_player import AudioManager
from constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_GENERATION_LENGTH, HARP_SLOWDOWN, CHORD_SLOWDOWN, \
    INITIAL_PROMPT
from prompt_manager import PromptManager
from threads import GenerationThread, PlaybackThread
from ui.dialogues import Dialogues
from util import chord_token_to_human


class GenerationSignals(QObject):
    status_update = pyqtSignal(str)
    token_playing = pyqtSignal(int)  # index, token


class MusicGeneratorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Omnisong")
        self.setMinimumSize(1000, 700)

        self.audio = None
        self.is_playing = False
        self.current_token_index = 0
        self.currently_playing_tokens = []
        self.generation_queue = Queue()

        self.playback_thread = None
        self.generation_thread = None

        self.signals = GenerationSignals()
        self.signals.status_update.connect(self.update_status)
        self.signals.token_playing.connect(self.update_current_token)

        self.dialogues = Dialogues(self)
        self.prompt_manager = PromptManager()

        self.init_ui()

    # https://www.pythonguis.com/pyqt6-tutorial/
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        controls_panel = self.create_controls_panel()
        main_layout.addWidget(controls_panel)

        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to start!")

    def create_controls_panel(self):
        panel = QGroupBox("Generation Controls")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Temperature slider
        self.temp_label = QLabel(f"Temperature: {DEFAULT_TEMPERATURE:.2f}")
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setMinimum(10)
        self.temp_slider.setMaximum(200)
        self.temp_slider.setValue(int(DEFAULT_TEMPERATURE * 100))
        self.temp_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temp_slider.setTickInterval(20)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"Temperature: {v / 100:.2f}"))
        layout.addWidget(self.temp_label)
        layout.addWidget(self.temp_slider)

        # Top-p slider
        self.top_p_label = QLabel(f"Top-p: {DEFAULT_TOP_P:.2f}")
        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setMinimum(50)
        self.top_p_slider.setMaximum(100)
        self.top_p_slider.setValue(int(DEFAULT_TOP_P * 100))
        self.top_p_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.top_p_slider.setTickInterval(10)
        self.top_p_slider.valueChanged.connect(lambda v: self.top_p_label.setText(f"Top-p: {v / 100:.2f}"))
        layout.addWidget(self.top_p_label)
        layout.addWidget(self.top_p_slider)

        # Max length slider
        self.max_len_label = QLabel(f"Max Length: {MAX_GENERATION_LENGTH}")
        self.max_len_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_len_slider.setMinimum(32)
        self.max_len_slider.setMaximum(512)
        self.max_len_slider.setValue(MAX_GENERATION_LENGTH)
        self.max_len_slider.setSingleStep(16)
        self.max_len_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.max_len_slider.setTickInterval(64)
        self.max_len_slider.valueChanged.connect(lambda v: self.max_len_label.setText(f"Max Length: {v}"))
        layout.addWidget(self.max_len_label)
        layout.addWidget(self.max_len_slider)

        # Chord slowdown slider
        self.chord_slowdown_label = QLabel(f"Chord Slowdown: {CHORD_SLOWDOWN:.2f}x")
        self.chord_slowdown_slider = QSlider(Qt.Orientation.Horizontal)
        self.chord_slowdown_slider.setMinimum(50)
        self.chord_slowdown_slider.setMaximum(400)
        self.chord_slowdown_slider.setValue(int(CHORD_SLOWDOWN * 100))
        self.chord_slowdown_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chord_slowdown_slider.setTickInterval(50)
        self.chord_slowdown_slider.valueChanged.connect(self.on_chord_slowdown_changed)
        layout.addWidget(self.chord_slowdown_label)
        layout.addWidget(self.chord_slowdown_slider)

        # Harp slowdown slider
        self.harp_slowdown_label = QLabel(f"Harp Slowdown: {HARP_SLOWDOWN:.2f}x")
        self.harp_slowdown_slider = QSlider(Qt.Orientation.Horizontal)
        self.harp_slowdown_slider.setMinimum(50)
        self.harp_slowdown_slider.setMaximum(400)
        self.harp_slowdown_slider.setValue(int(HARP_SLOWDOWN * 100))
        self.harp_slowdown_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.harp_slowdown_slider.setTickInterval(50)
        self.harp_slowdown_slider.valueChanged.connect(self.on_harp_slowdown_changed)
        layout.addWidget(self.harp_slowdown_label)
        layout.addWidget(self.harp_slowdown_slider)

        layout.addSpacing(10)

        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(32)
        self.start_button.clicked.connect(self.start_generation)
        self.start_button.setDefault(True)  # makes it blue per apple design guidelines

        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(32)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        self.clear_button = QPushButton("Clear History")
        self.clear_button.setMinimumHeight(32)
        self.clear_button.clicked.connect(self.clear_history)
        layout.addWidget(self.clear_button)

        self.info_button = QPushButton("Information")
        self.info_button.setMinimumHeight(32)
        self.info_button.clicked.connect(self.dialogues.show_about_dialog)
        layout.addWidget(self.info_button)

        layout.addSpacing(10)

        prompt_group = QGroupBox("Initial Prompt")
        prompt_layout = QVBoxLayout()

        self.prompt_display = QTextEdit()
        self.prompt_display.setReadOnly(True)
        self.prompt_display.setMaximumHeight(100)
        self.prompt_display.setPlainText(chord_token_to_human(INITIAL_PROMPT))
        prompt_layout.addWidget(self.prompt_display)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        panel.setLayout(layout)
        panel.setMaximumWidth(400)
        return panel

    def create_right_panel(self):
        playing_group = QGroupBox("Currently Playing")
        playing_layout = QVBoxLayout()

        self.current_display = QTextEdit()
        self.current_display.setReadOnly(True)
        self.current_display.setMinimumHeight(150)
        playing_layout.addWidget(self.current_display)

        playing_group.setLayout(playing_layout)

        return playing_group

    def on_chord_slowdown_changed(self, value):
        """Update chord slowdown label and audio manager"""
        multiplier = value / 100.0
        self.chord_slowdown_label.setText(f"Chord Slowdown: {multiplier:.2f}x")
        if self.audio:
            self.audio.slow_down_chord = multiplier

    def on_harp_slowdown_changed(self, value):
        """Update harp slowdown label and audio manager"""
        multiplier = value / 100.0
        self.harp_slowdown_label.setText(f"Harp Slowdown: {multiplier:.2f}x")
        if self.audio:
            self.audio.slow_down_harp = multiplier

    def start_generation(self):
        """Start the infinite generation and playback loop"""
        if self.is_playing:
            return

        if self.audio is None:
            chord_slowdown = self.chord_slowdown_slider.value() / 100.0
            harp_slowdown = self.harp_slowdown_slider.value() / 100.0
            self.audio = AudioManager(
                window=self,
                slow_down_chord=chord_slowdown,
                slow_down_harp=harp_slowdown
            )

        self.is_playing = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.signals.status_update.emit("Starting generation...")

        # custom threads to allow stopping
        self.generation_thread = GenerationThread(self)
        self.playback_thread = PlaybackThread(self)

        self.generation_thread.start()
        self.playback_thread.start()

    def stop_generation(self):
        """Stop generation and playback"""
        self.is_playing = False

        if self.generation_thread is not None:
            self.generation_thread.stop()
        if self.playback_thread is not None:
            self.playback_thread.stop()
        if self.audio is not None:
            self.audio.stop_all()
            self.audio = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.signals.status_update.emit("Stopped")

    def clear_history(self):
        """Clear token history"""
        self.current_display.clear()
        self.prompt_manager.clear()
        self.prompt_display.setPlainText(chord_token_to_human(INITIAL_PROMPT))
        self.generation_queue = Queue()
        self.current_token_index = 0
        self.currently_playing_tokens = []
        self.signals.status_update.emit("History cleared")

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def update_current_token(self, index):
        """Highlight current token being played"""
        self.current_token_index = index

        tokens = self.currently_playing_tokens
        self.current_display.clear()

        cursor = self.current_display.textCursor()

        normal_format = QTextCharFormat()
        normal_format.setForeground(QColor("#FFFFFF"))

        highlight_format = QTextCharFormat()
        highlight_format.setForeground(QColor("#007AFF"))  # mac blue
        highlight_format.setFontWeight(QFont.Weight.Bold)

        for i, tok in enumerate(tokens):
            if i == index:
                cursor.setCharFormat(highlight_format)
            else:
                cursor.setCharFormat(normal_format)

            cursor.insertText(tok + " ")

    def closeEvent(self, event):
        """Handle window close"""
        self.stop_generation()
        if self.audio:
            self.audio.stop_all()
        event.accept()


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("Omnisong")
    app.setOrganizationName("Renan Martins")

    app.setStyle("macOS")

    window = MusicGeneratorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
