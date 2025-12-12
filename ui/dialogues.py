from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QWidget


class Dialogues(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def show_about_dialog(self):
        about_dialog = QMessageBox(self)
        about_dialog.setWindowTitle("About Omnisong")
        about_dialog.setTextFormat(Qt.TextFormat.RichText)

        about_text = """
                <h2>Omnisong</h2>
                <p><i>An AI experiment on music generation, authorship, and meaning</i></p>
    
                <p>
                Omnisong explores what it music means, if it means anything at all.
                By training an AI model on common chord progressions, techniques such as arpeggios, and more, we can
                continue from a given musical idea, and infinitely generate new compositions.
                </p>
    
                <p>
                Refer to the Description.pdf file for a detailed explanation of the project, including dataset generation, model architecture, training process, and more.
                </p>
    
                <h3>Quick Control Overview:</h3>
                <ul>
                    <li><b>Temperature:</b> higher = more random, lower = more "conservative"</li>
                    <li><b>Top-p:</b> higher = more options, lower = fewer options</li>
                    <li><b>Max Length:</b> higher = longer generation, will take more time for changes to reflect</li>
                    <li><b>Slowdown:</b> higher = slower playback if you want to hear details</li>
                </ul>
    
                <p><i>This project was built for MUS 1051 - Fundamentals of Music. Thanks Dr. Francis for letting me have fun with this! You're awesome!</i></p>
                """

        about_dialog.setText(about_text)
        about_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)

        about_dialog.exec()
