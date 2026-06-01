from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QFrame, QProgressDialog, QComboBox
)
from PyQt5.QtGui import QFont, QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import sys
import cv2
from ultralytics import YOLO
from deep_translator import GoogleTranslator


# ------------------ YOLO Detection Thread ------------------
class YOLOThread(QThread):
    frame_updated = pyqtSignal(QImage, str)
    finished = pyqtSignal()

    def __init__(self, weights_path="best.pt", conf=0.25):
        super().__init__()
        self.weights_path = weights_path
        self.conf = conf
        self.running = True

    def run(self):
        model = YOLO(self.weights_path)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            self.finished.emit()
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform YOLO prediction
            results = model.predict(source=frame, conf=self.conf, save=False, verbose=False)
            annotated_frame = results[0].plot()

            # Extract detected word (if any)
            names = results[0].names
            boxes = results[0].boxes
            detected_word = ""
            if len(boxes) > 0:
                cls_id = int(boxes.cls[0])
                detected_word = names[cls_id]

            # Convert frame to QImage for PyQt
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_img = qt_img.scaled(640, 480, Qt.KeepAspectRatio)

            # Emit frame and detected word
            self.frame_updated.emit(scaled_img, detected_word)

        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False
        self.wait()


# ------------------ Detection Window ------------------
class DetectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖐️ Word Detection")
        self.setGeometry(350, 150, 800, 600)
        self.setStyleSheet("background-color: #f7faff;")
        self.detected_words = []
        self.translator = GoogleTranslator()
        self.current_word = ""

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.word_label = QLabel("Detected Word: ")
        self.word_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.word_label.setAlignment(Qt.AlignCenter)
        self.word_label.setStyleSheet("color: #2b4c7e;")
        layout.addWidget(self.word_label)

        # Sentence display
        self.sentence_label = QLabel("Formed Sentence: ")
        self.sentence_label.setFont(QFont("Segoe UI", 14))
        self.sentence_label.setAlignment(Qt.AlignCenter)
        self.sentence_label.setStyleSheet("color: #3b5998;")
        layout.addWidget(self.sentence_label)

        # Translate button
        self.translate_btn = QPushButton("🌐 Translate")
        self.translate_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #229954; }
        """)
        self.translate_btn.clicked.connect(self.show_dropdown)
        layout.addWidget(self.translate_btn, alignment=Qt.AlignCenter)

        # Language dropdown (initially hidden)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Hindi", "Kannada"])
        self.language_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #2b4c7e;
                font-size: 14px;
                border: 2px solid #a0b8d9;
                border-radius: 10px;
                padding: 5px;
                min-width: 150px;
            }
            QComboBox:hover { border: 2px solid #4a90e2; }
        """)
        self.language_combo.currentTextChanged.connect(self.translate_word)
        self.language_combo.hide()
        layout.addWidget(self.language_combo, alignment=Qt.AlignCenter)

        # Translation display
        self.translation_label = QLabel("Translation: ")
        self.translation_label.setFont(QFont("Segoe UI", 14))
        self.translation_label.setAlignment(Qt.AlignCenter)
        self.translation_label.setStyleSheet("color: #8e44ad;")
        layout.addWidget(self.translation_label)

        self.stop_btn = QPushButton("🛑 Stop Detection")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.stop_btn.clicked.connect(self.stop_detection)
        layout.addWidget(self.stop_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)

        # Start YOLO thread
        self.yolo_thread = YOLOThread("best.pt", conf=0.4)
        self.yolo_thread.frame_updated.connect(self.update_frame)
        self.yolo_thread.start()

    def update_frame(self, qt_img, word):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        if word:
            self.current_word = word
            self.word_label.setText(f"Detected Word: {word}")
            # Add unique words to form a sentence
            if not self.detected_words or self.detected_words[-1] != word:
                self.detected_words.append(word)
                self.sentence_label.setText("Formed Sentence: " + " ".join(self.detected_words))

    def stop_detection(self):
        self.yolo_thread.stop()
        self.close()

    def closeEvent(self, event):
        self.yolo_thread.stop()
        event.accept()

    def show_dropdown(self):
        self.language_combo.setVisible(not self.language_combo.isVisible())

    def translate_word(self, language):
        if not self.current_word:
            self.translation_label.setText("Translation: No word detected")
            return
        lang_code = 'hi' if language == 'Hindi' else 'kn'
        try:
            translator = GoogleTranslator(source='auto', target=lang_code)
            translated = translator.translate(self.current_word)
            self.translation_label.setText(f"Translation ({language}): {translated}")
        except Exception as e:
            self.translation_label.setText(f"Translation: Error - {str(e)}")


# ------------------ Main GUI ------------------
class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Sign Language Detection System")
        self.setWindowIcon(QIcon("hand_icon.png"))
        self.setGeometry(300, 100, 800, 600)
        self.setStyleSheet("background-color: #f2f6ff;")

        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("🤟 Sign Language Detection System")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2b4c7e;")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border: 2px solid #a0b8d9;")
        layout.addWidget(line)

        subtitle = QLabel("Select a Mode Below")
        subtitle.setFont(QFont("Segoe UI", 14, QFont.Bold))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #405d87;")
        layout.addWidget(subtitle)

        btn_style = """
            QPushButton {
                background-color: #4a90e2;
                color: white;
                font-size: 18px;
                border-radius: 20px;
                padding: 15px 30px;
                min-width: 300px;
            }
            QPushButton:hover { background-color: #357ABD; }
            QPushButton:pressed { background-color: #2b4c7e; }
        """

        self.word_btn = QPushButton("🖐️ Sentence Formation")
        self.word_btn.setStyleSheet(btn_style)
        self.word_btn.clicked.connect(self.open_word_detection)
        layout.addWidget(self.word_btn)

        self.alpha_num_btn = QPushButton("🔤 Alphabets & Numbers Recognition")
        self.alpha_num_btn.setStyleSheet(btn_style)
        self.alpha_num_btn.clicked.connect(self.start_alphabet_number_recognition)
        layout.addWidget(self.alpha_num_btn)

        footer = QLabel("")
        footer.setFont(QFont("Segoe UI", 10))
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray;")
        layout.addWidget(footer)

        self.setLayout(layout)

    def open_word_detection(self):
        self.detect_window = DetectionWindow()
        self.detect_window.show()

    def start_alphabet_number_recognition(self):
        QMessageBox.information(self, "Coming Soon", "Alphabet & Number recognition will be added soon.")


# ------------------ MAIN ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
