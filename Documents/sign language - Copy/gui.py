from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFrame, QComboBox, QMessageBox
)
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import cv2
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import traceback 

# ------------------ YOLO Detection Thread ------------------
class YOLOThread(QThread):
    frame_updated = pyqtSignal(QImage, str)
    finished = pyqtSignal()

    def __init__(self, weights_path="best.pt", conf=0.25):
        super().__init__()
        self.weights_path = weights_path
        self.conf = conf
        self.running = True
        self.cap = None 

    def run(self):
        try:
            print("Loading YOLO Model...")
            model = YOLO(self.weights_path)
            print("Model Loaded. Starting Webcam...")
            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                self.finished.emit()
                return

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Perform YOLO prediction
                results = model.predict(source=frame, conf=self.conf, save=False, verbose=False, stream=False)
                
                if not results:
                    continue

                annotated_frame = results[0].plot()

                # Extract detected word
                names = results[0].names
                boxes = results[0].boxes
                detected_word = ""
                
                if len(boxes) > 0:
                    cls_id = int(boxes.cls[0])
                    detected_word = names.get(cls_id, "") 

                # Convert frame to QImage
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_img = qt_img.scaled(640, 480, Qt.KeepAspectRatio)

                self.frame_updated.emit(scaled_img, detected_word)

        except Exception as e:
            print(f"An error occurred in YOLOThread: {e}")
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.finished.emit()

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.wait()


# ------------------ Translation Thread ------------------
class TranslationThread(QThread):
    translation_result = pyqtSignal(str, str, str) 

    def __init__(self, text, target_language, parent=None):
        super().__init__(parent)
        self.text = text
        self.target_language = target_language
        self.lang_code = 'hi' if target_language == 'Hindi' else 'kn'

    def run(self):
        try:
            clean_text = self.text.strip().lower()
            print(f"\n[DEBUG] Translator Thread Processing: '{clean_text}'")
            
            if not clean_text:
                self.translation_result.emit("No sentence to translate.", self.target_language, self.text)
                return

            # Force Source English to ensure sentence structure is respected
            translator = GoogleTranslator(source='en', target=self.lang_code)
            translated = translator.translate(clean_text)
            
            print(f"[DEBUG] API Returned: '{translated}'")
            self.translation_result.emit(translated, self.target_language, self.text)
            
        except Exception as e:
            print("\n--- Translation Error ---")
            traceback.print_exc()
            self.translation_result.emit(f"Error: {type(e).__name__}", self.target_language, self.text)


# ------------------ Detection Window ------------------
class DetectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖐️ Word Detection")
        self.setGeometry(350, 150, 800, 700) # Made window slightly taller
        self.setStyleSheet("background-color: #f7faff;")
        
        # --- DATA VARIABLES ---
        self.detected_words = []
        self.is_paused_for_translation = False 
        self.current_translation_thread = None 
        
        # --- STABILITY VARIABLES ---
        self.potential_word = ""
        self.stability_counter = 0
        
        # *** CHANGED FROM 15 TO 5 HERE ***
        self.STABILITY_THRESHOLD = 5 

        # Ensure keyboard presses work by clicking window
        self.setFocusPolicy(Qt.StrongFocus) 

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Current Word Feedback
        self.word_label = QLabel("Scanning...")
        self.word_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.word_label.setAlignment(Qt.AlignCenter)
        self.word_label.setStyleSheet("color: #2b4c7e;")
        layout.addWidget(self.word_label)

        # Sentence Display
        self.sentence_label = QLabel("Formed Sentence: (Empty)")
        self.sentence_label.setFont(QFont("Segoe UI", 14))
        self.sentence_label.setAlignment(Qt.AlignCenter)
        self.sentence_label.setStyleSheet("color: #3b5998; border: 2px solid #d0d0d0; padding: 8px; background: white; border-radius: 5px;")
        layout.addWidget(self.sentence_label)

        # Controls Layout
        controls_layout = QVBoxLayout()
        
        # Language Selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Hindi", "Kannada"])
        self.language_combo.setStyleSheet("font-size: 14px; padding: 5px; min-width: 200px;")
        controls_layout.addWidget(self.language_combo, alignment=Qt.AlignCenter)

        # --- NEW: BIG TRANSLATE BUTTON ---
        # This solves the issue of the 'D' key not working if focus is lost
        self.translate_btn = QPushButton("🌍 TRANSLATE SENTENCE NOW")
        self.translate_btn.setCursor(Qt.PointingHandCursor)
        self.translate_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; 
                color: white; 
                font-size: 16px; 
                font-weight: bold;
                padding: 12px; 
                border-radius: 8px;
                min-width: 250px;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.translate_btn.clicked.connect(self.translate_sentence_and_clear)
        controls_layout.addWidget(self.translate_btn, alignment=Qt.AlignCenter)
        
        # Clear Button
        self.clear_btn = QPushButton("🗑️ Clear Sentence")
        self.clear_btn.setStyleSheet("background-color: #f39c12; color: white; padding: 8px; border-radius: 5px;")
        self.clear_btn.clicked.connect(self.clear_sentence)
        controls_layout.addWidget(self.clear_btn, alignment=Qt.AlignCenter)

        layout.addLayout(controls_layout)

        # Translation Result Display
        self.translation_label = QLabel("Translation Result Will Appear Here")
        self.translation_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.translation_label.setAlignment(Qt.AlignCenter)
        self.translation_label.setStyleSheet("color: #8e44ad; margin-top: 10px;")
        layout.addWidget(self.translation_label)

        self.stop_btn = QPushButton("🛑 Stop Detection")
        self.stop_btn.setStyleSheet("background-color: #c0392b; color: white; padding: 8px; border-radius: 5px;")
        self.stop_btn.clicked.connect(self.stop_detection)
        layout.addWidget(self.stop_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)

        # Start YOLO thread
        self.yolo_thread = YOLOThread("best.pt", conf=0.4)
        self.yolo_thread.frame_updated.connect(self.update_frame)
        self.yolo_thread.start()

    def update_frame(self, qt_img, raw_word):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        
        if self.is_paused_for_translation:
            return

        # --- STABILITY LOGIC ---
        if raw_word:
            if raw_word == self.potential_word:
                self.stability_counter += 1
            else:
                self.potential_word = raw_word
                self.stability_counter = 0
                
            self.word_label.setText(f"Scanning: {raw_word} ({self.stability_counter}/{self.STABILITY_THRESHOLD})")

            if self.stability_counter == self.STABILITY_THRESHOLD:
                self.add_word_to_sentence(raw_word)
        else:
            self.potential_word = ""
            self.stability_counter = 0
            self.word_label.setText("Scanning...")

    def add_word_to_sentence(self, word):
        if not self.detected_words or self.detected_words[-1] != word:
            self.detected_words.append(word)
            current_sentence = " ".join(self.detected_words)
            self.sentence_label.setText(f"Formed Sentence: {current_sentence}")
            print(f"[DEBUG] Word Added: {word}. Full list: {self.detected_words}")

    # Keyboard Shortcut (Still exists, but Button is safer)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            self.translate_sentence_and_clear()
        elif event.key() == Qt.Key_C:
            self.clear_sentence()
        super().keyPressEvent(event)
        
    def clear_sentence(self):
        self.detected_words = []
        self.sentence_label.setText("Formed Sentence: (Empty)")
        self.translation_label.setText("Translation Result Will Appear Here")
        print("[DEBUG] Cleared by user")

    def translate_sentence_and_clear(self):
        sentence = " ".join(self.detected_words).strip()
        target_language = self.language_combo.currentText()
        
        print(f"\n[DEBUG] TRIGGERED: Sending to translator: '{sentence}'")

        if not sentence:
            self.translation_label.setText("⚠️ No words to translate!")
            return

        # PAUSE DETECTION
        self.is_paused_for_translation = True
        self.translate_btn.setEnabled(False) # Disable button so you can't click twice
        self.translate_btn.setText("⏳ Translating...")
        self.translation_label.setText(f"Processing...")
        
        # START THREAD
        if self.current_translation_thread and self.current_translation_thread.isRunning():
             self.current_translation_thread.quit()
        
        self.current_translation_thread = TranslationThread(sentence, target_language)
        self.current_translation_thread.translation_result.connect(self.handle_translation_result)
        self.current_translation_thread.start()
        
    def handle_translation_result(self, translated_text, target_language, original_sentence):
        # RESUME DETECTION
        self.is_paused_for_translation = False
        self.translate_btn.setEnabled(True)
        self.translate_btn.setText("🌍 TRANSLATE SENTENCE NOW")

        if translated_text.startswith("Error") or translated_text == "No sentence to translate.":
             self.translation_label.setText(f"Translation Failed: {translated_text}")
             return 

        # SHOW RESULT
        self.translation_label.setText(f"{translated_text}")
        
        # CLEAR DATA
        self.detected_words = []
        self.sentence_label.setText("Formed Sentence: (Empty)")
        print(f"[DEBUG] Success! Buffer Cleared.\n")

    def stop_detection(self):
        self.yolo_thread.stop()
        self.close()

    def closeEvent(self, event):
        self.yolo_thread.stop()
        if self.current_translation_thread:
            self.current_translation_thread.quit()
        event.accept()


# ------------------ Main GUI ------------------
class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Sign Language Detection System")
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

        self.word_btn = QPushButton("🖐️ Start Recognition")
        self.word_btn.setStyleSheet("""
            QPushButton { background-color: #4a90e2; color: white; font-size: 18px; border-radius: 20px; padding: 15px 30px; min-width: 300px; }
            QPushButton:hover { background-color: #357ABD; }
        """)
        self.word_btn.clicked.connect(self.open_word_detection)
        layout.addWidget(self.word_btn)

        self.setLayout(layout)
        self.detect_window = None 

    def open_word_detection(self):
        if self.detect_window is None or not self.detect_window.isVisible():
            self.detect_window = DetectionWindow()
            self.detect_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())