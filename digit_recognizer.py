"""손글씨 숫자 인식기.

마우스로 캔버스에 숫자(0-9)를 그리면 어떤 숫자인지 예측합니다.
처음 실행할 때 MNIST 데이터셋으로 작은 신경망(MLP)을 학습하고
디스크에 저장해 두므로, 다음 실행부터는 바로 시작됩니다.
"""

import os
import urllib.request

import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_PATH = os.path.join(BASE_DIR, "mnist.npz")
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.joblib")

CANVAS_SIZE = 280  # MNIST 해상도(28)의 10배
BRUSH_RADIUS = 10


def load_mnist():
    """MNIST 데이터셋을 (최초 1회) 내려받아 1차원 float 배열로 불러옵니다."""
    if not os.path.exists(MNIST_PATH):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_PATH)
    with np.load(MNIST_PATH) as data:
        x_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        x_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        return x_train, data["y_train"], x_test, data["y_test"]


def get_model():
    """저장된 모델을 불러오고, 없으면 새로 학습합니다."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    x_train, y_train, x_test, y_test = load_mnist()
    print("Training model (this takes about a minute)...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=15,
        batch_size=128,
        early_stopping=True,
        random_state=42,
        verbose=True,
    )
    model.fit(x_train, y_train)
    print(f"Test accuracy: {model.score(x_test, y_test):.4f}")
    joblib.dump(model, MODEL_PATH)
    return model


def preprocess(image):
    """그린 흑백 이미지를 MNIST 형식의 784차원 벡터로 변환합니다.

    MNIST 숫자는 20x20 영역에 맞춰지고 28x28 프레임 안에서 무게중심 기준으로
    가운데 정렬되어 있으므로, 그림도 같은 방식으로 정규화합니다.
    캔버스가 비어 있으면 None을 반환합니다.
    """
    pixels = np.array(image)
    rows = np.any(pixels > 0, axis=1)
    cols = np.any(pixels > 0, axis=0)
    if not rows.any():
        return None

    # 획이 있는 영역(바운딩 박스)만 잘라냄
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    digit = image.crop((left, top, right + 1, bottom + 1))

    # 비율을 유지하면서 긴 변이 20픽셀이 되도록 축소
    width, height = digit.size
    scale = 20.0 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = digit.resize(new_size, Image.LANCZOS)

    # 28x28 프레임의 가운데에 붙여넣기
    frame = Image.new("L", (28, 28), 0)
    frame.paste(digit, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    arr = np.array(frame, dtype=np.float32) / 255.0

    # 무게중심이 프레임 중앙에 오도록 이동
    cy, cx = ndimage.center_of_mass(arr)
    arr = ndimage.shift(arr, (14 - cy, 14 - cx), mode="constant")
    return np.clip(arr, 0, 1).reshape(1, 784)


class DigitApp:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        # 화면용 캔버스와, 같은 획을 따라 그리는 화면 밖 PIL 이미지
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonPress-1>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", lambda _event: self.clear())

        # 예측 결과 표시 영역
        panel = tk.Frame(root)
        panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.result_label = tk.Label(panel, text="?", font=("Arial", 72, "bold"))
        self.result_label.pack()
        self.confidence_label = tk.Label(panel, text="Draw a digit",
                                         font=("Arial", 12))
        self.confidence_label.pack()

        # 숫자별 확률 막대
        self.bars = tk.Canvas(panel, width=200, height=200)
        self.bars.pack(pady=10)

        buttons = tk.Frame(root)
        buttons.grid(row=1, column=1, padx=10, pady=10, sticky="s")
        tk.Button(buttons, text="Clear", width=10,
                  command=self.clear).pack(side="left", padx=5)
        tk.Button(buttons, text="Quit", width=10,
                  command=root.destroy).pack(side="left", padx=5)

        self.update_bars(np.zeros(10))

    def on_drag(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        if self.last_point:
            x0, y0 = self.last_point
            self.canvas.create_line(x0, y0, x, y, fill="white", width=2 * r,
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line((x0, y0, x, y), fill=255, width=2 * r)
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white")
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        self.last_point = (x, y)

    def on_release(self, _event):
        self.last_point = None
        self.predict()

    def predict(self):
        # 살짝 흐리게 해서 붓 가장자리를 MNIST 획처럼 부드럽게 만듦
        features = preprocess(self.image.filter(ImageFilter.GaussianBlur(2)))
        if features is None:
            return
        probs = self.model.predict_proba(features)[0]
        digit = int(np.argmax(probs))
        self.result_label.config(text=str(digit))
        self.confidence_label.config(text=f"Confidence: {probs[digit]:.1%}")
        self.update_bars(probs)

    def update_bars(self, probs):
        self.bars.delete("all")
        best = int(np.argmax(probs)) if probs.any() else -1
        for i, p in enumerate(probs):
            y = i * 20
            self.bars.create_text(10, y + 10, text=str(i), font=("Arial", 10))
            self.bars.create_rectangle(25, y + 4, 25 + 150 * p, y + 16,
                                       fill="#2e7d32" if i == best else "#90a4ae",
                                       outline="")
            self.bars.create_text(195, y + 10, text=f"{p:.0%}", anchor="e",
                                  font=("Arial", 9))

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        self.last_point = None
        self.result_label.config(text="?")
        self.confidence_label.config(text="Draw a digit")
        self.update_bars(np.zeros(10))


def main():
    model = get_model()
    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
