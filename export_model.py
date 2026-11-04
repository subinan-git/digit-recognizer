"""학습된 모델의 가중치를 웹 페이지용 JSON 파일로 내보냅니다.

브라우저는 scikit-learn 모델(.joblib)을 읽을 수 없으므로, 신경망의
가중치 행렬과 편향만 꺼내 docs/model.json에 저장합니다.
웹 페이지(docs/script.js)는 이 숫자들로 예측을 직접 계산합니다.
"""

import json
import os

import numpy as np

from digit_recognizer import BASE_DIR, get_model, load_mnist

OUTPUT_PATH = os.path.join(BASE_DIR, "docs", "model.json")
DECIMALS = 4  # 파일 크기를 줄이기 위해 소수점 4자리까지만 저장


def forward(x, weights, biases):
    """script.js와 같은 방식으로 예측을 계산합니다 (은닉층 ReLU, 출력층 softmax)."""
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = x @ w + b
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    return np.argmax(x, axis=1)


def main():
    model = get_model()
    # float32를 그대로 JSON에 쓰면 0.01229999959...처럼 길어지므로 float64로 바꾼 뒤 반올림
    weights = [np.round(w.astype(np.float64), DECIMALS) for w in model.coefs_]
    biases = [np.round(b.astype(np.float64), DECIMALS) for b in model.intercepts_]

    # 반올림한 가중치로도 정확도가 유지되는지 확인
    _, _, x_test, y_test = load_mnist()
    accuracy = np.mean(forward(x_test, weights, biases) == y_test)
    print(f"Accuracy with rounded weights: {accuracy:.4f}")

    # 각 층: weights는 [입력 수][출력 수] 2차원 배열, biases는 [출력 수] 배열
    data = {
        "layers": [
            {"weights": w.tolist(), "biases": b.tolist()}
            for w, b in zip(weights, biases)
        ]
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
