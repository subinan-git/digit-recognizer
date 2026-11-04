# Handwritten Digit Recognizer

마우스나 손가락으로 숫자(0~9)를 그리면 어떤 숫자인지 알아맞히는 프로그램입니다.
MNIST 데이터셋으로 학습한 신경망을 사용하며, 웹 버전과 데스크톱 버전이 있습니다.

**데모:** https://subinan-git.github.io/digit-recognizer/

## 웹 버전의 동작 방식

서버 없이 브라우저 안에서 모든 계산을 합니다.

1. 파이썬(scikit-learn)으로 신경망을 학습합니다. 구조는 784 → 256 → 128 → 10입니다.
2. `export_model.py`가 학습된 가중치를 `docs/model.json`으로 내보냅니다.
3. 브라우저가 `model.json`을 읽고, 그린 그림을 MNIST와 같은 28×28 형식으로 바꿉니다.
4. JavaScript가 행렬 곱셈 → ReLU → softmax 순서로 직접 계산해서 숫자를 예측합니다.

MNIST 시험 데이터 기준 정확도는 약 97.7%입니다.

## 파일 구성

```
├── docs/                 웹 버전 (GitHub Pages로 배포)
│   ├── index.html        페이지 구조
│   ├── style.css         디자인
│   ├── script.js         그리기, 전처리, 신경망 계산
│   └── model.json        학습된 가중치
├── digit_recognizer.py   데스크톱 버전 (tkinter) + 모델 학습
├── export_model.py       가중치를 docs/model.json으로 내보내기
└── requirements.txt      파이썬 패키지 목록
```

## 내 컴퓨터에서 실행하기

### 웹 버전

`fetch()`로 `model.json`을 읽기 때문에, HTML 파일을 더블클릭해서 열면 동작하지 않습니다.
로컬 서버로 열어야 합니다.

```bash
python -m http.server 8000 -d docs
```

그다음 브라우저에서 http://localhost:8000 에 접속합니다.

### 데스크톱 버전과 모델 다시 학습하기

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python digit_recognizer.py     # 데스크톱 앱 실행 (처음 실행 시 학습)
python export_model.py         # 웹용 model.json 다시 만들기
```

## GitHub Pages로 배포하기

1. 이 저장소를 GitHub에 올립니다.
2. 저장소 **Settings → Pages**로 이동합니다.
3. **Source**를 `Deploy from a branch`로, 브랜치를 `main`, 폴더를 `/docs`로 선택하고 저장합니다.
4. 1~2분 뒤 `https://<아이디>.github.io/<저장소 이름>/`에서 접속할 수 있습니다.
