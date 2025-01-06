from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import pickle
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from sklearn.preprocessing import LabelEncoder

# Flask 앱 생성
app = Flask(__name__)

# 모델 및 사전 준비
model_path = "./model/mail_category_model_test1_0.9770641922950745.h5"
tokenizer_path = "./model/mail_token_max_222.pickle"  # 모델 학습 시 사용된 토크나이저 경로
stopwords_path = "./stopwords/stopwords.csv"  # 스탑워드 경로
encoder_path = "./model/encoder.pickle3"  # LabelEncoder 경로

# 모델, 토크나이저, 스탑워드, LabelEncoder 로드
model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

stopwords = pd.read_csv(stopwords_path, index_col=0)

# 형태소 분석기 준비
okt = Okt()

# maxlen 설정 (모델 학습 시 사용된 값)
maxlen = 222

# 카테고리 목록 (LabelEncoder의 클래스 정보)
categories = encoder.classes_


# 입력 텍스트를 모델에 전달 가능한 형식으로 변환
def preprocess_text(text):
    # 1. 형태소 분석
    title_morphed = okt.morphs(text, stem=True)

    # 2. 불용어 제거 및 길이 1 이하의 단어 제거
    title_filtered = ' '.join(
        [word for word in title_morphed if len(word) > 1 and word not in list(stopwords['stopword'])])

    # 3. 토크나이저를 사용해 시퀀스로 변환 및 패딩
    title_tokenized = tokenizer.texts_to_sequences([title_filtered])
    title_padded = pad_sequences(title_tokenized, maxlen=maxlen)

    return title_padded


# 라우트 설정
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    confidence = None
    title = None

    if request.method == 'POST':
        # 폼 데이터에서 제목 가져오기
        title = request.form['title']

        # 입력 데이터 전처리
        processed_title = preprocess_text(title)

        # 모델 예측
        prediction = model.predict(processed_title)

        # 가장 높은 확률의 인덱스를 기반으로 카테고리 예측
        category_index = np.argmax(prediction)  # 가장 높은 확률의 인덱스
        prediction_result = categories[category_index]  # 인덱스에 해당하는 카테고리
        confidence = round(np.max(prediction) * 100, 2)  # 확률을 퍼센트로 변환

    # 결과를 템플릿으로 전달
    return render_template("index.html",
                           prediction_result=prediction_result, confidence=confidence, title=title)


# Flask 앱 실행
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)