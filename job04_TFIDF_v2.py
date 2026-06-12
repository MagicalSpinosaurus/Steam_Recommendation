import os
import pickle
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 경로 설정
# =========================
DATA_DIR = "./datasets"
MODEL_DIR = "./models"

INPUT_FILE = os.path.join(DATA_DIR, "steam_game_reviews_preprocessed_v2.csv")
TFIDF_MODEL_FILE = os.path.join(MODEL_DIR, "tfidf_steam_review_v2.pkl")
TFIDF_MATRIX_FILE = os.path.join(MODEL_DIR, "Tfidf_steam_review_v2.mtx")


# =========================
# TF-IDF 설정
# =========================
# 기존 영화 추천기 흐름을 최대한 유지하기 위해 기본 unigram + sublinear_tf=True 사용
# tokens 컬럼은 이미 Okt 전처리가 끝난 "공백으로 구분된 토큰 문자열"이므로 tokenizer는 따로 쓰지 않음
TFIDF_PARAMS = {
    "sublinear_tf": True,
}


# =========================
# 유틸 함수
# =========================
def check_required_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def main():
    start_time = time.time()

    print("=== job04: Steam 리뷰 TF-IDF 벡터화 시작 ===")

    print("[1/5] CSV 파일 읽는 중...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"입력 행 수: {len(df)}")
    print(f"입력 컬럼 수: {len(df.columns)}")

    print("[2/5] 필수 컬럼 확인 중...")
    check_required_columns(df, ["appid", "titles", "tokens"])

    print("[3/5] tokens 정리 중...")
    before_count = len(df)
    df["tokens"] = df["tokens"].fillna("").astype(str)
    df["tokens"] = df["tokens"].str.strip()
    df = df[df["tokens"] != ""].reset_index(drop=True)
    after_count = len(df)

    print(f"빈 tokens 제거: {before_count - after_count} 행")
    print(f"TF-IDF 대상 게임 수: {len(df)}")

    if len(df) == 0:
        raise ValueError("TF-IDF에 사용할 tokens 데이터가 없습니다.")

    print("[4/5] TF-IDF 학습 및 변환 중...")
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    tfidf_matrix = tfidf.fit_transform(df["tokens"])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"단어 사전 크기: {len(tfidf.vocabulary_)}")

    print("[5/5] 모델 저장 중...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(TFIDF_MODEL_FILE, "wb") as f:
        pickle.dump(tfidf, f)

    with open(TFIDF_MATRIX_FILE, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print("\n=== job04 완료 ===")
    print(f"저장 파일 1: {TFIDF_MODEL_FILE}")
    print(f"저장 파일 2: {TFIDF_MATRIX_FILE}")
    print(f"최종 게임 수: {tfidf_matrix.shape[0]}")
    print(f"최종 단어 수: {tfidf_matrix.shape[1]}")

    feature_names = tfidf.get_feature_names_out()
    print("\n단어 사전 앞부분 예시:")
    print(feature_names[:50])

    elapsed = time.time() - start_time
    print(f"\n총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
