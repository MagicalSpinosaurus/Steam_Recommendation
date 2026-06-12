# job04_build_tfidf_steam_v4.py
# Steam 게임 리뷰 기반 추천 시스템 v4
#
# 역할:
#   job03 결과 파일(게임 1개 = 전처리된 리뷰 1행)을 읽어서
#   영화 추천 앱 방식과 최대한 같은 TF-IDF 모델을 만든다.
#
# v4 원칙:
#   - 영화 앱의 TF-IDF 방식 사용: TfidfVectorizer(sublinear_tf=True)
#   - max_df 사용 안 함
#   - min_df 사용 안 함
#   - 메타데이터 TF-IDF 섞지 않음
#   - 불용어/단어 제거는 job03에서 끝난 것으로 본다.
#   - job04는 전처리된 reviews 컬럼을 숫자 벡터로 바꾸는 단계만 담당한다.

import os
import pickle
import time

import pandas as pd
from scipy.io import mmwrite
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 경로 설정
# =========================
DATA_DIR = "./datasets"
MODEL_DIR = "./models/v4"

INPUT_FILE = os.path.join(DATA_DIR, "steam_game_reviews_preprocessed_v4.csv")
INDEX_FILE = os.path.join(DATA_DIR, "steam_recommendation_index_v4.csv")

TFIDF_MODEL_FILE = os.path.join(MODEL_DIR, "tfidf_steam_review_v4.pkl")
TFIDF_MATRIX_FILE = os.path.join(MODEL_DIR, "Tfidf_steam_review_v4.mtx")
CONFIG_FILE = os.path.join(MODEL_DIR, "job04_tfidf_config_v4.txt")


# =========================
# TF-IDF 설정
# =========================
# 영화 추천 앱과 같은 핵심 설정.
# 일부러 max_df, min_df를 쓰지 않는다.
TFIDF_PARAMS = {
    "sublinear_tf": True,
}

# Steam 게임 추천에서 중요한 단어들.
# job04에서는 제거하지 않고, 사전에 들어갔는지 확인용으로만 출력한다.
CORE_WORDS_TO_CHECK = [
    "좋다", "나쁘다",
    "추천", "비추천",
    "재미", "재밌다", "재미있다", "재미없다",
    "갓겜", "망겜",
    "스토리", "그래픽", "난이도",
    "공포", "멀티", "싱글",
    "힐링", "농사", "생존",
    "있다", "없다", "않다", "자다",
]


# =========================
# 유틸 함수
# =========================
def check_required_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def make_recommendation_index(df):
    """job06에서 결과 표시와 행렬 row 매칭에 사용할 인덱스 파일을 만든다."""
    preferred_columns = [
        "appid",
        "titles",
        "review_count",
        "voted_up_count",
        "voted_down_count",
        "positive_ratio_collected",
        "release_year",
        "genres",
        "tags",
        "categories",
        "review_score_desc_ko",
        "total_positive",
        "total_negative",
        "total_reviews",
        "developers",
        "publishers",
        "short_description",
        "token_count",
    ]

    existing_columns = [col for col in preferred_columns if col in df.columns]
    index_df = df[existing_columns].copy()
    return index_df


def print_core_word_check(vectorizer):
    """핵심 평가/취향 단어가 TF-IDF 사전에 남아 있는지 확인한다."""
    vocab = set(vectorizer.get_feature_names_out())

    print("\n핵심 단어 TF-IDF 사전 포함 여부:")
    for word in CORE_WORDS_TO_CHECK:
        status = "있음" if word in vocab else "없음"
        print(f"- {word}: {status}")


def save_config(vectorizer, matrix_shape, elapsed):
    """이번 job04 실행 설정을 사람이 읽을 수 있는 텍스트로 저장한다."""
    lines = []
    lines.append("job04_build_tfidf_steam_v4.py")
    lines.append("Steam 게임 리뷰 기반 추천 시스템 v4")
    lines.append("")
    lines.append("입력 파일:")
    lines.append(f"- {INPUT_FILE}")
    lines.append("")
    lines.append("출력 파일:")
    lines.append(f"- {INDEX_FILE}")
    lines.append(f"- {TFIDF_MODEL_FILE}")
    lines.append(f"- {TFIDF_MATRIX_FILE}")
    lines.append("")
    lines.append("TF-IDF 설정:")
    for key, value in TFIDF_PARAMS.items():
        lines.append(f"- {key}: {value}")
    lines.append("- max_df: 사용 안 함")
    lines.append("- min_df: 사용 안 함")
    lines.append("- metadata TF-IDF: 사용 안 함")
    lines.append("")
    lines.append("결과:")
    lines.append(f"- matrix shape: {matrix_shape}")
    lines.append(f"- vocabulary size: {len(vectorizer.vocabulary_)}")
    lines.append(f"- elapsed seconds: {elapsed:.1f}")

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 메인
# =========================
def main():
    start_time = time.time()

    print("=== job04 v4: Steam 리뷰 TF-IDF 벡터화 시작 ===")

    print("[1/6] 입력 CSV 읽는 중...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"입력 행 수: {len(df)}")
    print(f"입력 컬럼 수: {len(df.columns)}")

    print("[2/6] 필수 컬럼 확인 중...")
    # 영화 앱에서는 reviews 컬럼을 전처리된 텍스트로 사용했다.
    # v4도 job03 결과의 reviews 컬럼을 TF-IDF 대상으로 사용한다.
    check_required_columns(df, ["appid", "titles", "reviews"])

    print("[3/6] reviews 정리 중...")
    before_count = len(df)
    df["reviews"] = df["reviews"].fillna("").astype(str).str.strip()
    df = df[df["reviews"] != ""].reset_index(drop=True)
    after_count = len(df)

    print(f"빈 reviews 제거: {before_count - after_count} 행")
    print(f"TF-IDF 대상 게임 수: {len(df)}")

    if len(df) == 0:
        raise ValueError("TF-IDF에 사용할 reviews 데이터가 없습니다.")

    print("\n전처리 reviews 예시, 앞 120자만 표시:")
    for text in df["reviews"].head(3):
        print("-", text[:120] + ("..." if len(text) > 120 else ""))

    print("[4/6] 추천 인덱스 데이터시트 저장 중...")
    os.makedirs(DATA_DIR, exist_ok=True)
    index_df = make_recommendation_index(df)
    index_df.to_csv(INDEX_FILE, index=False, encoding="utf-8-sig")

    print("[5/6] TF-IDF 학습 및 변환 중...")
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    tfidf_matrix = tfidf.fit_transform(df["reviews"])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"단어 사전 크기: {len(tfidf.vocabulary_)}")

    print_core_word_check(tfidf)

    print("[6/6] 모델 저장 중...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(TFIDF_MODEL_FILE, "wb") as f:
        pickle.dump(tfidf, f)

    # 영화 앱처럼 MatrixMarket 형식으로 저장한다.
    mmwrite(TFIDF_MATRIX_FILE, tfidf_matrix)

    elapsed = time.time() - start_time
    save_config(tfidf, tfidf_matrix.shape, elapsed)

    print("\n=== job04 v4 완료 ===")
    print(f"저장 데이터시트: {INDEX_FILE}")
    print(f"저장 모델 1: {TFIDF_MODEL_FILE}")
    print(f"저장 모델 2: {TFIDF_MATRIX_FILE}")
    print(f"저장 설정 파일: {CONFIG_FILE}")
    print(f"최종 게임 수: {tfidf_matrix.shape[0]}")
    print(f"최종 단어 수: {tfidf_matrix.shape[1]}")

    feature_names = tfidf.get_feature_names_out()
    print("\n단어 사전 앞부분 예시:")
    print(feature_names[:50])

    print(f"\n총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
