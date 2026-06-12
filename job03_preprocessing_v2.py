# job03_preprocessing_v2.py
# Steam 게임 리뷰 기반 추천 시스템 v2
# 역할: job02 결과 파일(게임 1개 = 1행)을 Okt 형태소 분석으로 전처리한다.

import os
import re
import time
import pandas as pd

# Okt가 긴 문장을 처리할 때 Java heap memory 오류가 날 수 있어서 여유를 준다.
# 반드시 konlpy import 전에 설정해야 한다.
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

from konlpy.tag import Okt


# =========================
# 경로 설정
# =========================
INPUT_PATH = "./datasets/steam_game_reviews_grouped_v2.csv"
OUTPUT_PATH = "./datasets/steam_game_reviews_preprocessed_v2.csv"


# =========================
# 전처리 설정
# =========================
CHUNK_MAX_LEN = 800
PROGRESS_INTERVAL = 50

# Okt 품사 중 추천에 사용할 품사
# Noun      : 명사
# Verb      : 동사
# Adjective : 형용사
# Alpha     : 영어 단어
POS_TO_KEEP = {"Noun", "Verb", "Adjective", "Alpha"}

# 불용어
# 주의: 아래 단어들은 평가 표현으로 중요하므로 불용어에 넣지 않는다.
# 좋다, 나쁘다, 추천, 비추천, 재미, 재밌다, 재미있다, 재미없다, 갓겜, 망겜
STOPWORDS = {
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한", "정말", "진짜",
    "너무", "아주", "매우", "완전", "약간", "조금", "많이", "거의", "계속",
    "그냥", "일단", "뭔가", "어느", "이런", "저런", "그런", "여기", "저기",
    "이거", "저거", "그거", "이것", "저것", "그것", "때문", "정도", "느낌",
    "생각", "사람", "유저", "플레이어", "경우", "부분", "처음", "마지막",
    "자체", "하나", "두개", "이번", "요즘", "현재", "과거", "다시", "한번",
    "때", "수", "것", "거", "듯", "점", "편", "내", "나", "우리", "저", "제",
    "하다", "되다", "있다", "없다", "같다", "보다", "싶다", "이다", "아니다",
    "가다", "오다", "주다", "받다", "만들다", "나오다", "들어가다", "해보다",
    "게임", "스팀", "steam"
}


def clean_text(text):
    """리뷰 문자열에서 URL, 특수문자 등을 제거하고 한글/영어만 남긴다."""
    if pd.isna(text):
        return ""

    text = str(text)

    # URL 제거
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # 영어는 소문자로 통일
    text = text.lower()

    # 한글, 영어, 공백만 남김
    text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_text_by_length(text, max_len=CHUNK_MAX_LEN):
    """긴 문자열을 Okt가 처리 가능한 길이의 chunk로 나눈다."""
    if not text:
        return []

    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1

        # 너무 긴 단어는 강제로 자른다.
        if len(word) > max_len:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, len(word), max_len):
                chunks.append(word[i:i + max_len])
            continue

        if current_len + word_len > max_len:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def is_valid_token(token):
    """추천에 사용할 토큰인지 판단한다."""
    if not token:
        return False

    token = token.strip().lower()

    # 한 글자 토큰은 대부분 의미가 약해서 제외
    if len(token) < 2:
        return False

    # 불용어 제외
    if token in STOPWORDS:
        return False

    # aaa, bbbb 같은 영어 반복 잡음 제거
    if re.fullmatch(r"([a-z])\1{2,}", token):
        return False

    return True


def tokenize_reviews(text, okt):
    """리뷰 문자열을 전처리하고 Okt 형태소 분석으로 토큰화한다."""
    cleaned = clean_text(text)
    chunks = split_text_by_length(cleaned)

    tokens = []

    for chunk in chunks:
        if not chunk:
            continue

        try:
            pos_result = okt.pos(chunk, norm=True, stem=True)
        except Exception as e:
            print("[경고] Okt 처리 실패. 해당 chunk는 건너뜁니다:", e)
            continue

        for word, pos in pos_result:
            word = word.strip().lower()

            if pos not in POS_TO_KEEP:
                continue

            if not is_valid_token(word):
                continue

            tokens.append(word)

    return tokens


def main():
    start_time = time.time()

    print("=== job03: Steam 리뷰 전처리 시작 ===")
    print("[1/5] CSV 파일 읽는 중...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("입력 행 수:", len(df))
    print("입력 컬럼 수:", len(df.columns))

    print("[2/5] 필수 컬럼 확인 중...")
    required_columns = ["appid", "titles", "reviews", "review_count"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    print("[3/5] Okt 준비 중...")
    okt = Okt()

    print("[4/5] 게임별 리뷰 전처리 중...")
    token_texts = []
    token_counts = []

    for idx, row in df.iterrows():
        title = row.get("titles", "")
        reviews = row.get("reviews", "")

        tokens = tokenize_reviews(reviews, okt)
        token_text = " ".join(tokens)

        token_texts.append(token_text)
        token_counts.append(len(tokens))

        if (idx + 1) % PROGRESS_INTERVAL == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            print(f"진행: {idx + 1}/{len(df)} | 현재 게임: {title} | 토큰 수: {len(tokens)} | 경과: {elapsed:.1f}초")

    print("[5/5] 결과 저장 중...")

    df["tokens"] = token_texts
    df["token_count"] = token_counts

    # 토큰이 하나도 없는 게임은 추천 품질이 낮으므로 제외한다.
    before_count = len(df)
    df = df[df["token_count"] > 0].copy()
    after_count = len(df)

    print("토큰 0개 게임 제거:", before_count - after_count, "행")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== job03 완료 ===")
    print("저장 파일:", OUTPUT_PATH)
    print("최종 행 수:", len(df))
    print("최종 컬럼 수:", len(df.columns))
    print("평균 토큰 수:", round(df["token_count"].mean(), 2))
    print("최소 토큰 수:", int(df["token_count"].min()))
    print("최대 토큰 수:", int(df["token_count"].max()))

    print("\n전처리 결과 예시:")
    print(df[["appid", "titles", "review_count", "token_count", "tokens"]].head())

    elapsed = time.time() - start_time
    print(f"\n총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
