# job03_preprocess_steam_reviews_v4.py
# Steam 게임 리뷰 기반 추천 시스템 v4
#
# 역할:
#   job02 결과 파일(게임 1개 = 리뷰 문서 1행)을 영화 추천 앱 방식에 맞춰 전처리한다.
#
# v4 원칙:
#   - 기존 영화 추천 앱의 전처리 방식이 기준이다.
#   - Okt.pos(..., stem=True)를 사용한다.
#   - 명사 / 동사 / 형용사 / 영어(Alpha)를 사용한다.
#   - stopwords CSV + 코드 기본 불용어로 제거한다.
#   - max_df 같은 자동 단어 제거는 사용하지 않는다.
#   - 좋다, 재밌다, 추천, 갓겜, 망겜 등 평가 단어는 제거하지 않는다.

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
DATA_DIR = "./datasets"
INPUT_PATH = os.path.join(DATA_DIR, "steam_game_review_documents_v4.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "steam_game_reviews_preprocessed_v4.csv")
STOPWORDS_PATH = os.path.join(DATA_DIR, "steam_stopwords_v4.csv")


# =========================
# 전처리 설정
# =========================
CHUNK_MAX_LEN = 800
PROGRESS_INTERVAL = 50

# 영화 앱은 명사/동사/형용사를 사용했다.
# Steam에서는 rpg, fps, coop 같은 영어 장르/플레이 용어도 의미가 있으므로 Alpha도 유지한다.
POS_TO_KEEP = {"Noun", "Verb", "Adjective", "Alpha"}

# 아래 단어들은 게임 추천에서 중요한 평가/취향 신호다.
# 불용어에 들어가면 추천 품질이 크게 떨어질 수 있으므로 프로그램을 중단한다.
# KEEP_WORDS로 되살리는 방식이 아니라, 처음부터 불용어에 넣지 못하게 하는 검사다.
PROTECTED_WORDS = {
    "좋다", "나쁘다",
    "추천", "비추천",
    "재미", "재밌다", "재미있다", "재미없다",
    "갓겜", "망겜",
    "스토리", "그래픽", "난이도",
    "공포", "멀티", "싱글",
    "힐링", "농사", "생존",
}

# 코드 안에서 기본으로 추가할 불용어.
# 영화 앱의 "stopwords.csv + 코드 추가 불용어" 구조를 따른다.
# 주의:
#   - 좋다/나쁘다/추천/재미/갓겜/망겜 등 평가어는 넣지 않는다.
#   - 없다/않다/아니다 같은 부정 표현은 가치 평가에 필요할 수 있으므로 넣지 않는다.
DEFAULT_STOPWORDS = [
    # 접속/담화 표현
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한", "혹은", "또는",
    "정말", "진짜", "너무", "아주", "매우", "완전", "약간", "조금", "많이", "거의", "계속",
    "그냥", "일단", "뭔가", "어느", "이런", "저런", "그런", "이렇게", "저렇게", "그렇게",

    # 지시/대명사/일반 명사
    "여기", "저기", "거기", "이거", "저거", "그거", "이것", "저것", "그것",
    "때문", "정도", "느낌", "생각", "사람", "유저", "플레이어", "게이머",
    "경우", "부분", "처음", "마지막", "자체", "하나", "두개", "이번", "요즘", "현재", "과거",
    "다시", "한번", "때", "수", "것", "거", "듯", "점", "편", "내", "나", "우리", "저", "제",

    # 일반 동사/형용사. Okt stem=True 후 기본형으로 들어온다.
    "하다", "되다", "이다", "같다", "보다", "싶다",
    "가다", "오다", "주다", "받다", "만들다", "나오다", "들어가다", "해보다",
    "모르다", "보여주다", "시키다", "버리다", "두다", "넣다", "알다", "느끼다",

    # 프로젝트 전체에서 너무 일반적인 단어
    "게임", "겜", "스팀", "steam", "플레이", "플레이하다",
]

# 의미 있는 짧은 영어 게임 용어는 허용한다.
ALLOWED_SHORT_ENGLISH = {
    "rpg", "fps", "tps", "rts", "mmo", "moba", "vr", "ar", "ui", "ux", "ai", "npc", "pvp", "pve", "dlc",
    "co", "op", "td",
}

# stopwords 파일이 없을 때 생성할 기본 목록.
# 코드 DEFAULT_STOPWORDS와 일부 겹쳐도 괜찮다. set으로 합친다.
DEFAULT_STOPWORDS_FILE_WORDS = [
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한",
    "정말", "진짜", "너무", "아주", "매우", "완전", "약간", "조금", "많이",
    "그냥", "일단", "뭔가", "이거", "저거", "그거", "이것", "저것", "그것",
    "때문", "정도", "느낌", "생각", "사람", "유저", "플레이어", "부분",
    "하다", "되다", "이다", "같다", "보다", "싶다", "가다", "오다", "주다", "받다",
    "만들다", "나오다", "모르다", "보여주다",
    "게임", "겜", "스팀", "steam", "플레이",
]


# =========================
# 유틸 함수
# =========================
def check_required_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def create_default_stopwords_file(path):
    """steam_stopwords_v4.csv가 없으면 기본 파일을 만든다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_stopwords = pd.DataFrame({"stopword": DEFAULT_STOPWORDS_FILE_WORDS})
    df_stopwords.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"기본 불용어 파일을 생성했습니다: {path}")


def load_stopwords(path):
    """불용어 CSV와 코드 기본 불용어를 합쳐서 set으로 반환한다."""
    if not os.path.exists(path):
        create_default_stopwords_file(path)

    df_stopwords = pd.read_csv(path)

    if "stopword" not in df_stopwords.columns:
        raise ValueError("불용어 CSV에는 'stopword' 컬럼이 있어야 합니다.")

    file_stopwords = (
        df_stopwords["stopword"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    stopwords = set(file_stopwords + DEFAULT_STOPWORDS)
    stopwords = {word for word in stopwords if word}

    wrong_words = PROTECTED_WORDS & stopwords
    if wrong_words:
        raise ValueError(
            "중요 추천 단어가 불용어에 들어가 있습니다. "
            f"steam_stopwords_v4.csv 또는 DEFAULT_STOPWORDS에서 제거하세요: {sorted(wrong_words)}"
        )

    return stopwords


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
    # 영화 앱은 한글만 남겼지만, Steam은 rpg/fps/coop 같은 영어 단어가 의미 있으므로 영어도 유지한다.
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


def is_valid_token(token, stopwords):
    """추천에 사용할 토큰인지 판단한다."""
    if not token:
        return False

    token = str(token).strip().lower()

    # 한 글자 토큰은 대부분 의미가 약해서 제외
    if len(token) < 2:
        return False

    if token in stopwords:
        return False

    # aaa, bbbb 같은 영어 반복 잡음 제거
    if re.fullmatch(r"([a-z])\1{2,}", token):
        return False

    # aaaaaawwww 같은 영어 반복 잡음 제거
    if re.search(r"([a-z])\1{3,}", token):
        return False

    # 너무 짧은 영어는 대부분 잡음. 단, 게임 약어는 허용한다.
    if re.fullmatch(r"[a-z]{1,2}", token) and token not in ALLOWED_SHORT_ENGLISH:
        return False

    return True


def tokenize_reviews(text, okt, stopwords):
    """리뷰 문자열을 전처리하고 Okt 형태소 분석으로 토큰화한다."""
    cleaned = clean_text(text)
    chunks = split_text_by_length(cleaned)

    tokens = []

    for chunk in chunks:
        if not chunk:
            continue

        try:
            # 영화 추천 앱과 같은 핵심 설정: stem=True
            pos_result = okt.pos(chunk, norm=True, stem=True)
        except Exception as e:
            print("[경고] Okt 처리 실패. 해당 chunk는 건너뜁니다:", e)
            continue

        for word, pos in pos_result:
            word = str(word).strip().lower()

            if pos not in POS_TO_KEEP:
                continue

            if not is_valid_token(word, stopwords):
                continue

            tokens.append(word)

    return tokens


def main():
    start_time = time.time()

    print("=== job03 v4: Steam 리뷰 전처리 시작 ===")

    print("[1/6] 입력 CSV 읽는 중...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print("입력 행 수:", len(df))
    print("입력 컬럼 수:", len(df.columns))

    print("[2/6] 필수 컬럼 확인 중...")
    check_required_columns(df, ["appid", "titles", "reviews", "review_count"])

    print("[3/6] 불용어 읽는 중...")
    stopwords = load_stopwords(STOPWORDS_PATH)
    print("불용어 수:", len(stopwords))
    print("중요 평가/취향 단어 보호 검사: 통과")

    print("[4/6] Okt 준비 중...")
    okt = Okt()

    print("[5/6] 게임별 리뷰 전처리 중...")
    processed_reviews = []
    token_counts = []

    for idx, row in df.iterrows():
        title = row.get("titles", "")
        reviews = row.get("reviews", "")

        tokens = tokenize_reviews(reviews, okt, stopwords)
        processed_text = " ".join(tokens)

        processed_reviews.append(processed_text)
        token_counts.append(len(tokens))

        if (idx + 1) % PROGRESS_INTERVAL == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            print(
                f"진행: {idx + 1}/{len(df)} | "
                f"현재 게임: {title} | "
                f"토큰 수: {len(tokens)} | "
                f"경과: {elapsed:.1f}초"
            )

    print("[6/6] 결과 저장 중...")

    # 영화 추천 앱 구조에 맞춰 reviews 컬럼을 전처리 결과로 교체한다.
    # 원본 리뷰는 job02 결과 파일에 남아 있다.
    df["reviews"] = processed_reviews
    df["token_count"] = token_counts

    before_count = len(df)
    df = df[df["token_count"] > 0].copy()
    after_count = len(df)

    print("토큰 0개 게임 제거:", before_count - after_count, "행")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== job03 v4 완료 ===")
    print("저장 파일:", OUTPUT_PATH)
    print("불용어 파일:", STOPWORDS_PATH)
    print("최종 행 수:", len(df))
    print("최종 컬럼 수:", len(df.columns))
    print("평균 토큰 수:", round(df["token_count"].mean(), 2))
    print("최소 토큰 수:", int(df["token_count"].min()))
    print("최대 토큰 수:", int(df["token_count"].max()))

    print("\n전처리 결과 예시:")
    print(df[["appid", "titles", "review_count", "token_count", "reviews"]].head())

    elapsed = time.time() - start_time
    print(f"\n총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
