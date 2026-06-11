import os
import re
import gc
import pandas as pd
from konlpy.tag import Okt
from gensim.models import Word2Vec

# ============================================================
# 0. 기본 설정
# ============================================================

os.makedirs("../Steam_Game_Recommendation/datasets", exist_ok=True)
os.makedirs("../Steam_Game_Recommendation/models", exist_ok=True)

input_path = "../Steam_Game_Recommendation/datasets/steam_game_reviews_grouped.csv"
output_path = "../Steam_Game_Recommendation/datasets/steam_game_reviews_preprocessed.csv"
word2vec_model_path = "../Steam_Game_Recommendation/models/word2vec_steam_review.model"

MIN_WORD_LEN = 2

# Okt가 너무 긴 문장을 한 번에 처리하면 Java heap 오류가 날 수 있으므로
# 긴 리뷰 묶음을 이 길이 기준으로 잘라서 처리합니다.
CHUNK_MAX_LEN = 800

# Word2Vec 설정
VECTOR_SIZE = 100
WINDOW = 4
MIN_COUNT = 20
WORKERS = 4
EPOCHS = 100
SG = 1


# ============================================================
# 1. 불용어 설정
# ============================================================
# 아래 단어들은 불용어에 넣지 않았습니다.
# 좋다, 나쁘다, 추천, 비추천,
# 재미, 재밌다, 재미있다, 재미없다, 갓겜, 망겜

stopwords = [
    # 너무 일반적인 동사
    "하다", "되다", "이다", "아니다",
    "보다", "싶다", "들다", "나다", "오다", "가다", "주다",
    "모르다", "알다", "같다",

    # 너무 일반적인 표현
    "정말", "진짜", "너무", "매우", "완전", "그냥", "약간",
    "조금", "계속", "다시", "아직", "이미", "거의", "일단",

    # 너무 일반적인 지시어/명사
    "이거", "저거", "그거", "이것", "저것", "그것",
    "사람", "생각", "느낌", "부분", "정도", "수준",
    "경우", "처음", "마지막", "이번", "요즘", "현재",
    "자체", "이상", "이하", "이후", "이전",

    # 프로젝트 목적상 의미가 약한 Steam/리뷰 관련 단어
    "스팀", "리뷰", "평가", "유저",
]

stopwords = set(stopwords)


# ============================================================
# 2. Okt 준비
# ============================================================

# Java heap을 넉넉하게 잡습니다.
# 그래도 핵심 해결책은 아래의 chunk 분할 처리입니다.
try:
    okt = Okt(max_heap_size=4096)
except TypeError:
    okt = Okt()


# ============================================================
# 3. 전처리 함수
# ============================================================

def clean_text(text):
    text = str(text)

    # Steam 리뷰에서는 FPS, RPG, DLC, PVP 같은 영어 약어도 의미가 있으므로
    # 한글과 영어만 남깁니다.
    text = re.sub("[^가-힣a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def split_text_to_chunks(text, max_len=800):
    """
    너무 긴 문자열을 Okt에 한 번에 넣지 않기 위해
    단어 단위로 적당한 길이의 chunk로 나눕니다.
    """
    words = text.split()

    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1

        if current_len + word_len > max_len and len(current_words) > 0:
            chunks.append(" ".join(current_words))
            current_words = []
            current_len = 0

        current_words.append(word)
        current_len += word_len

    if len(current_words) > 0:
        chunks.append(" ".join(current_words))

    return chunks


def tokenize_chunk(text):
    tokened_review = okt.pos(text, stem=True)

    words = []

    for word, word_class in tokened_review:
        word = str(word).strip()

        if word_class == "Alpha":
            word = word.lower()

        if (
            word_class == "Noun"
            or word_class == "Verb"
            or word_class == "Adjective"
            or word_class == "Alpha"
        ):
            if len(word) >= MIN_WORD_LEN:
                if word not in stopwords:
                    words.append(word)

    return words


def tokenize_long_review(text):
    """
    게임 1개에 합쳐진 긴 리뷰 문자열을 처리합니다.
    반환:
    - game_words: 게임 전체 토큰
    - chunk_sentences: Word2Vec 학습용 chunk별 토큰 리스트
    """
    text = clean_text(text)
    chunks = split_text_to_chunks(text, max_len=CHUNK_MAX_LEN)

    game_words = []
    chunk_sentences = []

    for chunk in chunks:
        words = tokenize_chunk(chunk)

        if len(words) > 0:
            game_words.extend(words)
            chunk_sentences.append(words)

    return game_words, chunk_sentences


# ============================================================
# 4. 데이터 불러오기
# ============================================================

df = pd.read_csv(input_path)

print("=" * 60)
print("원본 데이터 정보")
print("=" * 60)
df.info()

needed_columns = ["appid", "titles", "reviews", "review_count"]

for col in needed_columns:
    if col not in df.columns:
        raise ValueError(f"필요한 컬럼이 없습니다: {col}")

df = df.dropna(subset=["reviews"])
df["reviews"] = df["reviews"].astype(str)

print()
print("원본 게임 수:", len(df))
print("원본 appid 수:", df["appid"].nunique())

print()
print("=" * 60)
print("전처리 전 예시")
print("=" * 60)
print("제목:", df.iloc[0]["titles"])
print(df.iloc[0]["reviews"][:500])


# ============================================================
# 5. 형태소 분석
# ============================================================

cleaned_sentences = []
word2vec_sentences = []

for idx, review in enumerate(df["reviews"]):
    game_words, chunk_sentences = tokenize_long_review(review)

    cleaned_sentence = " ".join(game_words)
    cleaned_sentences.append(cleaned_sentence)

    word2vec_sentences.extend(chunk_sentences)

    if idx % 10 == 0:
        print(f"{idx} / {len(df)} 전처리 중")

    if idx % 100 == 0:
        gc.collect()

df["reviews"] = cleaned_sentences

# 전처리 후 비어버린 행 제거
df = df[df["reviews"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

# Word2Vec 학습용 빈 문장 제거
word2vec_sentences = [sentence for sentence in word2vec_sentences if len(sentence) > 0]

print()
print("=" * 60)
print("전처리 후 데이터 정보")
print("=" * 60)
df.info()

print()
print("전처리 후 게임 수:", len(df))
print("전처리 후 appid 수:", df["appid"].nunique())

print()
print("=" * 60)
print("전처리 후 예시")
print("=" * 60)
print("제목:", df.iloc[0]["titles"])
print(df.iloc[0]["reviews"][:500])


# ============================================================
# 6. 전처리 CSV 저장
# ============================================================

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print()
print("전처리 CSV 저장 완료:", output_path)


# ============================================================
# 7. Word2Vec 학습
# ============================================================

print()
print("=" * 60)
print("Word2Vec 학습 시작")
print("=" * 60)

print("Word2Vec 학습 문장 수:", len(word2vec_sentences))

if len(word2vec_sentences) == 0:
    raise ValueError("Word2Vec 학습용 문장이 없습니다.")

print("첫 번째 문장 토큰 예시:")
print(word2vec_sentences[0][:50])

embedding_model = Word2Vec(
    word2vec_sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    epochs=EPOCHS,
    sg=SG
)

embedding_model.save(word2vec_model_path)

print()
print("Word2Vec 모델 저장 완료:", word2vec_model_path)
print("Word2Vec 단어 수:", len(embedding_model.wv.index_to_key))

print()
print("Word2Vec 단어 예시:")
print(list(embedding_model.wv.index_to_key)[:100])


# ============================================================
# 8. Word2Vec 테스트
# ============================================================

test_words = [
    "힐링",
    "편안하다",
    "공포",
    "보스",
    "멀티",
    "추천",
    "비추천",
    "좋다",
    "나쁘다",
    "재미",
    "재밌다",
    "갓겜",
    "망겜",
]

print()
print("=" * 60)
print("Word2Vec 유사 단어 테스트")
print("=" * 60)

for word in test_words:
    if word in embedding_model.wv.index_to_key:
        print()
        print(f"[{word}] 유사 단어")
        print(embedding_model.wv.most_similar(word, topn=10))
    else:
        print()
        print(f"[{word}] Word2Vec vocabulary에 없음")