# job05_train_word2vec_steam_v4.py
# Steam 게임 리뷰 기반 추천 시스템 v4
#
# 역할:
#   job03 v4에서 전처리된 게임별 리뷰 문서를 이용해 Word2Vec 모델을 학습한다.
#
# v4 원칙:
#   - 토대가 되는 영화 추천 앱의 Word2Vec 방식을 주로 따른다.
#   - job03 전처리 결과를 그대로 split()해서 학습한다.
#   - TF-IDF 사전으로 Word2Vec 단어를 다시 거르지 않는다.
#   - max_df, min_df 기반 추가 제거를 하지 않는다.
#   - Word2Vec은 "동의어 사전"이 아니라 "문맥상 가까운 단어 후보 생성기"로 사용한다.
#
# 영화 앱 기준 설정:
#   Word2Vec(tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1)

import os
import time
import json
import multiprocessing
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# =========================
# 경로 설정
# =========================
DATA_DIR = Path("./datasets")
MODEL_DIR = Path("./models/v4")

INPUT_CSV = DATA_DIR / "steam_game_reviews_preprocessed_v4.csv"
OUTPUT_MODEL = MODEL_DIR / "word2vec_steam_review_v4.model"
OUTPUT_CHECK_CSV = DATA_DIR / "word2vec_check_v4.csv"
OUTPUT_CONFIG = MODEL_DIR / "job05_word2vec_config_v4.json"


# =========================
# Word2Vec 설정
# =========================
# 영화 추천 앱의 설정을 Steam 프로젝트에 맞춰 거의 그대로 사용한다.
VECTOR_SIZE = 100
WINDOW = 4
MIN_COUNT = 20
WORKERS = min(4, max(1, multiprocessing.cpu_count() - 1))
EPOCHS = 100
SG = 1          # 1 = Skip-gram
SEED = 42

# 대표 유사어 확인용 단어
CHECK_WORDS = [
    "좋다",
    "나쁘다",
    "추천",
    "비추천",
    "재미",
    "재밌다",
    "재미있다",
    "재미없다",
    "갓겜",
    "망겜",
    "스토리",
    "그래픽",
    "난이도",
    "공포",
    "멀티",
    "싱글",
    "힐링",
    "농사",
    "생존",
    "있다",
    "없다",
    "않다",
    "자다",
    "도트",
    "액션",
    "퍼즐",
    "번역",
    "최적화",
]


# =========================
# 유틸
# =========================
class EpochLogger(CallbackAny2Vec):
    """Word2Vec epoch 진행 상황 출력용 callback."""

    def __init__(self, total_epochs):
        self.epoch = 0
        self.total_epochs = total_epochs
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, model):
        self.train_start_time = time.time()
        print("학습 루프 시작")

    def on_epoch_begin(self, model):
        self.epoch += 1
        self.epoch_start_time = time.time()
        print(f"epoch {self.epoch}/{self.total_epochs} 시작")

    def on_epoch_end(self, model):
        epoch_elapsed = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.train_start_time
        print(
            f"epoch {self.epoch}/{self.total_epochs} 완료 | "
            f"이번 epoch: {epoch_elapsed:.1f}초 | "
            f"누적 학습: {total_elapsed:.1f}초"
        )


def check_required_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def split_review_text(review_text):
    """전처리된 reviews 문자열을 Word2Vec 입력용 단어 리스트로 바꾼다."""
    if pd.isna(review_text):
        return []

    review_text = str(review_text).strip()
    if not review_text:
        return []

    return review_text.split()


def count_word_in_sentences(sentences, word):
    """대표 단어가 실제 학습 문장 안에 몇 번 들어 있는지 확인한다."""
    count = 0
    for sentence in sentences:
        count += sentence.count(word)
    return count


def build_check_rows(model, sentences, check_words, topn=10):
    """대표 단어의 출현 수와 Word2Vec 유사어를 점검 CSV로 저장하기 위한 rows 생성."""
    rows = []

    for word in check_words:
        original_count = count_word_in_sentences(sentences, word)

        if word not in model.wv:
            rows.append({
                "word": word,
                "count": original_count,
                "in_word2vec": False,
                "similar_words": "",
            })
            continue

        similar = model.wv.most_similar(word, topn=topn)
        similar_text = ", ".join([f"{w}({score:.3f})" for w, score in similar])

        rows.append({
            "word": word,
            "count": original_count,
            "in_word2vec": True,
            "similar_words": similar_text,
        })

    return rows


def print_check_result(check_df):
    """대표 단어 유사어 확인 결과를 콘솔에 출력한다."""
    print("\n유사어 확인 예시:")

    for _, row in check_df.iterrows():
        word = row["word"]
        count = int(row["count"])
        in_word2vec = bool(row["in_word2vec"])
        similar_words = row["similar_words"]

        if not in_word2vec:
            print(f"- {word} | 출현 {count}회: 단어장에 없음")
        else:
            print(f"- {word} | 출현 {count}회: {similar_words}")


def main():
    start_time = time.time()

    print("=== job05 v4: Steam 리뷰 Word2Vec 학습 시작 ===")

    print("[1/7] 입력 CSV 읽는 중...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"입력 행 수: {len(df)}")
    print(f"입력 컬럼 수: {len(df.columns)}")

    print("[2/7] 필수 컬럼 확인 중...")
    check_required_columns(df, ["appid", "titles", "reviews"])

    print("[3/7] reviews를 Word2Vec 입력 형식으로 변환 중...")
    df["token_list"] = df["reviews"].apply(split_review_text)

    before_count = len(df)
    df = df[df["token_list"].apply(len) > 0].copy()
    after_count = len(df)

    print(f"빈 token_list 제거: {before_count - after_count} 행")

    sentences = df["token_list"].tolist()
    total_tokens = sum(len(sentence) for sentence in sentences)

    if not sentences:
        raise ValueError("Word2Vec 학습에 사용할 문장이 없습니다.")

    print(f"Word2Vec 대상 게임 수: {len(sentences)}")
    print(f"전체 토큰 수: {total_tokens}")
    print(f"평균 토큰 수/게임: {total_tokens / len(sentences):.2f}")

    print("[4/7] Word2Vec 단어장 생성 중...")
    print(
        "설정: "
        f"vector_size={VECTOR_SIZE}, window={WINDOW}, min_count={MIN_COUNT}, "
        f"sg={SG}, epochs={EPOCHS}, workers={WORKERS}, seed={SEED}"
    )

    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        seed=SEED,
    )

    model.build_vocab(sentences)

    print(f"단어장 크기: {len(model.wv.index_to_key)}")
    print(f"학습 대상 단어 수(corpus_total_words): {model.corpus_total_words}")

    print("[5/7] Word2Vec 학습 중...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=EPOCHS,
        callbacks=[EpochLogger(EPOCHS)],
    )

    print("[6/7] 모델과 설정 저장 중...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    model.save(str(OUTPUT_MODEL))

    config = {
        "input_csv": str(INPUT_CSV),
        "output_model": str(OUTPUT_MODEL),
        "output_check_csv": str(OUTPUT_CHECK_CSV),
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "workers": WORKERS,
        "epochs": EPOCHS,
        "sg": SG,
        "seed": SEED,
        "note": "v4는 영화 추천 앱의 Word2Vec 방식을 기준으로 하며, job03 전처리 결과를 그대로 split()해서 학습한다.",
        "total_games": len(sentences),
        "total_tokens": int(total_tokens),
        "vocab_size": len(model.wv.index_to_key),
    }

    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[7/7] 대표 키워드 유사어 점검 CSV 저장 중...")
    check_rows = build_check_rows(model, sentences, CHECK_WORDS, topn=10)
    check_df = pd.DataFrame(check_rows)
    check_df.to_csv(OUTPUT_CHECK_CSV, index=False, encoding="utf-8-sig")

    print("\n=== job05 v4 완료 ===")
    print(f"저장 모델: {OUTPUT_MODEL}")
    print(f"저장 점검 CSV: {OUTPUT_CHECK_CSV}")
    print(f"저장 설정 파일: {OUTPUT_CONFIG}")
    print(f"최종 사용 게임 수: {len(sentences)}")
    print(f"최종 토큰 수: {total_tokens}")
    print(f"최종 단어장 크기: {len(model.wv.index_to_key)}")

    print_check_result(check_df)

    elapsed = time.time() - start_time
    print(f"\n총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
