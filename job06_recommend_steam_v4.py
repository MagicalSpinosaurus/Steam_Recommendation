# job06_recommend_steam_v4.py
# Steam 게임 리뷰 기반 추천 시스템 v4
# 역할:
#   1) 게임 제목 기반 추천
#   2) 키워드 기반 추천
#   3) 키워드 + Word2Vec 유사어 확장 추천
#   4) Word2Vec 유사어 점검
#
# v4 설계 기준:
#   - 기존 영화 추천 앱의 TF-IDF + Word2Vec 추천 구조를 따른다.
#   - job03에서 만든 전처리 결과를 그대로 사용한다.
#   - max_df/min_df/metadata TF-IDF 같은 추가 구조는 사용하지 않는다.
#   - 단, Steam 리뷰에서는 Word2Vec 유사어에 반대 의미가 섞일 수 있으므로
#     추천 단계에서만 최소 안전 필터를 적용한다.

import os
import re
import pickle
from collections import defaultdict

import pandas as pd
from scipy.io import mmread
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec

# Okt가 사용자 입력을 분석할 때 사용할 Java 메모리 설정
# 반드시 konlpy import 전에 설정한다.
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

try:
    from konlpy.tag import Okt
except Exception:
    Okt = None


# =========================
# 경로 설정
# =========================
DATA_PATH = "./datasets/steam_recommendation_index_v4.csv"
STOPWORDS_PATH = "./datasets/steam_stopwords_v4.csv"
TFIDF_MODEL_PATH = "./models/v4/tfidf_steam_review_v4.pkl"
TFIDF_MATRIX_PATH = "./models/v4/Tfidf_steam_review_v4.mtx"
WORD2VEC_MODEL_PATH = "./models/v4/word2vec_steam_review_v4.model"


# =========================
# 추천 설정
# =========================
DEFAULT_TOP_N = 10
WORD2VEC_TOP_N = 10
ORIGINAL_KEYWORD_WEIGHT = 11
WORD2VEC_MIN_SIMILARITY = 0.55

# 영화 앱은 Word2Vec 유사어를 그대로 사용했다.
# Steam 리뷰에서는 멀티↔싱글, 좋다↔나쁘다처럼 반대 의미가 함께 나올 수 있으므로
# v4 추천 단계에서만 최소한의 충돌 차단을 적용한다.
USE_SAFE_WORD2VEC_FILTER = True

POS_TO_KEEP = {"Noun", "Verb", "Adjective", "Alpha"}

# query 토큰화용 기본 불용어.
# job03은 steam_stopwords_v4.csv를 사용한다.
# 여기서는 사용자 입력 키워드를 전처리할 때 같은 방향을 맞추기 위해 사용한다.
FALLBACK_STOPWORDS = {
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한",
    "정말", "진짜", "너무", "아주", "매우", "완전", "약간", "조금", "많이",
    "그냥", "일단", "뭔가", "이런", "저런", "그런", "여기", "저기",
    "이거", "저거", "그거", "이것", "저것", "그것", "때문", "정도", "느낌",
    "생각", "사람", "유저", "플레이어", "경우", "부분", "처음", "마지막",
    "자체", "하나", "이번", "요즘", "현재", "과거", "다시", "한번",
    "때", "수", "것", "거", "듯", "점", "편", "내", "나", "우리", "저", "제",
    "하다", "되다", "이다", "같다", "보다", "싶다", "가다", "오다", "주다", "받다",
    "게임", "스팀", "steam",
}

# Steam 추천에서 중요한 평가/취향 단어.
# 이 단어들은 불용어로 제거하지 않는 것이 v4의 기준이다.
PROTECTED_WORDS = {
    "좋다", "나쁘다", "추천", "비추천", "재미", "재밌다", "재미있다", "재미없다",
    "갓겜", "망겜", "스토리", "그래픽", "난이도", "공포", "멀티", "싱글",
    "힐링", "농사", "생존",
}

# Word2Vec 안전 필터용 의미 그룹
EXPANSION_GROUPS = {
    "positive": {
        "좋다", "추천", "재미", "재밌다", "재미있다", "갓겜", "명작", "수작",
        "꿀잼", "개꿀잼", "존잼", "강추", "굿굿", "만족스럽다", "훌륭하다",
        "재다",  # Okt가 재미있다/재밌다 주변에서 자주 만드는 토큰. 완전히 좋진 않지만 긍정 문맥이 많음.
    },
    "negative": {
        "나쁘다", "비추천", "재미없다", "망겜", "똥겜", "똥망겜", "운빨망겜",
        "좆망", "노잼", "별로", "최악", "실망", "병신", "쓰레기", "지루하다",
        "지겹다", "아쉽다", "그닥", "그다지", "비추다",
    },
    "multiplayer": {
        "멀티", "멀티플레이", "멀티플레이어", "코옵", "협동", "온라인", "pvp", "pve", "coop",
    },
    "singleplayer": {
        "싱글", "싱글플레이", "싱글플레이어", "솔플", "솔로", "오프라인", "캠페인",
    },
    "hard": {
        "어렵다", "어려움", "하드", "하드코어", "빡세다", "난도", "고난도",
    },
    "easy": {
        "쉽다", "쉬움", "이지", "캐주얼", "라이트", "입문", "무난",
    },
}

CONFLICT_GROUP_PAIRS = {
    ("positive", "negative"),
    ("multiplayer", "singleplayer"),
    ("hard", "easy"),
}

# Word2Vec 결과에서 추천 문장 확장용으로는 부적절한 형태소/잡음 일부
GLOBAL_BLOCKLIST = {
    "추하다", "천하다", "드리다", "드림", "는걸", "잘만", "해봤다", "모트",
    "겜임", "쇼니", "und", "fan", "casual", "답지", "전혀", "아무",
}


def build_group_index():
    word_to_groups = defaultdict(set)
    for group, words in EXPANSION_GROUPS.items():
        for word in words:
            word_to_groups[word].add(group)
    return word_to_groups


WORD_TO_GROUPS = build_group_index()


def groups_of(word):
    return WORD_TO_GROUPS.get(str(word).strip().lower(), set())


def are_conflicting(source_word, candidate_word):
    source_groups = groups_of(source_word)
    candidate_groups = groups_of(candidate_word)

    for a in source_groups:
        for b in candidate_groups:
            if (a, b) in CONFLICT_GROUP_PAIRS or (b, a) in CONFLICT_GROUP_PAIRS:
                return True
    return False


def is_valid_token(token, stopwords):
    if token is None:
        return False

    token = str(token).strip().lower()

    if len(token) < 2:
        return False

    if token in stopwords:
        return False

    if token in GLOBAL_BLOCKLIST:
        return False

    # aaa, bbbb 같은 영어 반복 잡음 제거
    if re.fullmatch(r"([a-z])\1{2,}", token):
        return False

    # 영어 한 글자/두 글자는 대부분 추천 키워드로 약함. 단, 게임 약어는 허용한다.
    allowed_short_english = {"vr", "ar", "ai", "ui", "ux", "dlc", "npc", "rpg", "fps", "tps", "pvp", "pve"}
    if re.fullmatch(r"[a-z]{1,2}", token) and token not in allowed_short_english:
        return False

    return True


def load_stopwords():
    stopwords = set(FALLBACK_STOPWORDS)

    if os.path.exists(STOPWORDS_PATH):
        df_stop = pd.read_csv(STOPWORDS_PATH)
        if "stopword" in df_stop.columns:
            stopwords.update(df_stop["stopword"].dropna().astype(str).str.strip().tolist())

    # 보호 단어가 불용어에 들어가면 추천 의도가 손상된다.
    # KEEP_WORDS로 되살리지 않고, 잘못된 불용어 설정을 알려준다.
    wrong_words = PROTECTED_WORDS & stopwords
    if wrong_words:
        raise ValueError(
            "중요 평가/취향 단어가 불용어에 들어가 있습니다. "
            f"steam_stopwords_v4.csv 또는 기본 불용어에서 제거하세요: {sorted(wrong_words)}"
        )

    return stopwords


def clean_query_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_query(query, okt, stopwords):
    cleaned = clean_query_text(query)

    if not cleaned:
        return []

    tokens = []

    if okt is not None:
        try:
            pos_result = okt.pos(cleaned, norm=True, stem=True)
            for word, pos in pos_result:
                word = word.strip().lower()
                if pos not in POS_TO_KEEP:
                    continue
                if not is_valid_token(word, stopwords):
                    continue
                tokens.append(word)
        except Exception as e:
            print("[경고] Okt 처리 실패. 공백 기준으로 대체합니다:", e)
            tokens = [word for word in cleaned.split() if is_valid_token(word, stopwords)]
    else:
        tokens = [word for word in cleaned.split() if is_valid_token(word, stopwords)]

    # 순서 유지 중복 제거
    return list(dict.fromkeys(tokens))


def load_resources():
    required_paths = [
        DATA_PATH,
        TFIDF_MODEL_PATH,
        TFIDF_MATRIX_PATH,
        WORD2VEC_MODEL_PATH,
    ]

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다: " + ", ".join(missing))

    print("[1/5] 추천 index CSV 읽는 중...")
    df = pd.read_csv(DATA_PATH)
    df = df.reset_index(drop=True)

    print("[2/5] TF-IDF 변환기 읽는 중...")
    with open(TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    print("[3/5] TF-IDF 행렬 읽는 중...")
    tfidf_matrix = mmread(TFIDF_MATRIX_PATH).tocsr()

    print("[4/5] Word2Vec 모델 읽는 중...")
    word2vec_model = Word2Vec.load(WORD2VEC_MODEL_PATH)

    print("[5/5] 불용어 읽는 중...")
    stopwords = load_stopwords()

    if len(df) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"index CSV 행 수와 TF-IDF 행렬 행 수가 다릅니다. CSV={len(df)}, TF-IDF={tfidf_matrix.shape[0]}"
        )

    print("\n=== 추천 준비 완료 ===")
    print("게임 수:", len(df))
    print("TF-IDF 행렬:", tfidf_matrix.shape)
    print("TF-IDF 단어 수:", len(tfidf.get_feature_names_out()))
    print("Word2Vec 단어장 크기:", len(word2vec_model.wv.index_to_key))
    print("불용어 수:", len(stopwords))

    return df, tfidf, tfidf_matrix, word2vec_model, stopwords


def create_okt():
    if Okt is None:
        print("[경고] konlpy를 불러오지 못했습니다. 공백 기준 토큰화로 대체합니다.")
        return None
    return Okt()


def find_game_candidates(df, title_query, max_results=20):
    query = str(title_query).strip().lower()
    if not query:
        return pd.DataFrame()

    titles = df["titles"].fillna("").astype(str)
    titles_lower = titles.str.lower()

    exact_mask = titles_lower == query
    contains_mask = titles_lower.str.contains(re.escape(query), na=False)

    candidates = df[exact_mask | contains_mask].copy()
    return candidates.head(max_results)


def format_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def print_recommendations(result_df, title="추천 결과"):
    if result_df.empty:
        print("추천 결과가 없습니다.")
        return

    print(f"\n=== {title} ===")

    for rank, (_, row) in enumerate(result_df.iterrows(), start=1):
        print(f"\n{rank}. {row.get('titles', '')}")
        print(f"   추천 점수: {row.get('similarity', 0):.4f}")
        print(f"   출시년도: {format_value(row.get('release_year', ''))}")
        print(f"   장르: {format_value(row.get('genres', ''))}")
        print(f"   태그: {format_value(row.get('tags', ''))}")
        print(f"   한국어 평가: {format_value(row.get('review_score_desc_ko', ''))}")
        print(f"   수집 리뷰 수: {format_value(row.get('review_count', ''))}")
        if "positive_ratio_collected" in row:
            ratio = row.get("positive_ratio_collected")
            if pd.notna(ratio):
                print(f"   수집 리뷰 추천 비율: {float(ratio):.2%}")


def recommend_by_title(df, tfidf_matrix, title_query, top_n=DEFAULT_TOP_N):
    candidates = find_game_candidates(df, title_query, max_results=20)

    if candidates.empty:
        print("해당 제목을 찾지 못했습니다.")
        return pd.DataFrame()

    if len(candidates) > 1:
        print("\n제목 후보가 여러 개입니다. 첫 번째 후보를 기준으로 추천합니다.")
        for i, (_, row) in enumerate(candidates.iterrows(), start=1):
            print(f"{i}. {row.get('titles', '')} | {row.get('release_year', '')} | {row.get('review_score_desc_ko', '')}")

    target_index = candidates.index[0]
    target_title = df.loc[target_index, "titles"]

    cosine_sim = linear_kernel(tfidf_matrix[target_index], tfidf_matrix).flatten()

    result_df = df.copy()
    result_df["similarity"] = cosine_sim
    result_df = result_df[result_df.index != target_index]
    result_df = result_df.sort_values("similarity", ascending=False).head(top_n)

    print_recommendations(result_df, title=f"'{target_title}' 기준 비슷한 게임 추천")
    return result_df


def should_accept_w2v_candidate(source, candidate, score):
    source = str(source).strip().lower()
    candidate = str(candidate).strip().lower()

    if score < WORD2VEC_MIN_SIMILARITY:
        return False, "유사도 낮음"

    if candidate == source:
        return False, "자기 자신"

    if candidate in GLOBAL_BLOCKLIST:
        return False, "확장용 잡음"

    if USE_SAFE_WORD2VEC_FILTER and are_conflicting(source, candidate):
        return False, "반대 의미 그룹"

    return True, "통과"


def build_movie_style_expanded_sentence(tokens, word2vec_model):
    """영화 앱의 키워드 확장 구조를 Steam v4에 맞게 확장한다.

    영화 앱 방식:
      - 원본 키워드 11회 반복
      - 유사어 10개를 10, 9, 8 ... 1회 반복

    Steam v4 방식:
      - 여러 키워드 입력을 지원한다.
      - 각 원본 토큰을 11회 반복한다.
      - Word2Vec 후보는 영화 앱처럼 가중 반복하되, 반대 의미 후보만 최소 차단한다.
    """
    sentence_tokens = []
    expansion_logs = []
    rejected_logs = []

    for token in tokens:
        sentence_tokens.extend([token] * ORIGINAL_KEYWORD_WEIGHT)

        if token not in word2vec_model.wv:
            expansion_logs.append(f"{token} → Word2Vec 단어장에 없음")
            continue

        try:
            sim_words = word2vec_model.wv.most_similar(token, topn=WORD2VEC_TOP_N)
        except Exception as e:
            expansion_logs.append(f"{token} → 유사어 추출 실패: {e}")
            continue

        accepted = []
        rejected = []
        weight = WORD2VEC_TOP_N

        for word, score in sim_words:
            word = str(word).strip().lower()
            ok, reason = should_accept_w2v_candidate(token, word, score)

            if ok:
                sentence_tokens.extend([word] * weight)
                accepted.append(f"{word}({score:.3f}, x{weight})")
                weight -= 1
            else:
                rejected.append(f"{word}({score:.3f}: {reason})")

            if weight <= 0:
                break

        if accepted:
            expansion_logs.append(f"{token} → " + ", ".join(accepted))
        else:
            expansion_logs.append(f"{token} → 채택된 유사어 없음")

        if rejected:
            rejected_logs.append(f"{token} 차단 → " + ", ".join(rejected[:8]))

    return " ".join(sentence_tokens), expansion_logs, rejected_logs


def recommend_by_keyword(df, tfidf, tfidf_matrix, query, okt, stopwords, word2vec_model=None, use_word2vec=False, top_n=DEFAULT_TOP_N):
    tokens = tokenize_query(query, okt, stopwords)

    if not tokens:
        print("입력에서 사용할 수 있는 토큰을 찾지 못했습니다.")
        return pd.DataFrame()

    print("\n입력 토큰:", ", ".join(tokens))

    if use_word2vec:
        query_sentence, expansion_logs, rejected_logs = build_movie_style_expanded_sentence(tokens, word2vec_model)

        print("\n채택한 Word2Vec 확장:")
        for log in expansion_logs:
            print("-", log)

        if rejected_logs:
            print("\n차단한 Word2Vec 후보 일부:")
            for log in rejected_logs:
                print("-", log)
    else:
        # 직접 키워드 추천에서도 원본 키워드의 영향력을 영화 앱 수준으로 유지한다.
        query_sentence = " ".join([token for token in tokens for _ in range(ORIGINAL_KEYWORD_WEIGHT)])

    query_vector = tfidf.transform([query_sentence])

    if query_vector.nnz == 0:
        print("입력 토큰이 TF-IDF 사전에 없어 추천을 계산할 수 없습니다.")
        return pd.DataFrame()

    cosine_sim = linear_kernel(query_vector, tfidf_matrix).flatten()

    result_df = df.copy()
    result_df["similarity"] = cosine_sim
    result_df = result_df.sort_values("similarity", ascending=False).head(top_n)

    if use_word2vec:
        title = "키워드 + Word2Vec 확장 추천"
    else:
        title = "키워드 직접 추천"

    print_recommendations(result_df, title=title)
    return result_df


def inspect_word2vec(word2vec_model, query, okt, stopwords):
    tokens = tokenize_query(query, okt, stopwords)

    if not tokens:
        print("입력에서 사용할 수 있는 토큰을 찾지 못했습니다.")
        return

    print("\n입력 토큰:", ", ".join(tokens))

    for token in tokens:
        if token not in word2vec_model.wv:
            print(f"\n- {token}: Word2Vec 단어장에 없음")
            continue

        print(f"\n- {token} 유사어:")
        for word, score in word2vec_model.wv.most_similar(token, topn=WORD2VEC_TOP_N):
            ok, reason = should_accept_w2v_candidate(token, word, score)
            status = "채택" if ok else f"차단: {reason}"
            print(f"  {word}({score:.3f}) | {status}")


def show_menu():
    print("\n==============================")
    print("Steam 게임 추천 테스트 v4")
    print("==============================")
    print("1. 게임 제목으로 비슷한 게임 추천")
    print("2. 키워드 직접 추천")
    print("3. 키워드 + Word2Vec 확장 추천")
    print("4. 게임 제목 검색만 하기")
    print("5. Word2Vec 유사어 확인")
    print("0. 종료")


def main():
    print("=== job06 v4: Steam 게임 추천 테스트 시작 ===")
    df, tfidf, tfidf_matrix, word2vec_model, stopwords = load_resources()
    okt = create_okt()

    while True:
        show_menu()
        menu = input("\n번호를 입력하세요: ").strip()

        if menu == "0":
            print("추천 테스트를 종료합니다.")
            break

        elif menu == "1":
            title_query = input("기준이 될 게임 제목을 입력하세요: ").strip()
            recommend_by_title(df, tfidf_matrix, title_query)

        elif menu == "2":
            query = input("추천 키워드를 입력하세요: ").strip()
            recommend_by_keyword(df, tfidf, tfidf_matrix, query, okt, stopwords, use_word2vec=False)

        elif menu == "3":
            query = input("추천 키워드를 입력하세요: ").strip()
            recommend_by_keyword(df, tfidf, tfidf_matrix, query, okt, stopwords, word2vec_model=word2vec_model, use_word2vec=True)

        elif menu == "4":
            title_query = input("찾을 게임 제목을 입력하세요: ").strip()
            candidates = find_game_candidates(df, title_query)
            if candidates.empty:
                print("검색 결과가 없습니다.")
            else:
                print("\n검색 결과:")
                for i, (_, row) in enumerate(candidates.iterrows(), start=1):
                    print(f"{i}. {row.get('titles', '')} | {row.get('release_year', '')} | {row.get('review_score_desc_ko', '')}")

        elif menu == "5":
            query = input("확인할 키워드를 입력하세요: ").strip()
            inspect_word2vec(word2vec_model, query, okt, stopwords)

        else:
            print("잘못 입력했습니다. 0~5 중에서 선택해주세요.")


if __name__ == "__main__":
    main()
