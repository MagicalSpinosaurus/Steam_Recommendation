# job02_make_steam_review_documents_v4.py
# Steam 게임 리뷰 기반 추천 시스템 v4
#
# 역할:
#   steam_reviews_raw_v2.csv  : 리뷰 1개 = 1행
#   steam_games_detail_v2.csv : 게임 1개 = 1행
#
#   위 두 파일을 appid 기준으로 합쳐서
#   게임 1개 = 리뷰 문서 1개 형태의 CSV를 만든다.
#
# v4 설계 원칙:
#   - 영화 추천 앱의 구조를 토대로 한다.
#   - job02는 전처리/불용어 제거를 하지 않는다.
#   - job02는 "게임별 리뷰 문서"를 만드는 단계다.
#   - 형태소 분석, stem=True, 불용어 제거는 job03에서 한다.
#   - 장르/태그/평가 정보는 추천 결과 표시용으로 함께 보존한다.

import os
import pandas as pd


# =========================
# 경로 설정
# =========================
DATA_DIR = "./datasets"

REVIEWS_PATH = os.path.join(DATA_DIR, "steam_reviews_raw_v2.csv")
GAMES_PATH = os.path.join(DATA_DIR, "steam_games_detail_v2.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "steam_game_review_documents_v4.csv")


# =========================
# 유틸 함수
# =========================
def check_required_columns(df, required_columns, file_label):
    """필수 컬럼이 없으면 명확한 오류를 낸다."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{file_label}에 필수 컬럼이 없습니다: {missing_columns}")


def make_bool_count(series, value=True):
    """True/False 컬럼의 개수를 안전하게 센다."""
    return (series == value).sum()


def safe_mean(series):
    """숫자 컬럼 평균을 안전하게 계산한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return 0
    return numeric.mean()


def main():
    print("=== job02 v4: Steam 게임별 리뷰 문서 만들기 시작 ===")

    # 1. CSV 파일 읽기
    print("[1/6] CSV 파일 읽는 중...")

    if not os.path.exists(REVIEWS_PATH):
        raise FileNotFoundError(f"리뷰 파일을 찾을 수 없습니다: {REVIEWS_PATH}")

    if not os.path.exists(GAMES_PATH):
        raise FileNotFoundError(f"게임 상세 파일을 찾을 수 없습니다: {GAMES_PATH}")

    df_reviews = pd.read_csv(REVIEWS_PATH, low_memory=False)
    df_games = pd.read_csv(GAMES_PATH, low_memory=False)

    print("리뷰 전체 행 수:", len(df_reviews))
    print("리뷰 appid 수:", df_reviews["appid"].nunique() if "appid" in df_reviews.columns else "appid 없음")
    print("게임 상세 정보 행 수:", len(df_games))
    print("게임 상세 appid 수:", df_games["appid"].nunique() if "appid" in df_games.columns else "appid 없음")

    # 2. 필수 컬럼 확인
    print("[2/6] 필수 컬럼 확인 중...")

    required_review_cols = ["appid", "game_title", "review"]
    required_game_cols = ["appid", "game_title"]

    check_required_columns(df_reviews, required_review_cols, "리뷰 파일")
    check_required_columns(df_games, required_game_cols, "게임 상세 파일")

    # 3. 리뷰 기본 정리
    print("[3/6] 리뷰 기본 정리 중...")

    # v2 수집 원칙은 한국어 리뷰만 저장하는 것이지만, 혹시 섞였을 경우를 대비해 다시 확인한다.
    if "language" in df_reviews.columns:
        before = len(df_reviews)
        df_reviews = df_reviews[df_reviews["language"].fillna("").astype(str).str.lower() == "koreana"].copy()
        after = len(df_reviews)
        print("한국어가 아닌 리뷰 제거:", before - after, "행")

    # review_id가 있으면 중복 리뷰 제거
    if "review_id" in df_reviews.columns:
        before = len(df_reviews)
        df_reviews = df_reviews.drop_duplicates(subset=["appid", "review_id"])
        after = len(df_reviews)
        print("중복 리뷰 제거:", before - after, "행")

    # 리뷰 본문 비어 있는 행 제거
    df_reviews["review"] = df_reviews["review"].fillna("").astype(str).str.strip()
    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["review"] != ""].copy()
    after = len(df_reviews)
    print("빈 리뷰 제거:", before - after, "행")

    # appid 타입 통일
    df_reviews["appid"] = pd.to_numeric(df_reviews["appid"], errors="coerce")
    df_games["appid"] = pd.to_numeric(df_games["appid"], errors="coerce")

    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["appid"].notna()].copy()
    df_reviews["appid"] = df_reviews["appid"].astype(int)
    print("appid 없는 리뷰 제거:", before - len(df_reviews), "행")

    before = len(df_games)
    df_games = df_games[df_games["appid"].notna()].copy()
    df_games["appid"] = df_games["appid"].astype(int)
    print("appid 없는 게임 상세 제거:", before - len(df_games), "행")

    # 4. appid 기준으로 리뷰 문서 만들기
    print("[4/6] appid 기준으로 리뷰 문서 생성 중...")
    print("리뷰 순서는 원본 CSV 순서를 유지합니다.")

    # 있으면 같이 집계할 리뷰 품질 컬럼들
    agg_dict = {
        "titles": ("game_title", "first"),
        "reviews": ("review", lambda x: "\n".join(x)),
        "review_count": ("review", "count"),
    }

    if "voted_up" in df_reviews.columns:
        agg_dict["voted_up_count"] = ("voted_up", lambda x: make_bool_count(x, True))
        agg_dict["voted_down_count"] = ("voted_up", lambda x: make_bool_count(x, False))

    if "votes_up" in df_reviews.columns:
        agg_dict["avg_votes_up"] = ("votes_up", safe_mean)
        agg_dict["max_votes_up"] = ("votes_up", "max")

    if "weighted_vote_score" in df_reviews.columns:
        agg_dict["avg_weighted_vote_score"] = ("weighted_vote_score", safe_mean)
        agg_dict["max_weighted_vote_score"] = ("weighted_vote_score", "max")

    if "playtime_at_review" in df_reviews.columns:
        agg_dict["avg_playtime_at_review"] = ("playtime_at_review", safe_mean)

    df_documents = (
        df_reviews
        .groupby("appid", as_index=False)
        .agg(**agg_dict)
    )

    if "voted_up_count" in df_documents.columns and "review_count" in df_documents.columns:
        df_documents["positive_ratio_collected"] = (
            df_documents["voted_up_count"] / df_documents["review_count"]
        ).round(4)

    print("문서화된 게임 수:", len(df_documents))

    # 5. 게임 상세 정보 붙이기
    print("[5/6] 게임 상세 정보 merge 중...")

    detail_cols = [
        "appid",
        "game_title",
        "release_date",
        "release_year",
        "genres",
        "tags",
        "categories",
        "review_score",
        "review_score_desc",
        "review_score_desc_ko",
        "total_positive",
        "total_negative",
        "total_reviews",
        "developers",
        "publishers",
        "supported_languages",
        "short_description",
        "price_currency",
        "price_initial",
        "price_final",
        "discount_percent",
        "platform_windows",
        "platform_mac",
        "platform_linux",
        "recommendations_total",
        "achievements_total",
    ]
    detail_cols = [col for col in detail_cols if col in df_games.columns]

    df_games_detail = df_games[detail_cols].drop_duplicates(subset=["appid"]).copy()

    # df_documents의 titles는 리뷰 파일의 제목이다.
    # 상세 파일의 game_title은 혹시 다를 수 있으므로 detail_game_title로 보존한다.
    if "game_title" in df_games_detail.columns:
        df_games_detail = df_games_detail.rename(columns={"game_title": "detail_game_title"})

    df_result = pd.merge(
        df_documents,
        df_games_detail,
        on="appid",
        how="left",
    )

    # 상세 정보 쪽 제목이 있고 리뷰 쪽 제목이 비어 있으면 보완한다.
    if "detail_game_title" in df_result.columns:
        df_result["titles"] = df_result["titles"].fillna(df_result["detail_game_title"])

    # 보기 좋은 컬럼 순서
    first_cols = [
        "appid",
        "titles",
        "reviews",
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
        "avg_votes_up",
        "max_votes_up",
        "avg_weighted_vote_score",
        "max_weighted_vote_score",
        "avg_playtime_at_review",
    ]

    first_cols = [col for col in first_cols if col in df_result.columns]
    other_cols = [col for col in df_result.columns if col not in first_cols]
    df_result = df_result[first_cols + other_cols]

    # 6. 저장 및 요약 출력
    print("[6/6] 결과 저장 중...")

    os.makedirs(DATA_DIR, exist_ok=True)
    df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    missing_detail_count = 0
    if "review_score_desc_ko" in df_result.columns:
        missing_detail_count = df_result["review_score_desc_ko"].isna().sum()

    print("\n=== job02 v4 완료 ===")
    print("저장 파일:", OUTPUT_PATH)
    print("최종 행 수:", len(df_result))
    print("최종 컬럼 수:", len(df_result.columns))
    print("상세 정보 누락 추정 행 수:", missing_detail_count)

    print("\n결과 예시:")
    preview_cols = ["appid", "titles", "review_count"]
    if "positive_ratio_collected" in df_result.columns:
        preview_cols.append("positive_ratio_collected")
    if "review_score_desc_ko" in df_result.columns:
        preview_cols.append("review_score_desc_ko")
    print(df_result[preview_cols].head())


if __name__ == "__main__":
    main()
