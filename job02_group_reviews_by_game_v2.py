import os
import pandas as pd


# ============================================================
# job02_group_reviews_by_game_v2.py
#
# 목적:
#   steam_reviews_raw_v2.csv  : 리뷰 1개 = 1행
#   steam_games_detail_v2.csv : 게임 1개 = 1행
#
#   위 두 파일을 appid 기준으로 합쳐서
#   게임 1개 = 1행 형태의 steam_game_reviews_grouped_v2.csv 생성
# ============================================================


DATA_DIR = "./datasets"

REVIEWS_PATH = os.path.join(DATA_DIR, "steam_reviews_raw_v2.csv")
GAMES_PATH = os.path.join(DATA_DIR, "steam_games_detail_v2.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "steam_game_reviews_grouped_v2.csv")


def main():
    print("=== job02: Steam 리뷰 게임별 묶기 시작 ===")

    # 1. 파일 읽기
    print("[1/5] CSV 파일 읽는 중...")
    df_reviews = pd.read_csv(REVIEWS_PATH, low_memory=False)
    df_games = pd.read_csv(GAMES_PATH, low_memory=False)

    print("리뷰 전체 행 수:", len(df_reviews))
    print("리뷰가 있는 게임 수:", df_reviews["appid"].nunique())
    print("게임 상세 정보 행 수:", len(df_games))
    print("게임 상세 정보 appid 수:", df_games["appid"].nunique())

    # 2. 필수 컬럼 확인
    print("[2/5] 필수 컬럼 확인 중...")
    required_review_cols = ["appid", "game_title", "review"]
    required_game_cols = ["appid", "release_year", "genres", "tags", "review_score_desc_ko"]

    for col in required_review_cols:
        if col not in df_reviews.columns:
            raise ValueError(f"리뷰 파일에 필요한 컬럼이 없습니다: {col}")

    for col in required_game_cols:
        if col not in df_games.columns:
            raise ValueError(f"게임 상세 파일에 필요한 컬럼이 없습니다: {col}")

    # 3. 리뷰 정리
    print("[3/5] 리뷰 본문 정리 중...")

    # review_id가 있으면 중복 리뷰 제거
    if "review_id" in df_reviews.columns:
        before = len(df_reviews)
        df_reviews = df_reviews.drop_duplicates(subset=["appid", "review_id"])
        after = len(df_reviews)
        print("중복 리뷰 제거:", before - after, "행")

    # 리뷰가 비어 있는 행 제거
    df_reviews["review"] = df_reviews["review"].fillna("").astype(str).str.strip()
    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["review"] != ""].copy()
    after = len(df_reviews)
    print("빈 리뷰 제거:", before - after, "행")

    # 4. appid 기준으로 리뷰 묶기
    print("[4/5] appid 기준으로 리뷰 묶는 중...")

    df_grouped = (
        df_reviews
        .groupby("appid", as_index=False)
        .agg(
            titles=("game_title", "first"),
            reviews=("review", lambda x: "\n".join(x)),
            review_count=("review", "count")
        )
    )

    print("묶은 뒤 게임 수:", len(df_grouped))

    # 5. 게임 상세 정보 붙이기
    print("[5/5] 게임 상세 정보 merge 중...")

    detail_cols = [
        "appid",
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

    # 실제 파일에 존재하는 컬럼만 사용
    detail_cols = [col for col in detail_cols if col in df_games.columns]

    df_games_detail = df_games[detail_cols].drop_duplicates(subset=["appid"])

    df_result = pd.merge(
        df_grouped,
        df_games_detail,
        on="appid",
        how="left"
    )

    # 보기 좋은 컬럼 순서
    first_cols = [
        "appid",
        "titles",
        "reviews",
        "review_count",
        "release_year",
        "genres",
        "tags",
        "review_score_desc_ko",
        "total_positive",
        "total_negative",
        "total_reviews",
        "developers",
        "publishers",
        "short_description",
    ]

    first_cols = [col for col in first_cols if col in df_result.columns]
    other_cols = [col for col in df_result.columns if col not in first_cols]
    df_result = df_result[first_cols + other_cols]

    # 저장 폴더 생성
    os.makedirs(DATA_DIR, exist_ok=True)

    # 저장
    df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=== job02 완료 ===")
    print("저장 파일:", OUTPUT_PATH)
    print("최종 행 수:", len(df_result))
    print("최종 컬럼 수:", len(df_result.columns))
    print()
    print(df_result[["appid", "titles", "review_count"]].head())


if __name__ == "__main__":
    main()
