import os
import pandas as pd

os.makedirs("../Steam_Game_Recommendation/datasets", exist_ok=True)

input_path = "../Steam_Game_Recommendation/datasets/steam_reviews_raw_large_balanced.csv"
output_path = "../Steam_Game_Recommendation/datasets/steam_game_reviews_grouped.csv"

df = pd.read_csv(input_path)

print("원본 데이터 정보")
print("=" * 60)
df.info()

print()
print("원본 전체 리뷰 수:", len(df))
print("원본 게임 수:", df["appid"].nunique())

# 필요한 컬럼 확인
needed_columns = ["appid", "game_title", "review"]

for col in needed_columns:
    if col not in df.columns:
        raise ValueError(f"필요한 컬럼이 없습니다: {col}")

# 리뷰가 비어 있는 행 제거
df = df.dropna(subset=["review"])
df["review"] = df["review"].astype(str)

# 너무 짧은 리뷰는 제거
df = df[df["review"].str.len() >= 5]

print()
print("리뷰 정리 후 리뷰 수:", len(df))
print("리뷰 정리 후 게임 수:", df["appid"].nunique())

# 가능하면 좋은 리뷰가 앞에 오도록 정렬
# 컬럼이 없을 수도 있으므로 있는 컬럼만 사용
sort_columns = []
ascending_values = []

if "appid" in df.columns:
    sort_columns.append("appid")
    ascending_values.append(True)

if "weighted_vote_score" in df.columns:
    df["weighted_vote_score"] = pd.to_numeric(df["weighted_vote_score"], errors="coerce").fillna(0)
    sort_columns.append("weighted_vote_score")
    ascending_values.append(False)

if "votes_up" in df.columns:
    df["votes_up"] = pd.to_numeric(df["votes_up"], errors="coerce").fillna(0)
    sort_columns.append("votes_up")
    ascending_values.append(False)

if "timestamp_created" in df.columns:
    df["timestamp_created"] = pd.to_numeric(df["timestamp_created"], errors="coerce").fillna(0)
    sort_columns.append("timestamp_created")
    ascending_values.append(False)

df = df.sort_values(by=sort_columns, ascending=ascending_values)

# 게임별 리뷰 합치기
df_grouped = (
    df.groupby(["appid", "game_title"])
      .agg(
          reviews=("review", lambda x: " ".join(x)),
          review_count=("review", "count")
      )
      .reset_index()
)

# 기존 영화 추천 코드와 맞추기 위해 titles 컬럼 생성
df_grouped.rename(columns={"game_title": "titles"}, inplace=True)

# 컬럼 순서 정리
df_grouped = df_grouped[["appid", "titles", "reviews", "review_count"]]

print()
print("=" * 60)
print("게임별 그룹화 결과")
print("=" * 60)

print("그룹화 후 게임 수:", len(df_grouped))
print(df_grouped.head())

df_grouped.to_csv(output_path, index=False, encoding="utf-8-sig")

print()
print("저장 완료:", output_path)