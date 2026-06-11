import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite

# ============================================================
# 0. 기본 설정
# ============================================================

os.makedirs("../Steam_Game_Recommendation/datasets", exist_ok=True)
os.makedirs("../Steam_Game_Recommendation/models", exist_ok=True)

input_path = "../Steam_Game_Recommendation/datasets/steam_game_reviews_preprocessed.csv"

tfidf_model_path = "../Steam_Game_Recommendation/models/tfidf_steam_review.pkl"
tfidf_matrix_path = "../Steam_Game_Recommendation/models/Tfidf_steam_review.mtx"

# ============================================================
# 1. 데이터 불러오기
# ============================================================

df_reviews = pd.read_csv(input_path)

print("=" * 60)
print("전처리 데이터 정보")
print("=" * 60)
df_reviews.info()

needed_columns = ["appid", "titles", "reviews", "review_count"]

for col in needed_columns:
    if col not in df_reviews.columns:
        raise ValueError(f"필요한 컬럼이 없습니다: {col}")

df_reviews = df_reviews.dropna(subset=["reviews"])
df_reviews["reviews"] = df_reviews["reviews"].astype(str)

# 빈 문자열 제거
df_reviews = df_reviews[df_reviews["reviews"].str.strip() != ""]
df_reviews.reset_index(drop=True, inplace=True)

print()
print("TF-IDF 대상 게임 수:", len(df_reviews))
print("appid 고유 개수:", df_reviews["appid"].nunique())
print("titles 고유 개수:", df_reviews["titles"].nunique())

print()
print("=" * 60)
print("TF-IDF 입력 예시")
print("=" * 60)
print("제목:", df_reviews.iloc[0]["titles"])
print(df_reviews.iloc[0]["reviews"][:500])

# ============================================================
# 2. TF-IDF 벡터화
# ============================================================

print()
print("=" * 60)
print("TF-IDF 학습 시작")
print("=" * 60)

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews["reviews"])

print("TF-IDF matrix shape:", Tfidf_matrix.shape)
print("단어 수:", len(Tfidf.vocabulary_))

# ============================================================
# 3. 저장
# ============================================================

with open(tfidf_model_path, "wb") as f:
    pickle.dump(Tfidf, f)

mmwrite(tfidf_matrix_path, Tfidf_matrix)

print()
print("=" * 60)
print("TF-IDF 저장 완료")
print("=" * 60)
print("TF-IDF 모델:", tfidf_model_path)
print("TF-IDF 행렬:", tfidf_matrix_path)

print()
print("상위 단어 예시:")
feature_names = Tfidf.get_feature_names_out()
print(feature_names[:100])