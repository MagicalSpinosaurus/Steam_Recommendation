# job06_steam_recommendation_ui_v8.py
# 실행 확인 문구: 콘솔에 [job06 v8 실행 중] 이 보이면 새 파일이 실행되는 것입니다.
# 조건:
# 1) 좋아하는 게임 입력 + 부분 문자열 자동완성
# 2) 장르 선택
# 3) 키워드 입력
# 4) 추천받기 버튼
# 5) 세 입력값 중 하나만 있어도 추천 가능

import os
import re
import math
import unicodedata
import webbrowser
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


DATA_DIR = "./datasets"

INDEX_FILE_NAME = "steam_recommendation_index_v4.csv"
PREPROCESSED_FILE_NAME = "steam_game_reviews_preprocessed_v4.csv"

MAX_SUGGESTIONS = 20
RECOMMEND_COUNT = 10

# 영어 제목만 있는 게임을 한글 일부 검색으로 찾기 위한 별칭 파일.
# 없어도 실행된다.
# 만들 경우 컬럼 예:
# title,aliases
# Stardew Valley,스타듀밸리|스타듀|스듀|밸리
ALIAS_FILE_NAME = "steam_title_aliases_v4.csv"

DEFAULT_KOREAN_ALIASES = {
    "stardewvalley": ["스타듀밸리", "스타듀", "스듀", "밸리"],
    "terraria": ["테라리아"],
    "hades": ["하데스"],
    "hollowknight": ["할로우나이트", "할나"],
    "dontstarve": ["돈스타브", "굶지마"],
    "dontstarvetogether": ["돈스타브투게더", "굶지마투게더", "굶지마"],
    "slaythespire": ["슬레이더스파이어", "슬더스"],
    "overcooked": ["오버쿡드"],
    "overcooked2": ["오버쿡드2"],
    "humanfallflat": ["휴먼폴플랫"],
    "phasmophobia": ["파스모포비아"],
    "amongus": ["어몽어스"],
    "deadbydaylight": ["데드바이데이라이트", "데바데"],
    "deadcells": ["데드셀"],
    "celeste": ["셀레스트"],
    "cuphead": ["컵헤드"],
    "raft": ["래프트"],
    "valheim": ["발헤임"],
    "subnautica": ["서브노티카"],
    "theforest": ["더포레스트", "포레스트"],
    "rimworld": ["림월드"],
    "factorio": ["팩토리오"],
    "portal": ["포탈"],
    "portal2": ["포탈2"],
    "left4dead2": ["레프트4데드2", "레포데2"],
    "counterstrike2": ["카운터스트라이크2", "카스2"],
    "dota2": ["도타2"],
    "rust": ["러스트"],
    "eldenring": ["엘든링"],
    "cyberpunk2077": ["사이버펑크2077", "사펑"],
    "thewitcher3wildhunt": ["위쳐3", "더위쳐3"],
    "monsterhunterworld": ["몬스터헌터월드", "몬헌월드"],
    "monsterhunterrise": ["몬스터헌터라이즈", "몬헌라이즈"],
    "civilizationvi": ["문명6", "시드마이어문명6"],
}


KEYWORD_ALIASES = {
    "1인": ["1인", "싱글", "싱글플레이어"],
    "싱글": ["싱글", "싱글플레이어", "1인"],
    "싱글플레이어": ["싱글플레이어", "싱글", "1인"],
    "협동": ["협동", "온라인협동"],
    "멀티": ["멀티", "멀티플레이어"],
    "멀티플레이어": ["멀티플레이어", "멀티"],
    "무료": ["무료", "무료플레이"],
    "무료플레이": ["무료플레이", "무료"],
}


def resolve_file(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(base_dir, DATA_DIR, file_name),
        os.path.join(base_dir, file_name),
        os.path.join(DATA_DIR, file_name),
        file_name,
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w가-힣]", "", text, flags=re.UNICODE)
    text = text.replace("_", "")
    return text


EXCLUDED_GENRE_KEYS = {
    normalize_text("앞서 해보기"),
    normalize_text("무료 플레이"),
    normalize_text("무료플레이"),
}


def split_csv_text(value):
    if pd.isna(value):
        return []

    result = []

    for item in str(value).split(","):
        item = item.strip()
        if item and item.lower() != "nan":
            result.append(item)

    return result


def safe_numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_alias_map():
    alias_map = dict(DEFAULT_KOREAN_ALIASES)

    alias_path = resolve_file(ALIAS_FILE_NAME)
    if alias_path is None:
        return alias_map

    try:
        df_alias = pd.read_csv(alias_path)
    except Exception:
        return alias_map

    if "title" not in df_alias.columns or "aliases" not in df_alias.columns:
        return alias_map

    for _, row in df_alias.iterrows():
        title_key = normalize_text(row.get("title", ""))
        aliases_text = str(row.get("aliases", ""))

        if not title_key or not aliases_text:
            continue

        aliases = re.split(r"[|,]", aliases_text)
        aliases = [alias.strip() for alias in aliases if alias.strip()]

        if title_key not in alias_map:
            alias_map[title_key] = []

        alias_map[title_key].extend(aliases)

    return alias_map


def read_game_data():
    index_path = resolve_file(INDEX_FILE_NAME)
    if index_path is None:
        raise FileNotFoundError(
            "steam_recommendation_index_v4.csv 파일을 찾을 수 없습니다."
        )

    df = pd.read_csv(index_path, low_memory=False)

    if "appid" not in df.columns:
        raise ValueError("steam_recommendation_index_v4.csv에 appid 컬럼이 없습니다.")

    if "titles" not in df.columns:
        raise ValueError("steam_recommendation_index_v4.csv에 titles 컬럼이 없습니다.")

    df["appid"] = safe_numeric(df["appid"], default=-1).astype(int)
    df["titles"] = df["titles"].fillna("").astype(str).str.strip()
    df = df[(df["appid"] != -1) & (df["titles"] != "")].copy()
    df = df.drop_duplicates(subset=["appid"]).reset_index(drop=True)

    needed_cols = [
        "genres", "tags", "categories", "short_description",
        "review_count", "positive_ratio_collected", "release_year"
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    preprocessed_path = resolve_file(PREPROCESSED_FILE_NAME)
    if preprocessed_path is not None:
        df_pre = pd.read_csv(
            preprocessed_path,
            usecols=lambda col: col in {"appid", "reviews"},
            low_memory=False,
        )

        if "appid" in df_pre.columns and "reviews" in df_pre.columns:
            df_pre["appid"] = safe_numeric(df_pre["appid"], default=-1).astype(int)
            df_pre = df_pre[df_pre["appid"] != -1].drop_duplicates(subset=["appid"])
            df_pre = df_pre.rename(columns={"reviews": "preprocessed_reviews"})

            df = pd.merge(
                df,
                df_pre[["appid", "preprocessed_reviews"]],
                on="appid",
                how="left"
            )

    if "preprocessed_reviews" not in df.columns:
        df["preprocessed_reviews"] = ""

    df["preprocessed_reviews"] = df["preprocessed_reviews"].fillna("").astype(str)

    return df.reset_index(drop=True)


def collect_genres(df):
    genres = set()

    for value in df["genres"].fillna(""):
        for genre in split_csv_text(value):
            if normalize_text(genre) in EXCLUDED_GENRE_KEYS:
                continue
            genres.add(genre)

    return sorted(genres)


def build_search_records(df, alias_map):
    records = []

    for idx, row in df.iterrows():
        title = str(row["titles"]).strip()
        title_key = normalize_text(title)

        keys = {title_key}

        for alias in alias_map.get(title_key, []):
            alias_key = normalize_text(alias)
            if alias_key:
                keys.add(alias_key)

        records.append({
            "row_index": idx,
            "title": title,
            "title_key": title_key,
            "keys": list(keys),
        })

    return records


def get_match_score(record, query_key):
    if not query_key:
        return None

    title_key = record["title_key"]
    keys = record["keys"]

    if title_key == query_key:
        return 0

    if query_key in keys:
        return 1

    if title_key.startswith(query_key):
        return 2

    if any(key.startswith(query_key) for key in keys):
        return 3

    if query_key in title_key:
        return 4

    if any(query_key in key for key in keys):
        return 5

    return None


def extract_years(keyword_text):
    years = []

    for number_text in re.findall(r"\d{2,4}", keyword_text):
        number = int(number_text)

        if len(number_text) == 2:
            if number <= 30:
                year = 2000 + number
            else:
                year = 1900 + number
        else:
            year = number

        if 1980 <= year <= 2030:
            years.append(year)

    return sorted(set(years))


def keyword_terms(keyword_text):
    raw_terms = re.split(r"[\s,;/|]+", keyword_text.strip())
    terms = []

    for term in raw_terms:
        key = normalize_text(term)
        if not key:
            continue

        expanded = KEYWORD_ALIASES.get(key, [key])

        for item in expanded:
            item_key = normalize_text(item)
            if item_key and item_key not in terms:
                terms.append(item_key)

    return terms


class SteamRecommendationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steam Game Recommendation UI v8")

        self.df = read_game_data()
        self.alias_map = load_alias_map()
        self.search_records = build_search_records(self.df, self.alias_map)
        self.genres = collect_genres(self.df)

        self.game_var = tk.StringVar()
        self.genre_var = tk.StringVar()
        self.keyword_var = tk.StringVar()

        self.tfidf = None
        self.tfidf_matrix = None
        self.result_item_indices = []

        self.prepare_search_columns()
        self.prepare_tfidf()
        self.make_widgets()

    def prepare_search_columns(self):
        self.df["_title_norm"] = self.df["titles"].apply(normalize_text)
        self.df["_genres_norm"] = self.df["genres"].fillna("").apply(normalize_text)
        self.df["_tags_norm"] = self.df["tags"].fillna("").apply(normalize_text)
        self.df["_categories_norm"] = self.df["categories"].fillna("").apply(normalize_text)
        self.df["_description_norm"] = self.df["short_description"].fillna("").apply(normalize_text)
        self.df["_reviews_norm"] = self.df["preprocessed_reviews"].fillna("").apply(normalize_text)

        self.df["_metadata_blob"] = (
            self.df["_title_norm"] + " " +
            self.df["_genres_norm"] + " " +
            self.df["_tags_norm"] + " " +
            self.df["_categories_norm"] + " " +
            self.df["_description_norm"]
        )

        self.df["_full_blob"] = (
            self.df["_metadata_blob"] + " " + self.df["_reviews_norm"]
        )

        self.df["_review_count_num"] = safe_numeric(self.df["review_count"], default=0)
        self.df["_positive_ratio_num"] = safe_numeric(self.df["positive_ratio_collected"], default=0.5)
        self.df["_release_year_num"] = safe_numeric(self.df["release_year"], default=0).astype(int)

    def prepare_tfidf(self):
        if TfidfVectorizer is None or cosine_similarity is None:
            return

        texts = self.df["preprocessed_reviews"].fillna("").astype(str).str.strip()

        if (texts != "").sum() == 0:
            return

        try:
            self.tfidf = TfidfVectorizer(sublinear_tf=True)
            self.tfidf_matrix = self.tfidf.fit_transform(texts)
        except Exception:
            self.tfidf = None
            self.tfidf_matrix = None

    def make_widgets(self):
        main = ttk.Frame(self.root, padding=15)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        ttk.Label(main, text="좋아하는 게임").grid(row=0, column=0, sticky="w")

        self.game_entry = ttk.Entry(main, textvariable=self.game_var)
        self.game_entry.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.game_entry.bind("<KeyRelease>", self.update_suggestions)
        self.game_entry.bind("<FocusIn>", self.update_suggestions)

        self.suggestion_list = tk.Listbox(main, height=8)
        self.suggestion_list.grid(row=2, column=0, sticky="ew", pady=(5, 15))
        self.suggestion_list.bind("<<ListboxSelect>>", self.select_suggestion)

        ttk.Label(main, text="장르").grid(row=3, column=0, sticky="w")

        self.genre_combo = ttk.Combobox(
            main,
            textvariable=self.genre_var,
            values=self.genres,
            state="readonly"
        )
        self.genre_combo.grid(row=4, column=0, sticky="ew", pady=(5, 15))

        ttk.Label(main, text="키워드").grid(row=5, column=0, sticky="w")

        self.keyword_entry = ttk.Entry(main, textvariable=self.keyword_var)
        self.keyword_entry.grid(row=6, column=0, sticky="ew", pady=(5, 15))

        self.recommend_button = ttk.Button(
            main,
            text="추천받기",
            command=self.recommend_games
        )
        self.recommend_button.grid(row=7, column=0, sticky="ew", pady=(0, 15))

        self.result_list = tk.Listbox(main, height=10)
        self.result_list.grid(row=8, column=0, sticky="nsew")
        self.result_list.bind("<Double-Button-1>", self.show_selected_game_detail)

        main.rowconfigure(8, weight=1)

    def get_suggestions(self, query_text):
        query_key = normalize_text(query_text)

        if not query_key:
            return []

        matched = []

        for record in self.search_records:
            score = get_match_score(record, query_key)
            if score is not None:
                matched.append((score, record["title"].casefold(), record))

        if not matched and len(query_key) >= 4:
            for record in self.search_records:
                best_ratio = 0

                for key in record["keys"]:
                    ratio = SequenceMatcher(None, query_key, key).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio

                if best_ratio >= 0.62:
                    matched.append((10 - best_ratio, record["title"].casefold(), record))

        matched.sort(key=lambda x: (x[0], x[1]))

        return [record for _, _, record in matched[:MAX_SUGGESTIONS]]

    def update_suggestions(self, event=None):
        self.suggestion_list.delete(0, tk.END)

        suggestions = self.get_suggestions(self.game_var.get())

        for record in suggestions:
            self.suggestion_list.insert(tk.END, record["title"])

    def select_suggestion(self, event=None):
        selected = self.suggestion_list.curselection()

        if not selected:
            return

        title = self.suggestion_list.get(selected[0])
        self.game_var.set(title)

        self.suggestion_list.delete(0, tk.END)
        self.suggestion_list.insert(tk.END, title)

    def find_seed_game_index(self, game_text):
        query_key = normalize_text(game_text)

        if not query_key:
            return None

        exact_matches = []

        for record in self.search_records:
            title_key = record["title_key"]
            keys = record["keys"]

            if title_key == query_key or query_key in keys:
                exact_matches.append(record)

        if exact_matches:
            return exact_matches[0]["row_index"]

        suggestions = self.get_suggestions(game_text)
        if suggestions:
            return suggestions[0]["row_index"]

        return None

    def base_popularity_score(self):
        review_counts = self.df["_review_count_num"].fillna(0).astype(float)
        positive_ratio = self.df["_positive_ratio_num"].fillna(0.5).astype(float)

        max_log_review = math.log1p(review_counts.max()) if review_counts.max() > 0 else 1

        review_score = review_counts.apply(lambda x: math.log1p(x) / max_log_review if max_log_review else 0)
        rating_score = positive_ratio.clip(lower=0, upper=1)

        return 0.15 * review_score + 0.15 * rating_score

    def add_game_score(self, scores, seed_idx, game_text):
        scores = scores.copy()
        game_key = normalize_text(game_text)

        if seed_idx is not None:
            if self.tfidf_matrix is not None:
                try:
                    similarity = cosine_similarity(
                        self.tfidf_matrix[seed_idx],
                        self.tfidf_matrix
                    ).ravel()

                    scores += similarity * 3.0
                    return scores
                except Exception:
                    pass

            seed_row = self.df.iloc[seed_idx]
            seed_terms = []

            for col in ["_genres_norm", "_tags_norm", "_categories_norm"]:
                value = str(seed_row.get(col, ""))
                if value:
                    seed_terms.append(value)

            for idx, row in self.df.iterrows():
                blob = row["_metadata_blob"]
                for term in seed_terms:
                    if term and term in blob:
                        scores[idx] += 0.4

            return scores

        if game_key:
            for idx, row in self.df.iterrows():
                if game_key in row["_title_norm"]:
                    scores[idx] += 1.5
                elif game_key in row["_full_blob"]:
                    scores[idx] += 0.4

        return scores

    def add_genre_score(self, scores, genre_text):
        scores = scores.copy()
        genre_key = normalize_text(genre_text)

        if not genre_key:
            return scores

        for idx, row in self.df.iterrows():
            if genre_key in row["_genres_norm"]:
                scores[idx] += 2.0
            if genre_key in row["_tags_norm"]:
                scores[idx] += 0.6
            if genre_key in row["_categories_norm"]:
                scores[idx] += 0.4

        return scores

    def add_keyword_score(self, scores, keyword_text):
        scores = scores.copy()
        if not keyword_text.strip():
            return scores

        years = extract_years(keyword_text)
        terms = keyword_terms(keyword_text)

        for idx, row in self.df.iterrows():
            for year in years:
                if int(row["_release_year_num"]) == year:
                    scores[idx] += 2.0

            for term in terms:
                if not term:
                    continue

                if term in row["_metadata_blob"]:
                    scores[idx] += 1.2
                elif term in row["_reviews_norm"]:
                    scores[idx] += 0.5

        return scores

    def recommend_games(self):
        game_text = self.game_var.get().strip()
        genre_text = self.genre_var.get().strip()
        keyword_text = self.keyword_var.get().strip()

        if not game_text and not genre_text and not keyword_text:
            messagebox.showwarning(
                "입력 필요",
                "좋아하는 게임, 장르, 키워드 중 하나 이상 입력하세요."
            )
            return

        seed_idx = self.find_seed_game_index(game_text) if game_text else None

        scores = self.base_popularity_score()
        scores = scores.astype(float).to_numpy(copy=True)

        scores = self.add_game_score(scores, seed_idx, game_text)
        scores = self.add_genre_score(scores, genre_text)
        scores = self.add_keyword_score(scores, keyword_text)

        if seed_idx is not None:
            scores[seed_idx] = -1

        result_df = self.df.copy()
        result_df["_score"] = scores

        result_df = result_df[result_df["_score"] > 0].copy()
        result_df = result_df.sort_values("_score", ascending=False).head(RECOMMEND_COUNT)

        self.result_list.delete(0, tk.END)
        self.result_item_indices = []

        if len(result_df) == 0:
            self.result_list.insert(tk.END, "추천 결과 없음")
            return

        for row_index, row in result_df.iterrows():
            self.result_item_indices.append(row_index)
            self.result_list.insert(tk.END, row["titles"])


    def show_selected_game_detail(self, event=None):
        selected = self.result_list.curselection()

        if not selected:
            return

        list_index = selected[0]

        if list_index >= len(self.result_item_indices):
            return

        row_index = self.result_item_indices[list_index]
        row = self.df.loc[row_index]

        appid = int(row.get("appid", 0))
        title = str(row.get("titles", "")).strip()
        genres = str(row.get("genres", "")).strip()
        tags = str(row.get("tags", "")).strip()
        release_year = str(row.get("release_year", "")).strip()
        description = str(row.get("short_description", "")).strip()

        if not description or description.lower() == "nan":
            description = "간단한 설명 데이터가 없습니다."

        if not release_year or release_year.lower() == "nan":
            release_year = "정보 없음"

        detail_window = tk.Toplevel(self.root)
        detail_window.title(title)
        detail_window.geometry("520x360")

        frame = ttk.Frame(detail_window, padding=15)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)

        ttk.Label(frame, text=title, font=("Arial", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        ttk.Label(frame, text=f"장르: {genres if genres else '정보 없음'}").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )

        ttk.Label(frame, text=f"출시연도: {release_year}").grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )

        ttk.Label(frame, text=f"연관검색어: {tags if tags else '정보 없음'}", wraplength=480).grid(
            row=3, column=0, sticky="w", pady=(0, 10)
        )

        description_box = tk.Text(frame, height=7, wrap="word")
        description_box.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        description_box.insert("1.0", description)
        description_box.config(state="disabled")

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=5, column=0, sticky="ew")

        ttk.Button(
            button_frame,
            text="스팀 페이지 열기",
            command=lambda: self.open_steam_page(appid)
        ).pack(side="left")

        ttk.Button(
            button_frame,
            text="닫기",
            command=detail_window.destroy
        ).pack(side="right")

    def open_steam_page(self, appid):
        if not appid:
            messagebox.showwarning("오류", "AppID가 없어 스팀 페이지를 열 수 없습니다.")
            return

        webbrowser.open(f"https://store.steampowered.com/app/{appid}/")


def main():
    print("[job06 v8 실행 중] 추천받기 버튼 + 연관검색어 표시 + Enter 추천 + 더블클릭 상세보기 파일입니다.")
    try:
        root = tk.Tk()
        app = SteamRecommendationUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("오류", str(e))


if __name__ == "__main__":
    main()
