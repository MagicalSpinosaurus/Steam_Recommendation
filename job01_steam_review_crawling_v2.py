import os
import re
import csv
import time
import html as html_lib
import datetime
import requests
import pandas as pd

# ============================================================
# 0. 기본 설정
# ============================================================

os.makedirs("./datasets", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# ------------------------------------------------------------
# Steam 상점 검색 필터
# ------------------------------------------------------------
# category1=998         : 게임 카테고리
# supportedlang=koreana : 한국어 지원 게임
STORE_LANGUAGE = "koreana"
REVIEW_LANGUAGE = "koreana"
CATEGORY_GAME = "998"

SEARCH_COUNT = 100
MAX_SEARCH_PAGES = None

# ------------------------------------------------------------
# 리뷰 수집 설정
# ------------------------------------------------------------
# 게임당 최대 300개 수집
MAX_REVIEWS_PER_GAME = 300
MAX_REVIEW_PAGES_PER_GAME = 3
REVIEWS_PER_PAGE = 100

# 리뷰 정렬 기준
# all + day_range=365: 최근 365일 범위에서 helpfulness 기준 상위 노출 리뷰 우선
# 만약 해당 조건에서 한국어 리뷰가 없으면 recent로 한 번 더 확인할 수 있게 함
REVIEW_FILTER = "all"
DAY_RANGE = 365
FALLBACK_TO_RECENT_IF_EMPTY = True

MIN_REVIEW_CHARS = 5

# 요청 간격
SEARCH_SLEEP_SEC = 0.5
DETAIL_SLEEP_SEC = 0.4
REVIEW_SLEEP_SEC = 0.8
GAME_SLEEP_SEC = 0.5

# 요청 실패 시 재시도
MAX_RETRY = 3
RETRY_SLEEP_SEC = 5
REQUEST_TIMEOUT = 20

# ------------------------------------------------------------
# 저장 파일 경로
# ------------------------------------------------------------
APP_LIST_PATH = "./datasets/steam_koreana_supported_games_v2.csv"
GAME_DETAIL_PATH = "./datasets/steam_games_detail_v2.csv"
RAW_REVIEW_PATH = "./datasets/steam_reviews_raw_v2.csv"
PROGRESS_LOG_PATH = "./datasets/steam_crawling_progress_v2.csv"

# 기존 v1의 한국어 지원 게임 목록이 있으면 재사용 가능
OLD_APP_LIST_PATH = "./datasets/steam_koreana_supported_games.csv"

# True면 기존 app list CSV가 있어도 다시 수집
REFRESH_APP_LIST = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 Steam review collector for personal Python ML project",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


# ============================================================
# 1. 사용자 평가 매핑
# ============================================================

REVIEW_SCORE_DESC_KO = {
    "Overwhelmingly Positive": "압도적으로 긍정적",
    "Very Positive": "매우 긍정적",
    "Positive": "긍정적",
    "Mostly Positive": "대체로 긍정적",
    "Mixed": "복합적",
    "Mostly Negative": "대체로 부정적",
    "Negative": "부정적",
    "Very Negative": "매우 부정적",
    "Overwhelmingly Negative": "압도적으로 부정적",

    # 혹시 이미 한국어로 들어올 경우 그대로 처리하기 위한 값
    "압도적으로 긍정적": "압도적으로 긍정적",
    "매우 긍정적": "매우 긍정적",
    "긍정적": "긍정적",
    "대체로 긍정적": "대체로 긍정적",
    "복합적": "복합적",
    "대체로 부정적": "대체로 부정적",
    "부정적": "부정적",
    "매우 부정적": "매우 부정적",
    "압도적으로 부정적": "압도적으로 부정적",
}


# ============================================================
# 2. 공통 유틸 함수
# ============================================================

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = html_lib.unescape(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_html_tags(text):
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def unix_to_datetime(timestamp_value):
    try:
        return datetime.datetime.fromtimestamp(
            int(timestamp_value)
        ).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ""


def extract_year_from_release_date(release_date):
    release_date = str(release_date)

    match = re.search(r"(19|20)\d{2}", release_date)
    if match:
        return int(match.group(0))

    return ""


def list_to_text(items, key="description"):
    if not items:
        return ""

    result = []

    for item in items:
        if isinstance(item, dict):
            value = item.get(key, "")
        else:
            value = str(item)

        value = clean_text(value)

        if value:
            result.append(value)

    return ", ".join(result)


def request_get_with_retry(url, params=None):
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 429:
                print("429 Too Many Requests. 잠시 대기합니다.")
                time.sleep(RETRY_SLEEP_SEC * attempt)
                continue

            response.raise_for_status()
            return response

        except Exception as e:
            last_error = e
            print(f"요청 실패 {attempt}/{MAX_RETRY}:", e)
            time.sleep(RETRY_SLEEP_SEC * attempt)

    raise last_error


def request_json_with_retry(url, params=None):
    response = request_get_with_retry(url, params=params)
    return response.json()


# ============================================================
# 3. Steam 한국어 지원 게임 목록 수집
# ============================================================

def parse_search_results_html(results_html):
    apps = []

    row_pattern = re.compile(
        r'<a[^>]*class="[^"]*search_result_row[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE
    )

    appid_pattern_1 = re.compile(
        r'href="(?:https?:)?//store\.steampowered\.com/app/(\d+)/',
        re.DOTALL | re.IGNORECASE
    )

    appid_pattern_2 = re.compile(
        r'data-ds-appid="(\d+)"',
        re.DOTALL | re.IGNORECASE
    )

    title_pattern = re.compile(
        r'<span[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</span>',
        re.DOTALL | re.IGNORECASE
    )

    for row_match in row_pattern.finditer(results_html):
        row_html = row_match.group(0)

        appid_match = appid_pattern_1.search(row_html)

        if appid_match:
            appid = appid_match.group(1)
        else:
            appid_match = appid_pattern_2.search(row_html)
            if appid_match:
                appid = appid_match.group(1)
            else:
                continue

        title_match = title_pattern.search(row_html)

        if title_match:
            game_title = remove_html_tags(title_match.group(1))
        else:
            game_title = ""

        if appid and game_title:
            apps.append({
                "appid": int(appid),
                "game_title": game_title,
                "source": "steam_search_supportedlang_koreana",
            })

    return apps


def save_app_list(apps):
    if len(apps) == 0:
        return

    df = pd.DataFrame(apps)
    df.drop_duplicates(subset=["appid"], inplace=True)
    df.sort_values("appid", inplace=True)
    df.to_csv(APP_LIST_PATH, index=False, encoding="utf-8-sig")


def load_app_list():
    if os.path.exists(APP_LIST_PATH):
        df = pd.read_csv(APP_LIST_PATH)
        df.drop_duplicates(subset=["appid"], inplace=True)
        df["appid"] = df["appid"].astype(int)
        return df

    if os.path.exists(OLD_APP_LIST_PATH):
        print("기존 v1 한국어 지원 게임 목록을 v2 목록으로 복사 사용합니다.")
        df = pd.read_csv(OLD_APP_LIST_PATH)
        df.drop_duplicates(subset=["appid"], inplace=True)
        df["appid"] = df["appid"].astype(int)

        if "source" not in df.columns:
            df["source"] = "steam_search_supportedlang_koreana"

        df.to_csv(APP_LIST_PATH, index=False, encoding="utf-8-sig")
        return df

    return None


def crawl_koreana_supported_app_list():
    print()
    print("=" * 80)
    print("Steam 한국어 지원 게임 목록 수집 시작")
    print("=" * 80)

    search_url = "https://store.steampowered.com/search/results/"

    all_apps_by_id = {}
    start = 0
    page = 1
    total_count = None

    while True:
        if MAX_SEARCH_PAGES is not None and page > MAX_SEARCH_PAGES:
            print("MAX_SEARCH_PAGES에 도달하여 게임 목록 수집을 중단합니다.")
            break

        params = {
            "query": "",
            "term": "",
            "start": start,
            "count": SEARCH_COUNT,
            "dynamic_data": "",
            "sort_by": "_ASC",
            "category1": CATEGORY_GAME,
            "supportedlang": STORE_LANGUAGE,
            "force_infinite": 1,
            "infinite": 1,
            "ignore_preferences": 1,
            "l": "koreana",
            "cc": "KR",
            "ndl": 1,
        }

        try:
            response = request_get_with_retry(search_url, params=params)

            try:
                data = response.json()
                results_html = data.get("results_html", "")
                total_count = data.get("total_count", total_count)

            except:
                results_html = response.text

        except Exception as e:
            print("검색 결과 요청 실패:", e)
            break

        apps = parse_search_results_html(results_html)

        if len(apps) == 0:
            print("검색 결과에서 더 이상 AppID를 찾지 못했습니다.")
            break

        new_count = 0

        for app in apps:
            appid = int(app["appid"])

            if appid not in all_apps_by_id:
                all_apps_by_id[appid] = app
                new_count += 1

        print(
            f"검색 페이지 {page} / start={start} / "
            f"이번 페이지 {len(apps)}개 / 신규 {new_count}개 / "
            f"누적 {len(all_apps_by_id)}개 / total_count={total_count}"
        )

        save_app_list(list(all_apps_by_id.values()))

        start += SEARCH_COUNT
        page += 1

        if total_count is not None:
            try:
                if start >= int(total_count):
                    print("한국어 지원 게임 목록 전체 수집 완료")
                    break
            except:
                pass

        time.sleep(SEARCH_SLEEP_SEC)

    apps = list(all_apps_by_id.values())
    save_app_list(apps)

    print()
    print("한국어 지원 게임 목록 저장 완료:", APP_LIST_PATH)
    print("수집된 게임 수:", len(apps))

    return pd.DataFrame(apps)


# ============================================================
# 4. 태그 수집
# ============================================================

def fetch_store_tags(appid):
    """
    Steam 상점 페이지 HTML에서 사용자 태그를 수집합니다.
    """
    url = f"https://store.steampowered.com/app/{appid}/"

    params = {
        "l": "koreana",
        "cc": "KR",
    }

    try:
        response = request_get_with_retry(url, params=params)
        html_text = response.text

    except Exception as e:
        print(f"태그 수집 실패 AppID {appid}:", e)
        return ""

    tag_pattern = re.compile(
        r'<a[^>]*class="[^"]*app_tag[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE
    )

    tags = []

    for match in tag_pattern.finditer(html_text):
        tag = remove_html_tags(match.group(1))
        tag = tag.replace("+", " ")
        tag = re.sub(r"\s+", " ", tag).strip()

        if tag and tag not in tags:
            tags.append(tag)

    return ", ".join(tags)


# ============================================================
# 5. 한국어 리뷰 요약 정보 수집
# ============================================================

def translate_review_score_desc(desc):
    desc = clean_text(desc)

    if desc in REVIEW_SCORE_DESC_KO:
        return REVIEW_SCORE_DESC_KO[desc]

    return desc


def fetch_koreana_review_summary(appid):
    """
    전체 언어가 아니라 한국어 리뷰 기준 query_summary만 수집합니다.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"

    params = {
        "json": 1,
        "filter": "summary",
        "language": REVIEW_LANGUAGE,
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 1,
        "cursor": "*",
    }

    try:
        data = request_json_with_retry(url, params=params)

    except Exception as e:
        print(f"한국어 리뷰 요약 수집 실패 AppID {appid}:", e)
        return {}

    summary = data.get("query_summary", {})

    review_score_desc = clean_text(summary.get("review_score_desc", ""))

    return {
        "review_score": summary.get("review_score", ""),
        "review_score_desc": review_score_desc,
        "review_score_desc_ko": translate_review_score_desc(review_score_desc),
        "total_positive": summary.get("total_positive", ""),
        "total_negative": summary.get("total_negative", ""),
        "total_reviews": summary.get("total_reviews", ""),
    }


# ============================================================
# 6. 게임 상세 정보 수집
# ============================================================

def fetch_appdetails(appid):
    url = "https://store.steampowered.com/api/appdetails"

    params = {
        "appids": appid,
        "cc": "KR",
        "l": "koreana",
    }

    try:
        data = request_json_with_retry(url, params=params)

    except Exception as e:
        print(f"appdetails 수집 실패 AppID {appid}:", e)
        return {}

    app_data = data.get(str(appid), {})

    if not app_data.get("success"):
        return {}

    detail = app_data.get("data", {})

    price = detail.get("price_overview", {}) or {}
    platforms = detail.get("platforms", {}) or {}
    recommendations = detail.get("recommendations", {}) or {}
    achievements = detail.get("achievements", {}) or {}
    release_date_info = detail.get("release_date", {}) or {}

    release_date = clean_text(release_date_info.get("date", ""))

    result = {
        "name_from_appdetails": clean_text(detail.get("name", "")),
        "type": clean_text(detail.get("type", "")),
        "required_age": detail.get("required_age", ""),
        "is_free": detail.get("is_free", ""),
        "release_date": release_date,
        "release_year": extract_year_from_release_date(release_date),
        "genres": list_to_text(detail.get("genres", [])),
        "categories": list_to_text(detail.get("categories", [])),
        "developers": list_to_text(detail.get("developers", []), key=None),
        "publishers": list_to_text(detail.get("publishers", []), key=None),
        "supported_languages": remove_html_tags(detail.get("supported_languages", "")),
        "short_description": clean_text(detail.get("short_description", "")),
        "header_image": clean_text(detail.get("header_image", "")),
        "website": clean_text(detail.get("website", "")),
        "price_currency": price.get("currency", ""),
        "price_initial": price.get("initial", ""),
        "price_final": price.get("final", ""),
        "discount_percent": price.get("discount_percent", ""),
        "platform_windows": platforms.get("windows", ""),
        "platform_mac": platforms.get("mac", ""),
        "platform_linux": platforms.get("linux", ""),
        "recommendations_total": recommendations.get("total", ""),
        "achievements_total": achievements.get("total", ""),
    }

    return result


# ============================================================
# 7. 기존 저장 정보 불러오기
# ============================================================

def load_existing_review_info():
    seen_review_ids = set()
    app_review_counts = {}

    if not os.path.exists(RAW_REVIEW_PATH):
        return seen_review_ids, app_review_counts

    try:
        df = pd.read_csv(
            RAW_REVIEW_PATH,
            usecols=["appid", "review_id"],
            dtype={"review_id": str}
        )

        df.dropna(subset=["review_id"], inplace=True)
        df["appid"] = df["appid"].astype(int)
        df["review_id"] = df["review_id"].astype(str)

        seen_review_ids = set(df["review_id"].tolist())
        app_review_counts = df.groupby("appid").size().to_dict()

        print()
        print("=" * 80)
        print("기존 v2 리뷰 CSV 발견")
        print("=" * 80)
        print("기존 전체 리뷰 수:", len(seen_review_ids))
        print("기존 수집 게임 수:", len(app_review_counts))

    except Exception as e:
        print("기존 리뷰 CSV 읽기 실패. 새로 시작합니다:", e)

    return seen_review_ids, app_review_counts


def load_finished_appids_from_progress():
    finished_appids = set()

    if not os.path.exists(PROGRESS_LOG_PATH):
        return finished_appids

    try:
        df = pd.read_csv(PROGRESS_LOG_PATH)

        finished_statuses = [
            "finished_limited_reviews",
            "no_korean_reviews",
            "no_more_reviews",
            "cursor_end",
            "api_fail",
            "request_error",
        ]

        df_finished = df[df["status"].isin(finished_statuses)]

        for appid in df_finished["appid"]:
            finished_appids.add(int(appid))

        print()
        print("기존 v2 진행 기록 발견")
        print("이미 완료 처리된 게임 수:", len(finished_appids))

    except Exception as e:
        print("진행 기록 읽기 실패. 진행 기록 없이 시작합니다:", e)

    return finished_appids


def load_existing_game_detail_appids():
    done_appids = set()

    if not os.path.exists(GAME_DETAIL_PATH):
        return done_appids

    try:
        df = pd.read_csv(GAME_DETAIL_PATH, usecols=["appid"])
        for appid in df["appid"]:
            done_appids.add(int(appid))

        print("기존 게임 상세 정보 수집 완료 수:", len(done_appids))

    except Exception as e:
        print("게임 상세 정보 기존 파일 읽기 실패:", e)

    return done_appids


def append_csv_row(path, row):
    file_exists = os.path.exists(path)
    write_header = not file_exists

    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def append_progress_log(row):
    append_csv_row(PROGRESS_LOG_PATH, row)


def append_reviews_to_csv(rows):
    if len(rows) == 0:
        return

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["review_id"], inplace=True)

    file_exists = os.path.exists(RAW_REVIEW_PATH)
    write_header = not file_exists
    encoding = "utf-8-sig" if write_header else "utf-8"

    df.to_csv(
        RAW_REVIEW_PATH,
        mode="a",
        header=write_header,
        index=False,
        encoding=encoding
    )


# ============================================================
# 8. 게임 정보 row 만들기
# ============================================================

def make_game_detail_row(appid, game_title):
    app_detail = fetch_appdetails(appid)
    time.sleep(DETAIL_SLEEP_SEC)

    tags = fetch_store_tags(appid)
    time.sleep(DETAIL_SLEEP_SEC)

    review_summary = fetch_koreana_review_summary(appid)
    time.sleep(DETAIL_SLEEP_SEC)

    row = {
        "appid": int(appid),
        "game_title": game_title,

        "release_date": app_detail.get("release_date", ""),
        "release_year": app_detail.get("release_year", ""),

        "genres": app_detail.get("genres", ""),
        "tags": tags,
        "categories": app_detail.get("categories", ""),

        "review_score": review_summary.get("review_score", ""),
        "review_score_desc": review_summary.get("review_score_desc", ""),
        "review_score_desc_ko": review_summary.get("review_score_desc_ko", ""),
        "total_positive": review_summary.get("total_positive", ""),
        "total_negative": review_summary.get("total_negative", ""),
        "total_reviews": review_summary.get("total_reviews", ""),

        "name_from_appdetails": app_detail.get("name_from_appdetails", ""),
        "type": app_detail.get("type", ""),
        "required_age": app_detail.get("required_age", ""),
        "is_free": app_detail.get("is_free", ""),
        "developers": app_detail.get("developers", ""),
        "publishers": app_detail.get("publishers", ""),
        "supported_languages": app_detail.get("supported_languages", ""),
        "short_description": app_detail.get("short_description", ""),
        "header_image": app_detail.get("header_image", ""),
        "website": app_detail.get("website", ""),
        "price_currency": app_detail.get("price_currency", ""),
        "price_initial": app_detail.get("price_initial", ""),
        "price_final": app_detail.get("price_final", ""),
        "discount_percent": app_detail.get("discount_percent", ""),
        "platform_windows": app_detail.get("platform_windows", ""),
        "platform_mac": app_detail.get("platform_mac", ""),
        "platform_linux": app_detail.get("platform_linux", ""),
        "recommendations_total": app_detail.get("recommendations_total", ""),
        "achievements_total": app_detail.get("achievements_total", ""),
    }

    return row


# ============================================================
# 9. 리뷰 row 만들기
# ============================================================

def make_review_row(appid, game_title, item):
    author = item.get("author", {}) or {}

    review_text = clean_text(item.get("review", ""))
    review_id = str(item.get("recommendationid", "")).strip()

    row = {
        "appid": int(appid),
        "game_title": game_title,
        "review_id": review_id,
        "review": review_text,
        "voted_up": item.get("voted_up"),
        "language": item.get("language"),
        "timestamp_created": item.get("timestamp_created"),
        "timestamp_created_datetime": unix_to_datetime(item.get("timestamp_created")),
        "timestamp_updated": item.get("timestamp_updated"),
        "timestamp_updated_datetime": unix_to_datetime(item.get("timestamp_updated")),

        "playtime_forever": author.get("playtime_forever"),
        "playtime_last_two_weeks": author.get("playtime_last_two_weeks"),
        "playtime_at_review": author.get("playtime_at_review"),
        "deck_playtime_at_review": author.get("deck_playtime_at_review"),
        "last_played": author.get("last_played"),
        "last_played_datetime": unix_to_datetime(author.get("last_played")),

        "votes_up": item.get("votes_up"),
        "votes_funny": item.get("votes_funny"),
        "weighted_vote_score": item.get("weighted_vote_score"),
        "comment_count": item.get("comment_count"),

        "steam_purchase": item.get("steam_purchase"),
        "received_for_free": item.get("received_for_free"),
        "written_during_early_access": item.get("written_during_early_access"),
        "primarily_steam_deck": item.get("primarily_steam_deck"),

        "developer_response": clean_text(item.get("developer_response", "")),
        "timestamp_dev_responded": item.get("timestamp_dev_responded"),
        "timestamp_dev_responded_datetime": unix_to_datetime(item.get("timestamp_dev_responded")),
    }

    return row


# ============================================================
# 10. 게임별 한국어 리뷰 수집
# ============================================================

def crawl_reviews_with_filter(appid, game_title, seen_review_ids, existing_count, review_filter, day_range=None):
    appid = int(appid)
    cursor = "*"
    new_saved_count = 0
    page = 0

    while True:
        if existing_count + new_saved_count >= MAX_REVIEWS_PER_GAME:
            return "finished_limited_reviews", new_saved_count

        if page >= MAX_REVIEW_PAGES_PER_GAME:
            if existing_count + new_saved_count == 0:
                return "no_korean_reviews", new_saved_count
            return "finished_limited_reviews", new_saved_count

        page += 1

        url = f"https://store.steampowered.com/appreviews/{appid}"

        params = {
            "json": 1,
            "filter": review_filter,
            "language": REVIEW_LANGUAGE,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": REVIEWS_PER_PAGE,
            "cursor": cursor,
        }

        if day_range is not None:
            params["day_range"] = day_range

        try:
            data = request_json_with_retry(url, params=params)

        except Exception as e:
            print(f"[오류] {game_title} 리뷰 요청 실패:", e)
            return "request_error", new_saved_count

        if data.get("success") != 1:
            print(f"[중단] {game_title} API success가 1이 아닙니다.")
            return "api_fail", new_saved_count

        reviews = data.get("reviews", [])

        if len(reviews) == 0:
            if existing_count + new_saved_count == 0:
                return "no_korean_reviews", new_saved_count
            return "no_more_reviews", new_saved_count

        page_rows = []

        for item in reviews:
            review_id = str(item.get("recommendationid", "")).strip()
            review_text = clean_text(item.get("review", ""))

            if not review_id:
                continue

            if review_id in seen_review_ids:
                continue

            if len(review_text) < MIN_REVIEW_CHARS:
                continue

            row = make_review_row(appid, game_title, item)
            page_rows.append(row)
            seen_review_ids.add(review_id)

            if existing_count + new_saved_count + len(page_rows) >= MAX_REVIEWS_PER_GAME:
                break

        if len(page_rows) > 0:
            append_reviews_to_csv(page_rows)
            new_saved_count += len(page_rows)

        print(
            f"{game_title} / filter={review_filter} / page {page} / "
            f"API 한국어 리뷰 {len(reviews)}개 / "
            f"저장 {len(page_rows)}개 / "
            f"이 게임 누적 {existing_count + new_saved_count}개"
        )

        next_cursor = data.get("cursor", "")

        if not next_cursor or next_cursor == cursor:
            return "cursor_end", new_saved_count

        cursor = next_cursor
        time.sleep(REVIEW_SLEEP_SEC)


def crawl_reviews_for_one_game(appid, game_title, seen_review_ids, existing_count):
    # 1차: helpfulness 기준 상위 노출 리뷰 우선
    status, new_count = crawl_reviews_with_filter(
        appid=appid,
        game_title=game_title,
        seen_review_ids=seen_review_ids,
        existing_count=existing_count,
        review_filter=REVIEW_FILTER,
        day_range=DAY_RANGE,
    )

    # 2차: 최근 365일 helpfulness 기준에서 하나도 못 얻은 경우,
    # 한국어 리뷰가 오래된 게임을 놓치지 않기 위해 recent로 한 번 더 확인
    if (
        FALLBACK_TO_RECENT_IF_EMPTY
        and new_count == 0
        and status == "no_korean_reviews"
    ):
        print("상위 노출 기준에서 한국어 리뷰가 없어 recent 기준으로 재확인합니다.")

        status, new_count = crawl_reviews_with_filter(
            appid=appid,
            game_title=game_title,
            seen_review_ids=seen_review_ids,
            existing_count=existing_count,
            review_filter="recent",
            day_range=None,
        )

    return status, new_count


# ============================================================
# 11. 메인 실행
# ============================================================

if __name__ == "__main__":
    start_time = now_str()

    print()
    print("=" * 80)
    print("Steam 한국어 리뷰 + 게임 정보 v2 수집 시작")
    print("=" * 80)
    print("시작 시간:", start_time)
    print("탐색 조건: 한국어 지원 게임만")
    print("리뷰 조건: 한국어 리뷰만")
    print("게임당 최대 리뷰 수:", MAX_REVIEWS_PER_GAME)
    print("게임당 최대 리뷰 페이지:", MAX_REVIEW_PAGES_PER_GAME)
    print("게임 상세 정보 저장:", GAME_DETAIL_PATH)
    print("리뷰 저장:", RAW_REVIEW_PATH)

    # --------------------------------------------------------
    # 1) 한국어 지원 게임 목록 준비
    # --------------------------------------------------------
    if not REFRESH_APP_LIST:
        df_apps = load_app_list()
    else:
        df_apps = None

    if df_apps is None or len(df_apps) == 0:
        df_apps = crawl_koreana_supported_app_list()

    if df_apps is None or len(df_apps) == 0:
        print("한국어 지원 게임 목록이 비어 있습니다. 종료합니다.")
        exit()

    df_apps.drop_duplicates(subset=["appid"], inplace=True)
    df_apps["appid"] = df_apps["appid"].astype(int)
    df_apps.reset_index(drop=True, inplace=True)

    print()
    print("=" * 80)
    print("수집 대상 게임 수:", len(df_apps))
    print("=" * 80)
    print(df_apps.head())

    # --------------------------------------------------------
    # 2) 기존 저장 정보 불러오기
    # --------------------------------------------------------
    seen_review_ids, app_review_counts = load_existing_review_info()
    finished_appids = load_finished_appids_from_progress()
    detail_done_appids = load_existing_game_detail_appids()

    success_game_count = 0
    skipped_game_count = 0
    error_game_count = 0
    detail_saved_count = 0

    total_apps = len(df_apps)

    # --------------------------------------------------------
    # 3) 게임 순회
    # --------------------------------------------------------
    for idx, row in df_apps.iterrows():
        appid = int(row["appid"])
        game_title = clean_text(row["game_title"])
        existing_count = int(app_review_counts.get(appid, 0))

        print()
        print("=" * 80)
        print(f"[{idx + 1}/{total_apps}] {game_title}")
        print("AppID:", appid)
        print("기존 저장 리뷰 수:", existing_count)
        print("=" * 80)

        # ----------------------------------------------------
        # 3-1) 게임 상세 정보 수집
        # ----------------------------------------------------
        if appid not in detail_done_appids:
            try:
                detail_row = make_game_detail_row(appid, game_title)
                append_csv_row(GAME_DETAIL_PATH, detail_row)
                detail_done_appids.add(appid)
                detail_saved_count += 1

                print("게임 상세 정보 저장 완료")
                print("장르:", detail_row.get("genres", ""))
                print("태그:", detail_row.get("tags", ""))
                print("한국어 리뷰 평가:", detail_row.get("review_score_desc_ko", ""))

            except Exception as e:
                print("게임 상세 정보 저장 실패:", e)

        else:
            print("게임 상세 정보는 이미 저장되어 있습니다.")

        # ----------------------------------------------------
        # 3-2) 리뷰 수집
        # ----------------------------------------------------
        if appid in finished_appids:
            print("이미 리뷰 수집 완료 처리된 게임입니다. 건너뜁니다.")
            skipped_game_count += 1
            continue

        if existing_count >= MAX_REVIEWS_PER_GAME:
            print("이미 목표 리뷰 수에 도달했습니다. 건너뜁니다.")

            append_progress_log({
                "time": now_str(),
                "appid": appid,
                "game_title": game_title,
                "status": "finished_limited_reviews",
                "new_reviews": 0,
                "total_reviews_for_game": existing_count,
            })

            finished_appids.add(appid)
            skipped_game_count += 1
            continue

        try:
            status, new_count = crawl_reviews_for_one_game(
                appid=appid,
                game_title=game_title,
                seen_review_ids=seen_review_ids,
                existing_count=existing_count,
            )

            total_count_for_game = existing_count + new_count
            app_review_counts[appid] = total_count_for_game

            append_progress_log({
                "time": now_str(),
                "appid": appid,
                "game_title": game_title,
                "status": status,
                "new_reviews": new_count,
                "total_reviews_for_game": total_count_for_game,
            })

            print()
            print("게임 처리 완료:", game_title)
            print("상태:", status)
            print("신규 저장 리뷰 수:", new_count)
            print("이 게임 총 저장 리뷰 수:", total_count_for_game)

            if status in [
                "finished_limited_reviews",
                "no_korean_reviews",
                "no_more_reviews",
                "cursor_end",
                "api_fail",
                "request_error",
            ]:
                finished_appids.add(appid)

            if new_count > 0:
                success_game_count += 1
            else:
                skipped_game_count += 1

        except Exception as e:
            print("[예상 밖 오류]", game_title, e)
            error_game_count += 1

            append_progress_log({
                "time": now_str(),
                "appid": appid,
                "game_title": game_title,
                "status": "unexpected_error",
                "new_reviews": 0,
                "total_reviews_for_game": existing_count,
            })

        time.sleep(GAME_SLEEP_SEC)

    # --------------------------------------------------------
    # 4) 최종 요약
    # --------------------------------------------------------
    end_time = now_str()

    print()
    print("=" * 80)
    print("Steam 한국어 리뷰 + 게임 정보 v2 수집 종료")
    print("=" * 80)
    print("시작 시간:", start_time)
    print("종료 시간:", end_time)
    print("게임 상세 정보 신규 저장 수:", detail_saved_count)
    print("신규 리뷰 저장 성공 게임 수:", success_game_count)
    print("건너뛴 게임 수:", skipped_game_count)
    print("오류 게임 수:", error_game_count)
    print("앱 목록 파일:", APP_LIST_PATH)
    print("게임 상세 정보 파일:", GAME_DETAIL_PATH)
    print("리뷰 저장 파일:", RAW_REVIEW_PATH)
    print("진행 로그 파일:", PROGRESS_LOG_PATH)

    if os.path.exists(GAME_DETAIL_PATH):
        try:
            df_detail = pd.read_csv(GAME_DETAIL_PATH)
            print()
            print("현재 게임 상세 정보 수:", len(df_detail))
            print(df_detail[["appid", "game_title", "release_year", "genres", "tags", "review_score_desc_ko", "total_reviews"]].head())
        except Exception as e:
            print("게임 상세 정보 요약 출력 실패:", e)

    if os.path.exists(RAW_REVIEW_PATH):
        try:
            df_result = pd.read_csv(RAW_REVIEW_PATH)
            print()
            print("현재 전체 리뷰 수:", len(df_result))
            print("현재 수집된 게임 수:", df_result["appid"].nunique())
            print()
            print(df_result["game_title"].value_counts().head(30))
        except Exception as e:
            print("최종 리뷰 CSV 요약 출력 실패:", e)