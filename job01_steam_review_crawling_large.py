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

os.makedirs("../Steam_Game_Recommendation/datasets", exist_ok=True)
os.makedirs("../Steam_Game_Recommendation/models", exist_ok=True)

# ------------------------------------------------------------
# Steam 상점 검색 필터
# ------------------------------------------------------------
# supportedlang=koreana : 한국어 지원 게임만 탐색
# category1=998         : 게임 카테고리만 탐색
STORE_LANGUAGE = "koreana"
REVIEW_LANGUAGE = "koreana"
CATEGORY_GAME = "998"

# Steam 검색 결과는 한 번에 100개씩 가져옵니다.
SEARCH_COUNT = 100

# None이면 한국어 지원 게임 목록을 끝까지 수집합니다.
MAX_SEARCH_PAGES = None

# ------------------------------------------------------------
# 리뷰 수집 설정
# ------------------------------------------------------------
# 게임당 한국어 리뷰 최대 수
MAX_REVIEWS_PER_GAME = 100

# 게임당 리뷰 페이지 확인 수
# 1페이지당 최대 100개이므로, 3페이지면 최대 300개까지 확인합니다.
# 그중 중복/짧은 리뷰를 제외하고 최대 100개만 저장합니다.
MAX_REVIEW_PAGES_PER_GAME = 3

# Steam 리뷰 API 요청당 리뷰 수
REVIEWS_PER_PAGE = 100

# 너무 짧은 리뷰 제외
MIN_REVIEW_CHARS = 5

# 리뷰 정렬 방식
# recent: 최근 리뷰 기준
# 나중에 "도움됨/좋아요 많은 리뷰 위주"가 필요하면 이 값을 조정 검토
REVIEW_FILTER = "recent"

# 요청 간격
SEARCH_SLEEP_SEC = 0.5
REVIEW_SLEEP_SEC = 0.8
GAME_SLEEP_SEC = 0.5

# 요청 실패 시 재시도
MAX_RETRY = 3
RETRY_SLEEP_SEC = 5
REQUEST_TIMEOUT = 20

# ------------------------------------------------------------
# 저장 파일 경로
# ------------------------------------------------------------
APP_LIST_PATH = "../Steam_Game_Recommendation/datasets/steam_koreana_supported_games.csv"
RAW_REVIEW_PATH = "../Steam_Game_Recommendation/datasets/steam_reviews_raw_large.csv"
PROGRESS_LOG_PATH = "../Steam_Game_Recommendation/datasets/steam_review_crawling_progress.csv"

# True면 기존 AppID 목록 CSV가 있어도 Steam 상점에서 다시 목록을 수집합니다.
REFRESH_APP_LIST = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 Steam review collector for personal Python ML project",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


# ============================================================
# 1. 공통 함수
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
# 2. Steam 한국어 지원 게임 목록 수집
# ============================================================

def parse_search_results_html(results_html):
    """
    Steam 검색 결과 HTML에서 appid와 게임 제목을 추출합니다.
    beautifulsoup4 없이 정규식으로 처리합니다.
    """
    apps = []

    # 검색 결과의 한 행
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
                "source": "steam_search_supportedlang_koreana"
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
    if not os.path.exists(APP_LIST_PATH):
        return None

    df = pd.read_csv(APP_LIST_PATH)
    df.drop_duplicates(subset=["appid"], inplace=True)
    df["appid"] = df["appid"].astype(int)
    return df


def crawl_koreana_supported_app_list():
    """
    Steam 상점 검색에서 한국어 지원 필터를 걸고 게임 목록을 수집합니다.
    이 단계는 리뷰 수집이 아니라 AppID 목록 수집입니다.
    """
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

        # 중간 저장
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
# 3. 기존 리뷰 / 진행 기록 불러오기
# ============================================================

def load_existing_review_info():
    """
    기존 raw review CSV가 있으면 review_id 중복 방지 set과
    appid별 리뷰 수를 만듭니다.
    """
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
        print("기존 리뷰 CSV 발견")
        print("=" * 80)
        print("기존 전체 리뷰 수:", len(seen_review_ids))
        print("기존 수집 게임 수:", len(app_review_counts))

    except Exception as e:
        print("기존 리뷰 CSV 읽기 실패. 새로 시작합니다:", e)

    return seen_review_ids, app_review_counts


def load_finished_appids_from_progress():
    """
    이미 완료 처리된 appid를 불러옵니다.
    다시 실행할 때 no_korean_reviews, finished 등의 게임은 건너뛰기 위함입니다.
    """
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
        ]

        df_finished = df[df["status"].isin(finished_statuses)]

        for appid in df_finished["appid"]:
            finished_appids.add(int(appid))

        print()
        print("기존 진행 기록 발견")
        print("이미 완료 처리된 게임 수:", len(finished_appids))

    except Exception as e:
        print("진행 기록 읽기 실패. 진행 기록 없이 시작합니다:", e)

    return finished_appids


def append_progress_log(row):
    file_exists = os.path.exists(PROGRESS_LOG_PATH)
    write_header = not file_exists

    with open(PROGRESS_LOG_PATH, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


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
# 4. 게임별 한국어 리뷰 수집
# ============================================================

def make_review_row(appid, game_title, item):
    author = item.get("author", {})

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
        "playtime_at_review": author.get("playtime_at_review"),
        "votes_up": item.get("votes_up"),
        "votes_funny": item.get("votes_funny"),
        "weighted_vote_score": item.get("weighted_vote_score"),
        "comment_count": item.get("comment_count"),
        "steam_purchase": item.get("steam_purchase"),
        "received_for_free": item.get("received_for_free"),
        "written_during_early_access": item.get("written_during_early_access"),
    }

    return row


def crawl_reviews_for_one_game(appid, game_title, seen_review_ids, existing_count):
    """
    한 게임에 대해 한국어 리뷰만 요청합니다.
    모든 언어 리뷰를 받은 뒤 한국어만 고르는 방식이 아닙니다.
    요청 단계에서 language=koreana를 넣습니다.
    """
    appid = int(appid)
    cursor = "*"

    new_saved_count = 0
    page = 0

    while True:
        if new_saved_count + existing_count >= MAX_REVIEWS_PER_GAME:
            return "finished_limited_reviews", new_saved_count

        if page >= MAX_REVIEW_PAGES_PER_GAME:
            if new_saved_count == 0 and existing_count == 0:
                return "no_korean_reviews", new_saved_count
            else:
                return "finished_limited_reviews", new_saved_count

        page += 1

        url = f"https://store.steampowered.com/appreviews/{appid}"

        params = {
            "json": 1,
            "filter": REVIEW_FILTER,
            "language": REVIEW_LANGUAGE,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": REVIEWS_PER_PAGE,
            "cursor": cursor,
        }

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
            if new_saved_count == 0 and existing_count == 0:
                return "no_korean_reviews", new_saved_count
            else:
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
            f"{game_title} / page {page} / "
            f"API 한국어 리뷰 {len(reviews)}개 / "
            f"저장 {len(page_rows)}개 / "
            f"이 게임 누적 {existing_count + new_saved_count}개"
        )

        next_cursor = data.get("cursor", "")

        if not next_cursor or next_cursor == cursor:
            return "cursor_end", new_saved_count

        cursor = next_cursor

        time.sleep(REVIEW_SLEEP_SEC)


# ============================================================
# 5. 메인 실행
# ============================================================

if __name__ == "__main__":
    start_time = now_str()

    print()
    print("=" * 80)
    print("Steam 한국어 리뷰 대량 수집 시작")
    print("=" * 80)
    print("시작 시간:", start_time)
    print("탐색 조건: 한국어 지원 게임만")
    print("리뷰 조건: 한국어 리뷰만")
    print("게임당 최대 리뷰 수:", MAX_REVIEWS_PER_GAME)
    print("게임당 최대 리뷰 페이지:", MAX_REVIEW_PAGES_PER_GAME)

    # --------------------------------------------------------
    # 1) 한국어 지원 게임 목록 준비
    # --------------------------------------------------------
    if os.path.exists(APP_LIST_PATH) and not REFRESH_APP_LIST:
        print()
        print("기존 한국어 지원 게임 목록 CSV를 사용합니다.")
        df_apps = load_app_list()
    else:
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
    # 2) 기존 리뷰와 진행 기록 불러오기
    # --------------------------------------------------------
    seen_review_ids, app_review_counts = load_existing_review_info()
    finished_appids = load_finished_appids_from_progress()

    success_game_count = 0
    skipped_game_count = 0
    error_game_count = 0

    # --------------------------------------------------------
    # 3) 한국어 지원 게임을 순서대로 검사
    # --------------------------------------------------------
    total_apps = len(df_apps)

    for idx, row in df_apps.iterrows():
        appid = int(row["appid"])
        game_title = str(row["game_title"])
        existing_count = int(app_review_counts.get(appid, 0))

        print()
        print("=" * 80)
        print(f"[{idx + 1}/{total_apps}] {game_title}")
        print("AppID:", appid)
        print("기존 저장 리뷰 수:", existing_count)
        print("=" * 80)

        # 이미 처리 완료된 게임은 건너뜀
        if appid in finished_appids:
            print("이미 완료 처리된 게임입니다. 건너뜁니다.")
            skipped_game_count += 1
            continue

        # 이미 충분히 모은 게임은 건너뜀
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
                existing_count=existing_count
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
    print("Steam 한국어 리뷰 대량 수집 종료")
    print("=" * 80)
    print("시작 시간:", start_time)
    print("종료 시간:", end_time)
    print("신규 리뷰 저장 성공 게임 수:", success_game_count)
    print("건너뛴 게임 수:", skipped_game_count)
    print("오류 게임 수:", error_game_count)
    print("앱 목록 파일:", APP_LIST_PATH)
    print("리뷰 저장 파일:", RAW_REVIEW_PATH)
    print("진행 로그 파일:", PROGRESS_LOG_PATH)

    if os.path.exists(RAW_REVIEW_PATH):
        try:
            df_result = pd.read_csv(RAW_REVIEW_PATH)
            print()
            print("현재 전체 리뷰 수:", len(df_result))
            print("현재 수집된 게임 수:", df_result["appid"].nunique())
            print()
            print(df_result["game_title"].value_counts().head(30))
        except Exception as e:
            print("최종 CSV 요약 출력 실패:", e)