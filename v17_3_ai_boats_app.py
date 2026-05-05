# -*- coding: utf-8 -*-
"""
v17.4 全艇スコア解析アプリ（LightGBM AI予測 ＋ 並列処理による爆速化版）
"""

import re
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# --- AI(LightGBM)用のライブラリ ---
import lightgbm as lgb
import numpy as np


# ============================================================
# kyoteibiyori.com 場別コース別データ
# 集計期間: 2023年03月12日 - 2024年03月12日
# ============================================================

# 1着率(%)
COURSE_WIN_RATE: Dict[str, List[float]] = {
    "全国":   [55.1, 14.0, 12.8, 11.1, 6.1, 1.8],
    "桐生":   [53.8, 13.2, 12.6, 12.5, 7.2, 1.4],
    "戸田":   [43.9, 15.9, 16.6, 14.5, 7.7, 2.5],
    "江戸川": [45.7, 18.4, 15.1, 12.3, 7.6, 2.6],
    "平和島": [45.1, 17.0, 14.4, 13.1, 7.7, 3.7],
    "多摩川": [52.9, 16.5, 12.5, 11.5, 5.9, 1.9],
    "浜名湖": [50.9, 15.9, 14.4, 11.5, 6.8, 1.6],
    "蒲郡":   [54.4, 11.8, 13.6, 13.7, 6.2, 1.4],
    "常滑":   [57.8, 12.8, 10.9, 10.8, 7.0, 1.6],
    "津":     [57.7, 15.6, 11.9,  9.5, 4.8, 1.4],
    "三国":   [55.2, 14.9, 13.5, 11.0, 5.3, 1.3],
    "びわこ": [56.8, 14.6, 11.8, 11.5, 4.6, 1.6],
    "住之江": [57.9, 14.6, 11.6,  9.8, 5.3, 1.6],
    "尼崎":   [57.6, 12.0, 12.0, 11.9, 5.6, 1.7],
    "鳴門":   [47.5, 14.9, 16.1, 12.2, 7.7, 2.3],
    "丸亀":   [56.2, 15.2, 11.7, 10.3, 5.0, 2.5],
    "児島":   [55.6, 12.9, 12.1, 12.3, 6.1, 2.0],
    "宮島":   [57.0, 13.1, 12.9,  9.6, 6.3, 2.0],
    "徳山":   [65.9, 12.8,  9.2,  6.6, 4.7, 1.1],
    "下関":   [59.6, 10.6, 10.9, 10.9, 6.2, 2.6],
    "若松":   [56.8, 11.8, 12.9, 11.2, 6.4, 2.0],
    "芦屋":   [59.1, 11.3, 11.3, 10.8, 6.1, 2.2],
    "福岡":   [56.0, 14.8, 15.2,  9.2, 4.8, 1.0],
    "唐津":   [55.3, 14.2, 13.5, 10.3, 6.6, 1.3],
    "大村":   [61.3, 12.1, 11.3,  9.6, 5.0, 1.3],
}

# 差し率(%)
COURSE_SASHI_RATE: Dict[str, List[float]] = {
    "全国":   [ 8.8, 1.5, 2.1, 0.3, 0.2],
    "桐生":   [ 8.1, 1.2, 1.7, 0.3, 0.0],
    "戸田":   [ 8.2, 1.9, 2.4, 0.3, 0.1],
    "江戸川": [11.3, 1.7, 2.6, 0.5, 0.0],
    "平和島": [11.6, 2.0, 4.0, 0.8, 1.2],
    "多摩川": [10.6, 1.8, 2.2, 0.3, 0.4],
    "浜名湖": [ 9.1, 1.8, 1.8, 0.4, 0.3],
    "蒲郡":   [ 6.1, 1.0, 1.5, 0.3, 0.1],
    "常滑":   [ 7.7, 0.7, 1.4, 0.1, 0.0],
    "津":     [11.1, 1.5, 2.3, 0.3, 0.1],
    "三国":   [10.2, 2.3, 2.6, 0.4, 0.1],
    "びわこ": [ 9.1, 1.9, 2.9, 0.2, 0.2],
    "住之江": [ 9.8, 1.6, 2.4, 0.5, 0.2],
    "尼崎":   [ 7.2, 1.5, 1.9, 0.3, 0.0],
    "鳴門":   [ 9.1, 1.9, 2.6, 0.7, 0.3],
    "丸亀":   [11.6, 1.1, 2.4, 0.3, 0.5],
    "児島":   [ 9.2, 1.8, 2.5, 0.1, 0.3],
    "宮島":   [ 7.8, 1.1, 1.7, 0.2, 0.1],
    "徳山":   [ 9.3, 1.2, 1.3, 0.3, 0.2],
    "下関":   [ 7.1, 1.3, 1.7, 0.4, 0.4],
    "若松":   [ 7.5, 1.1, 2.1, 0.4, 0.1],
    "芦屋":   [ 6.0, 0.8, 1.4, 0.2, 0.1],
    "福岡":   [ 8.4, 1.7, 1.6, 0.2, 0.0],
    "唐津":   [ 9.2, 1.5, 2.2, 0.2, 0.0],
    "大村":   [ 7.3, 1.4, 1.4, 0.0, 0.0],
}

# まくり率(%)
COURSE_MAKURI_RATE: Dict[str, List[float]] = {
    "全国":   [3.6, 5.1, 5.1, 1.3, 0.4],
    "桐生":   [3.7, 4.9, 7.1, 1.7, 0.4],
    "戸田":   [6.5, 8.5, 7.9, 2.0, 0.7],
    "江戸川": [4.4, 6.7, 5.9, 2.2, 0.9],
    "平和島": [3.1, 6.7, 5.1, 1.3, 0.7],
    "多摩川": [4.1, 4.8, 5.7, 1.3, 0.3],
    "浜名湖": [5.0, 4.0, 4.7, 0.9, 0.3],
    "蒲郡":   [4.7, 4.7, 7.1, 1.2, 0.6],
    "常滑":   [3.6, 4.1, 6.1, 1.9, 0.5],
    "津":     [3.3, 3.6, 3.6, 0.6, 0.1],
    "三国":   [2.9, 5.2, 4.3, 0.9, 0.3],
    "びわこ": [4.1, 4.6, 4.3, 0.7, 0.4],
    "住之江": [3.5, 5.1, 3.7, 1.0, 0.4],
    "尼崎":   [3.4, 4.4, 5.6, 0.7, 0.6],
    "鳴門":   [4.6, 7.2, 5.3, 1.3, 0.4],
    "丸亀":   [2.2, 3.9, 4.1, 0.7, 0.6],
    "児島":   [2.1, 4.2, 5.1, 0.9, 0.4],
    "宮島":   [4.2, 5.6, 4.4, 1.6, 0.7],
    "徳山":   [2.1, 3.2, 2.7, 0.7, 0.1],
    "下関":   [2.4, 4.4, 5.9, 2.1, 1.0],
    "若松":   [2.9, 6.1, 5.0, 1.6, 0.4],
    "芦屋":   [3.1, 4.2, 5.6, 1.4, 0.4],
    "福岡":   [4.7, 9.0, 4.0, 1.0, 0.4],
    "唐津":   [3.3, 4.3, 4.6, 1.2, 0.3],
    "大村":   [2.8, 3.1, 4.3, 0.9, 0.3],
}

# まくり差し率(%)
COURSE_MAKURI_SASHI_RATE: Dict[str, List[float]] = {
    "全国":   [4.6, 2.7, 3.6, 0.7],
    "桐生":   [5.3, 2.5, 4.5, 0.5],
    "戸田":   [4.7, 3.1, 4.4, 1.1],
    "江戸川": [3.7, 2.0, 3.4, 0.7],
    "平和島": [3.6, 2.9, 4.7, 1.3],
    "多摩川": [4.6, 2.6, 3.5, 0.4],
    "浜名湖": [6.6, 3.7, 4.5, 0.7],
    "蒲郡":   [6.4, 3.8, 3.9, 0.4],
    "常滑":   [4.7, 2.3, 4.1, 0.8],
    "津":     [5.0, 2.4, 3.3, 0.6],
    "三国":   [4.9, 2.9, 3.0, 0.5],
    "びわこ": [3.7, 2.8, 2.7, 0.8],
    "住之江": [3.8, 2.5, 3.0, 0.6],
    "尼崎":   [4.8, 3.1, 3.8, 0.7],
    "鳴門":   [5.4, 2.9, 4.3, 1.1],
    "丸亀":   [5.4, 2.9, 3.6, 1.0],
    "児島":   [4.5, 3.3, 4.1, 0.8],
    "宮島":   [4.7, 2.2, 3.7, 0.9],
    "徳山":   [3.4, 1.5, 2.9, 0.6],
    "下関":   [4.0, 2.4, 2.8, 0.9],
    "若松":   [3.6, 3.0, 3.2, 0.8],
    "芦屋":   [4.8, 2.9, 3.7, 1.1],
    "福岡":   [2.5, 2.3, 2.7, 0.4],
    "唐津":   [5.8, 2.4, 4.4, 0.7],
    "大村":   [5.4, 2.8, 3.2, 0.7],
}

COURSE_BASE_POINTS: Dict[int, float] = {
    1: 2.0, 2: 0.5, 3: 0.0, 4: -0.5, 5: -1.0, 6: -1.5,
}

VENUE_WIN_RATE_COEF = 0.10
VENUE_ATTACK_COEF = 0.08

def venue_course_bonus(venue: str, lane: int) -> float:
    if venue not in COURSE_WIN_RATE or not 1 <= lane <= 6:
        return 0.0
    nat = COURSE_WIN_RATE["全国"][lane - 1]
    ven = COURSE_WIN_RATE[venue][lane - 1]
    return round((ven - nat) * VENUE_WIN_RATE_COEF, 2)

def venue_attack_bonus(venue: str, lane: int) -> float:
    if lane < 3 or venue not in COURSE_MAKURI_RATE:
        return 0.0
    idx = lane - 2
    nat = COURSE_MAKURI_RATE["全国"][idx]
    ven = COURSE_MAKURI_RATE[venue][idx]
    ms_idx = lane - 3
    nat += COURSE_MAKURI_SASHI_RATE["全国"][ms_idx]
    ven += COURSE_MAKURI_SASHI_RATE[venue][ms_idx]
    return round((ven - nat) * VENUE_ATTACK_COEF, 2)

def venue_tendency_label(venue: str) -> str:
    if venue not in COURSE_WIN_RATE:
        return ""
    v = COURSE_WIN_RATE[venue]
    nat = COURSE_WIN_RATE["全国"]
    diff_1c = v[0] - nat[0]
    if diff_1c >= 5: return f"🟢 イン強烈({v[0]:.1f}%)"
    elif diff_1c >= 2: return f"🟢 インやや有利({v[0]:.1f}%)"
    elif diff_1c <= -5: return f"🔴 イン不利({v[0]:.1f}%)・荒れ水面"
    elif diff_1c <= -2: return f"🟡 インやや不利({v[0]:.1f}%)"
    else: return f"⚪ 標準({v[0]:.1f}%)"

# ============================================================
# データ構造・スコアリング
# ============================================================
@dataclass
class Racer:
    name: str = ""
    cls: str = ""
    win_rate: Optional[float] = None
    avg_st: Optional[float] = None
    settle_st: Optional[float] = None
    settle_avg_rank: Optional[float] = None
    motor_2rate: Optional[float] = None
    f_count: int = 0
    exhibit_rank: Optional[int] = None
    course5_avg_st: Optional[float] = None
    weight: Optional[float] = None
    makuri_rate: Optional[float] = None

def _band(v: Optional[float], bands: List[Tuple[float, float, float]], default: float = 0.0) -> float:
    if v is None: return default
    for lo, hi, pts in bands:
        if lo <= v < hi: return pts
    return default

def score_boat(r: Racer, venue: str, lane: int) -> Dict[str, float]:
    parts: Dict[str, float] = {}

    parts["コース基礎"] = COURSE_BASE_POINTS.get(lane, 0.0)
    parts["級別"] = {"A1": 2.5, "A2": 1.5, "B1": 0.0, "B2": -1.5}.get(r.cls, 0.0)
    parts["勝率"] = _band(r.win_rate, [(6.50, 99, 1.5), (5.50, 6.50, 1.0), (5.00, 5.50, 0.5), (4.00, 5.00, -0.5), (0.00, 4.00, -1.2)])

    st_val = r.course5_avg_st if r.course5_avg_st is not None else r.avg_st
    att    = 1.0 if r.course5_avg_st is not None else 0.5
    parts["ST"] = att * _band(st_val, [(0.00, 0.14, 2.0), (0.14, 0.16, 1.3), (0.16, 0.18, 0.5), (0.18, 0.20, -0.3), (0.20, 9.99, -1.3)])

    parts["節平順"] = _band(r.settle_avg_rank, [(0.99, 1.50, 2.0), (1.50, 2.50, 1.2), (2.50, 3.50, 0.3), (3.50, 4.50, -0.5), (4.50, 6.01, -1.5)])

    if r.settle_st is not None and r.avg_st is not None:
        delta = r.avg_st - r.settle_st
        if delta >= 0.04: parts["節ST改善"] = 1.5
        elif delta >= 0.02: parts["節ST改善"] = 1.0
        elif delta >= 0.00: parts["節ST改善"] = 0.3
        elif delta >= -0.02: parts["節ST改善"] = -0.3
        else: parts["節ST改善"] = -1.0
    else:
        parts["節ST改善"] = 0.0

    parts["モーター"] = _band(r.motor_2rate, [(0.45, 1.01, 1.5), (0.35, 0.45, 0.8), (0.30, 0.35, 0.3), (0.25, 0.30, -0.3), (0.00, 0.25, -1.2)])

    exhibit_scores = {1: 1.5, 2: 0.8, 3: 0.3, 4: -0.2, 5: -0.6, 6: -1.0}
    parts["展示"] = exhibit_scores.get(r.exhibit_rank, 0.0)

    if r.weight is not None:
        if r.weight <= 52.0: parts["体重"] = 0.5
        elif r.weight >= 57.0: parts["体重"] = -0.5
        else: parts["体重"] = 0.0
    else:
        parts["体重"] = 0.0

    if r.f_count == 1: parts["F持ち"] = -1.5
    elif r.f_count >= 2: parts["F持ち"] = -3.0
    else: parts["F持ち"] = 0.0

    parts["場×コース"] = venue_course_bonus(venue, lane)
    parts["場×攻め"]   = venue_attack_bonus(venue, lane)

    total = round(sum(parts.values()), 2)
    parts["合計"] = total
    return parts

# ============================================================
# LightGBM AI予測の追加設定
# ============================================================
@st.cache_resource
def load_lgb_model():
    try:
        return lgb.Booster(model_file='lgb_model.txt')
    except Exception as e:
        return None

def get_lgb_features(r: Racer, venue: str, lane: int) -> list:
    return [
        float(lane), float(r.win_rate or 0.0), float(r.avg_st or 0.17), float(r.motor_2rate or 0.0)
    ]

def rank_all(racers: List[Racer], venue: str) -> List[Dict]:
    out = []
    lgb_model = load_lgb_model()
    for i, r in enumerate(racers):
        lane = i + 1
        bd = score_boat(r, venue, lane)
        
        ai_score = 0.0
        if lgb_model is not None:
            features = get_lgb_features(r, venue, lane)
            ai_pred = lgb_model.predict([features])[0]
            ai_score = round(ai_pred * 10, 2)
            bd["AI加点"] = ai_score 
            
        final_score = bd["合計"] + ai_score
        bd["総合計(AI込)"] = round(final_score, 2)
        out.append({"lane": lane, "racer": r, "score": final_score, "breakdown": bd})
        
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def make_bets(ranked: List[Dict], strategy: str = "standard", odds_map: Optional[Dict[str, float]] = None, min_odds: float = 0.0) -> List[str]:
    if len(ranked) < 4: return []
    lanes = [x["lane"] for x in ranked]
    l1, l2, l3, l4, l5 = lanes[0], lanes[1], lanes[2], lanes[3], lanes[4]

    if strategy == "safe": raw = [f"{l1}-{l2}-{l3}", f"{l1}-{l3}-{l2}"]
    elif strategy == "wide":
        raw = []
        for s in (l2, l3, l4):
            for t in (l2, l3, l4, l5):
                if t != s and t != l1 and s != l1:
                    c = f"{l1}-{s}-{t}"
                    if c not in raw: raw.append(c)
    else:
        raw = []
        for s in (l2, l3):
            for t in (l2, l3, l4):
                if t != s and t != l1 and s != l1:
                    c = f"{l1}-{s}-{t}"
                    if c not in raw: raw.append(c)

    if odds_map and min_odds > 0:
        raw = [c for c in raw if odds_map.get(c, 0) >= min_odds]
    return raw

def strategy_label(strategy: str) -> str:
    return {"safe": "安全2点", "standard": "標準4点", "wide": "拡張9点"}.get(strategy, strategy)

# ============================================================
# 定数・UI初期設定
# ============================================================
st.set_page_config(page_title="v17.4 全艇スコア解析(爆速版)", layout="centered")
st.title("🚤 v17.4 全艇スコア解析")
st.caption("AI(LightGBM)搭載 ＆ 並列処理による爆速データ取得版！")

UCHI   = "https://uchisankaku.sakura.ne.jp"
BOAT   = "https://www.boatrace.jp/owpc/pc/race"
KYOTEI = "https://kyotei.sakura.ne.jp"
UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36"}

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村",
}
NAME_JCD = {v: k for k, v in JCD_NAME.items()}

# ============================================================
# HTTP (並列処理対応セッション)
# ============================================================
req_session = requests.Session()
req_session.headers.update(UA)

def get(url: str, retries: int = 2) -> Optional[str]:
    """通信を使い回すセッションで高速にGETし、失敗時は自動リトライする"""
    for _ in range(retries):
        try:
            r = req_session.get(url, timeout=10)
            r.encoding = r.apparent_encoding or "utf-8"
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
    return None

def fnum(s: Optional[str]) -> Optional[float]:
    if not s: return None
    m = re.search(r"-?\d+\.\d+|-?\d+", s)
    return float(m.group()) if m else None

# ============================================================
# スレイピング関数 (公式・kyotei)
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def boatrace_venues(date_str: str) -> List[int]:
    html = get(f"{BOAT}/index?hd={date_str}")
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    pat = re.compile(rf'raceindex\?jcd=(\d+)&hd={re.escape(date_str)}')
    jcds = set()
    for a in soup.find_all("a", href=True):
        m = pat.search(a["href"])
        if m:
            jcd = int(m.group(1))
            if jcd in JCD_NAME: jcds.add(jcd)
    return sorted(jcds)

@st.cache_data(ttl=600, show_spinner=False)
def venues_for_date(d: datetime.date) -> List[Tuple[int, str]]:
    today_ = datetime.now().date()
    date_str = d.strftime("%Y%m%d")
    jcds = boatrace_venues(date_str)
    if jcds: return [(j, JCD_NAME[j]) for j in jcds if j in JCD_NAME]

    if d == today_: url = f"{UCHI}/raceindex.php"
    elif d == today_ + timedelta(days=1): url = f"{UCHI}/raceindex.php?date=tomorrow"
    else:
        kjcds = kyotei_venues(date_str)
        if kjcds: return [(j, JCD_NAME[j]) for j in kjcds if j in JCD_NAME]
        return []

    html = get(url)
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    seen, result = set(), []
    for a in soup.find_all("a", href=True):
        m = re.search(r"racelist\.php\?jcode=(\d+)", a["href"])
        if not m or "出走表" not in a.get_text(): continue
        jcd = int(m.group(1))
        if jcd not in seen and jcd in JCD_NAME:
            seen.add(jcd)
            result.append((jcd, JCD_NAME[jcd]))
    result.sort(key=lambda x: x[0])
    return result

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_official_racelist_html(jcd: int, rno: int, date_str: str) -> Optional[str]:
    return get(f"{BOAT}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}")

def _parse_official_racelist(html: str) -> List[Racer]:
    soup = BeautifulSoup(html, "html.parser")
    target = None
    for tbl in soup.find_all("table"):
        head = tbl.get_text(" ", strip=True)
        if all(k in head for k in ["ボートレーサー", "全国", "当地", "モーター"]):
            target = tbl
            break
    if not target: return []

    racers: List[Racer] = []
    rows = target.find_all("tr")
    lane_map = {"１": 1, "２": 2, "３": 3, "４": 4, "５": 5, "６": 6, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    main_rows: List[Tuple[int, "BeautifulSoup"]] = []
    seen_lanes = set()
    for tr in rows:
        a_test = tr.find("a", href=re.compile(r"profile\?toban=\d+"))
        if not a_test: continue
        cells = tr.find_all(["td", "th"])
        if not cells: continue
        first_text = cells[0].get_text(strip=True)
        if first_text in lane_map and lane_map[first_text] not in seen_lanes:
            lane = lane_map[first_text]
            main_rows.append((lane, tr))
            seen_lanes.add(lane)
            if len(main_rows) >= 6: break

    if len(main_rows) < 6: return []
    main_rows.sort(key=lambda x: x[0])
    all_trs = list(target.find_all("tr"))
    main_tr_indices = {lane: all_trs.index(tr) for lane, tr in main_rows if tr in all_trs}

    for lane, tr in main_rows:
        full_text = re.sub(r"\s+", " ", tr.get_text(" ", strip=True))
        cls_ = re.search(r"/\s*(A1|A2|B1|B2)\b", full_text)
        cls_ = cls_.group(1) if cls_ else ""
        wt_match = re.search(r"(\d+\.\d+)\s*kg", full_text)
        weight = float(wt_match.group(1)) if wt_match else None
        a_tag = tr.find("a", href=re.compile(r"profile\?toban=\d+"))
        name = a_tag.get_text(strip=True).replace(" ", "").replace("　", "") if a_tag else f"選手{lane}"

        fl_match = re.search(r"F\s*(\d+)\s+L\s*(\d+)", full_text)
        f_count = int(fl_match.group(1)) if fl_match else 0

        avg_st, win_rate, motor_2rate = None, None, None
        if fl_match:
            tail = full_text[fl_match.end():]
            nums = re.findall(r"-?\d+\.\d+|\d+", tail)
            try: avg_st = float(nums[0]) if "." in nums[0] else None
            except: pass
            try: win_rate = float(nums[1])
            except: pass
            try:
                m2v = float(nums[8])
                motor_2rate = m2v / 100.0 if m2v > 1.0 else m2v
            except: pass

        settle_st, settle_avg_rank = None, None
        idx = main_tr_indices.get(lane)
        if idx is not None and idx + 3 < len(all_trs):
            st_tr, fn_tr = all_trs[idx + 2], all_trs[idx + 3]
            st_cells = [td.get_text(strip=True) for td in st_tr.find_all(["td", "th"])]
            fn_cells = [td.get_text(strip=True) for td in fn_tr.find_all(["td", "th"])]
            st_vals = []
            for c in st_cells:
                if re.search(r"[FLK失]", c): continue
                if re.fullmatch(r"\.\d+", c): st_vals.append(float("0" + c))
                elif re.fullmatch(r"0\.\d+", c): st_vals.append(float(c))
            if st_vals: settle_st = round(sum(st_vals) / len(st_vals), 3)

            ranks = [int(c.translate(str.maketrans("１２３４５６", "123456"))) for c in fn_cells if re.fullmatch(r"[1-6]", c.translate(str.maketrans("１２３４５６", "123456")))]
            if ranks: settle_avg_rank = round(sum(ranks) / len(ranks), 2)

        racers.append(Racer(name=name, cls=cls_, win_rate=win_rate, avg_st=avg_st, settle_st=settle_st, settle_avg_rank=settle_avg_rank, motor_2rate=motor_2rate, f_count=f_count, weight=weight, course5_avg_st=avg_st))
    return racers

def fetch_race(jcd: int, rno: int, date_str: str) -> List[Racer]:
    html = _fetch_official_racelist_html(jcd, rno, date_str)
    return _parse_official_racelist(html) if html else []

# ★ ここが爆速化の要（並列処理） ★
def fetch_racelist(jcd: int, date_str: str) -> Dict[int, List[Racer]]:
    out: Dict[int, List[Racer]] = {}
    # 12レース分を同時に取得しにいく
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_rno = {executor.submit(fetch_race, jcd, rno, date_str): rno for rno in range(1, 13)}
        for future in concurrent.futures.as_completed(future_to_rno):
            rno = future_to_rno[future]
            try:
                racers = future.result()
                if len(racers) == 6:
                    out[rno] = racers
            except Exception:
                pass
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kyotei_day(date_str: str) -> Dict[Tuple[int, int], int]:
    html = get(f"{KYOTEI}/kako-{date_str}.html")
    if not html: return {}
    soup = BeautifulSoup(html, "html.parser")
    payouts = {}
    pat = re.compile(rf'info-{re.escape(date_str)}-(\d+)-(\d+)\.html')
    for a in soup.find_all("a", href=True):
        if "race.kyotei.club" not in a["href"]: continue
        m = pat.search(a["href"])
        if not m: continue
        jcd, rno = int(m.group(1)), int(m.group(2))
        if jcd not in JCD_NAME or rno not in range(1, 13) or (jcd, rno) in payouts: continue
        tr = a.find_parent("tr")
        if not tr: continue
        for a2 in tr.find_all("a", href=True):
            if "info.kyotei.fun" not in a2["href"]: continue
            txt = a2.get_text(strip=True).replace(",", "")
            if re.fullmatch(r"\d+", txt) and int(txt) > 0:
                payouts[(jcd, rno)] = int(txt)
                break
    return payouts

def kyotei_venues(date_str: str) -> List[int]:
    html = get(f"{KYOTEI}/kako-{date_str}.html")
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    pat = re.compile(rf'info-{re.escape(date_str)}-(\d+)-\d+\.html')
    jcds = {int(m.group(1)) for a in soup.find_all("a", href=True) if "race.kyotei.club" in a["href"] and (m := pat.search(a["href"])) and int(m.group(1)) in JCD_NAME}
    return sorted(jcds)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_result(date_str: str, jcd: int, rno: int) -> Optional[Dict]:
    html  = get(f"{BOAT}/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    if not html: return None
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if "まだ結果がありません" in text or "発売中" in text: return None

    finish = [int(tds[1]) for tbody in soup.select("div.table1 table tbody") if len(tds := [td.get_text(strip=True) for td in tbody.find_all("td")]) >= 2 and re.fullmatch(r"\d+", tds[0]) and re.fullmatch(r"[1-6]", tds[1])][:6]
    if len(finish) < 3: return None

    combo, pay = None, 0
    m = re.search(r"3連単.*?([1-6])\s*[-ー‐]\s*([1-6])\s*[-ー‐]\s*([1-6]).*?¥?\s*([\d,]+)", text)
    if m: combo, pay = f"{m.group(1)}-{m.group(2)}-{m.group(3)}", int(m.group(4).replace(",", ""))
    kim = km.group(1) if (km := re.search(r"(逃げ|まくり差し|まくり|差し|抜き|恵まれ)", text)) else None

    return {"finish": finish, "combo": combo, "payout": pay, "kimarite": kim}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds_3t(date_str: str, jcd: int, rno: int) -> Dict[str, float]:
    html = get(f"{BOAT}/odds3t?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    if not html: return {}
    soup = BeautifulSoup(html, "html.parser")
    if "発売前" in soup.get_text() or "まだ発売されていません" in soup.get_text(): return {}

    combo_order = [f"{a}-{b}-{c}" for a in range(1, 7) for b in range(1, 7) if b != a for c in range(1, 7) if c not in (a, b)]
    
    def collect_odds_cells(tbl):
        return [float(txt) for td in tbl.find_all("td") if (txt := td.get_text(strip=True).replace(",", "").replace(" ", "")) and (re.fullmatch(r"\d+\.\d+", txt) or (re.fullmatch(r"\d+", txt) and len(txt) >= 2))]
    
    target_cells = next((cells for tbl in soup.find_all("table") if len(cells := collect_odds_cells(tbl)) == 120), [])
    if not target_cells:
        best_cells = max((collect_odds_cells(tbl) for tbl in soup.find_all("table")), key=len, default=[])
        if len(best_cells) == 120: target_cells = best_cells
        else: return {}

    return {combo_order[(i % 6) * 20 + (i // 6)]: v for i, v in enumerate(target_cells) if 0 <= (i % 6) * 20 + (i // 6) < 120}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_exhibit_times(date_str: str, jcd: int, rno: int) -> List[Optional[float]]:
    html = get(f"{BOAT}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    if not html: return [None] * 6
    soup = BeautifulSoup(html, "html.parser")
    if "情報がありません" in soup.get_text() or "発表前" in soup.get_text(): return [None] * 6

    times: List[Optional[float]] = [None] * 6
    target_table = next((tbl for tbl in soup.find_all("table") if all(k in tbl.get_text(" ", strip=True) for k in ["展示", "タイム", "ボートレーサー"])), None)
    if not target_table: return times

    lane_idx = 0
    for tr in target_table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells: continue
        first = cells[0].get_text(strip=True)
        if first in ("1", "2", "3", "4", "5", "6") and lane_idx < 6:
            lane = int(first) - 1
            for m in re.finditer(r"\b(\d+\.\d+)\b", tr.get_text(" ", strip=True)):
                v = float(m.group(1))
                if 5.50 <= v <= 8.50:
                    times[lane] = v
                    break
            lane_idx = lane + 1
    return times

def assign_exhibit_ranks(times: List[Optional[float]]) -> List[Optional[int]]:
    valid = sorted([(i, t) for i, t in enumerate(times) if t is not None], key=lambda x: x[1])
    ranks: List[Optional[int]] = [None] * 6
    for rank, (i, _) in enumerate(valid, start=1): ranks[i] = rank
    return ranks

def render_venue_summary(venue: str):
    if venue not in COURSE_WIN_RATE: return
    st.markdown(f"### 🗺️ 場の傾向 — {venue} {venue_tendency_label(venue)}")
    st.caption("kyoteibiyori集計 2023/03/12〜2024/03/12")
    nat = COURSE_WIN_RATE["全国"]
    ven = COURSE_WIN_RATE[venue]
    rows = [{"C": i+1, "基礎": f"{COURSE_BASE_POINTS[i+1]:+.1f}", "場1着率": f"{ven[i]:.1f}%", "場×C": f"{venue_course_bonus(venue, i+1):+.2f}", "場×攻": f"{venue_attack_bonus(venue, i+1):+.2f}" if i+1 >= 2 else "-"} for i in range(6)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# UI処理 (タブ1・タブ2)
# ============================================================
today = datetime.now().date()
tab1, tab2 = st.tabs(["🔍 1レース解析", "📊 期間バックテスト/当日スキャン"])

with tab1:
    c1, c2 = st.columns([3, 1])
    with c1: t1_date = st.date_input("日付", value=today, min_value=datetime(2020,1,1).date(), max_value=today + timedelta(days=1), key="t1_date")
    with c2:
        st.write("")
        st.write("")
        if st.button("🔄", key="t1_reload"): st.cache_data.clear()

    dstr = t1_date.strftime("%Y%m%d")
    with st.spinner("開催場を取得中..."): vlist = venues_for_date(t1_date)

    if not vlist:
        st.warning("開催場が見つかりません。")
    else:
        vname = st.selectbox("開催場", [n for _, n in vlist], key="t1_venue")
        jcd   = NAME_JCD[vname]
        rno   = st.selectbox("レース", list(range(1, 13)), format_func=lambda r: f"{r}R", key="t1_rno")

        sc1, sc2 = st.columns([2, 3])
        with sc1: t1_strategy = st.radio("戦略", ["safe", "standard", "wide"], index=1, format_func=strategy_label, horizontal=True, key="t1_strategy")
        with sc2: t1_min_odds = st.slider("最低オッズ (0=無効)", min_value=0.0, max_value=30.0, value=0.0, step=1.0, key="t1_min_odds", help="指定オッズ未満を除外")

        with st.expander("🗺️ 場の傾向を見る", expanded=False): render_venue_summary(vname)

        if st.button("🎯 解析する", type="primary", use_container_width=True, key="t1_run"):
            with st.spinner("選手データ取得中（並列処理で爆速！）..."):
                all_r = fetch_racelist(jcd, dstr)

            racers = all_r.get(rno)
            if not racers or len(racers) < 6:
                st.error("選手データを取得できませんでした。時間をおいて再試行してください。")
            else:
                exhibit_times: List[Optional[float]] = [None] * 6
                if t1_date <= today + timedelta(days=1):
                    with st.spinner("展示タイム取得中..."): exhibit_times = fetch_exhibit_times(dstr, jcd, rno)
                exhibit_ranks = assign_exhibit_ranks(exhibit_times)
                exhibit_loaded = any(t is not None for t in exhibit_times)
                for i, r in enumerate(racers): r.exhibit_rank = exhibit_ranks[i]

                ranked = rank_all(racers, vname)
                odds_map = fetch_odds_3t(dstr, jcd, rno) if t1_date >= today else {}
                bets = make_bets(ranked, strategy=t1_strategy, odds_map=odds_map if odds_map else None, min_odds=t1_min_odds)
                res = fetch_result(dstr, jcd, rno) if t1_date <= today else None

                st.markdown(f"### {vname} {rno}R {venue_tendency_label(vname)}")
                st.caption(f"戦略: **{strategy_label(t1_strategy)}**" + (f" / オッズ≥{t1_min_odds:.0f}倍" if t1_min_odds > 0 else "") + (" / オッズ取得成功" if odds_map else "") + (" / 展示適用" if exhibit_loaded else " / 展示未発表"))

                if len(ranked) >= 4:
                    m12, m23, m34 = ranked[0]["score"] - ranked[1]["score"], ranked[1]["score"] - ranked[2]["score"], ranked[2]["score"] - ranked[3]["score"]
                    conf = "🟢 高信頼" if m12 >= 1.5 else "🟡 中信頼" if m12 >= 0.8 else "🔴 低信頼"
                    st.caption(f"スコア差: 1-2={m12:+.2f} / 2-3={m23:+.2f} / 3-4={m34:+.2f} {conf}")

                df_rows = []
                for rk, x in enumerate(ranked, 1):
                    r, bd = x["racer"], x["breakdown"]
                    df_rows.append({
                        "順位": rk, "艇": x["lane"], "名前": r.name, "級": r.cls or "-",
                        "勝率": f"{r.win_rate:.2f}" if r.win_rate else "-", "ST": f"{r.avg_st:.2f}" if r.avg_st else "-",
                        "M2率": f"{r.motor_2rate*100:.0f}" if r.motor_2rate else "-", "節平順": f"{r.settle_avg_rank:.1f}" if r.settle_avg_rank else "-",
                        "節ST": f"{r.settle_st:.2f}" if r.settle_st else "-", "F": r.f_count or 0,
                        "展示": f"{exhibit_times[x['lane']-1]:.2f}({r.exhibit_rank})" if exhibit_times[x['lane']-1] and r.exhibit_rank else "-",
                        "基礎+場": f"{bd['コース基礎']+bd['場×コース']+bd['場×攻め']:+.2f}", "スコア": f"{x['score']:+.2f}"
                    })
                st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

                with st.expander("📋 スコア内訳(上位3艇)"):
                    br_rows = []
                    for rk, x in enumerate(ranked[:3], 1):
                        row = {"順位": rk, "艇": x["lane"], "名前": x["racer"].name}
                        row.update({k: f"{v:+.2f}" if isinstance(v, float) else v for k, v in x["breakdown"].items() if k not in ("合計", "総合計(AI込)")})
                        row["合計"] = f"{x['score']:+.2f}"
                        br_rows.append(row)
                    st.dataframe(pd.DataFrame(br_rows), use_container_width=True, hide_index=True)

                if bets:
                    st.subheader(f"🎯 推奨買い目 ({strategy_label(t1_strategy)} / {len(bets)}点)")
                    st.markdown(" / ".join(f"**{i+1}位** {ranked[i]['lane']}号艇({ranked[i]['racer'].name})" for i in range(min(4, len(ranked)))))
                    if odds_map:
                        bet_rows, total_inv, odds_values = [], len(bets) * 100, [odds_map.get(b, 0) for b in bets]
                        for b, o in zip(bets, odds_values):
                            bet_rows.append({"買い目": b, "オッズ": f"{o:.1f}倍" if o > 0 else "-", "的中時回収": f"¥{int(o*100):,}" if o > 0 else "-", "回収率": f"{o*100/total_inv*100:.0f}%" if o > 0 else "-"})
                        st.dataframe(pd.DataFrame(bet_rows), use_container_width=True, hide_index=True)
                        valid_odds = [o for o in odds_values if o > 0]
                        if valid_odds: st.caption(f"💡 オッズ範囲: {min(valid_odds):.1f}〜{max(valid_odds):.1f}倍 / 平均{sum(valid_odds)/len(valid_odds):.1f}倍 / 投資¥{total_inv} / 1点でも的中すれば¥{int(min(valid_odds)*100):,}回収")
                    else:
                        st.code("\n".join(bets))
                        st.caption("オッズ未取得 (未発売または解析失敗)")
                elif t1_min_odds > 0: st.warning(f"⚠️ オッズ≥{t1_min_odds:.0f}倍の買い目がありません。戦略変更またはオッズ閾値を下げてください。")

                if res:
                    st.markdown("---")
                    st.subheader("🏁 レース結果")
                    c1r, c2r = st.columns(2)
                    c1r.markdown(f"**着順**: {'-'.join(str(n) for n in res['finish'][:3])}")
                    if res.get("kimarite"): c2r.markdown(f"**決まり手**: {res['kimarite']}")
                    hit, inv_yen = res["combo"] in bets if res["combo"] else False, len(bets) * 100
                    payout = res["payout"] if hit else 0
                    rr, profit = (payout / inv_yen * 100) if inv_yen > 0 else 0, payout - inv_yen
                    if res.get("combo"): st.metric("3連単 払戻", res["combo"], f"¥{res['payout']:,}")
                    st.markdown(f"### 💰 買い目収支（{len(bets)}点=¥{inv_yen}）")
                    ca, cb, cc = st.columns(3)
                    ca.metric("投資", f"¥{inv_yen}"); cb.metric("回収", f"¥{payout:,}"); cc.metric("回収率", f"{rr:.0f}%", f"{profit:+,}円", delta_color="normal" if rr >= 100 else "inverse")
                    if hit: st.success(f"✅ 的中: `{res['combo']}` → ¥{payout:,}")
                    else: st.info("買い目不的中")
                else: st.caption("🕓 結果未確定（未発走または取得失敗）")

with tab2:
    st.subheader("📊 期間バックテスト / 当日スキャン")
    st.caption("予想1位が1号艇のレースを抽出。過去日は結果取得、当日は予想のみ。")

    bc1, bc2 = st.columns(2)
    with bc1: bt_s = st.date_input("開始日", value=today - timedelta(days=3), min_value=datetime(2020,1,1).date(), max_value=today, key="bt_s")
    with bc2: bt_e = st.date_input("終了日", value=today, min_value=datetime(2020,1,1).date(), max_value=today, key="bt_e")

    bt_venues = st.multiselect("対象場 (空=全場)", options=[JCD_NAME[j] for j in sorted(JCD_NAME.keys())], default=[], key="bt_venues", help="選択した場のみ解析。")
    bt_target_jcds = {NAME_JCD[v] for v in bt_venues if v in NAME_JCD} if bt_venues else None
    
    if bt_target_jcds: st.caption(f"🎯 対象 {len(bt_target_jcds)} 場")
    else: st.caption("⚡ 全場対象（並列処理で高速化されていますが、全場だと数分かかります）")

    bt_strategy = st.radio("戦略", ["safe", "standard", "wide"], index=1, format_func=strategy_label, horizontal=True, key="bt_strategy")

    with st.expander("🔧 品質フィルター (デフォルト推奨)", expanded=True):
        qc1, qc2 = st.columns(2)
        with qc1:
            bt_skip_b2 = st.checkbox("B2の1号艇を除外", value=True, key="bt_skip_b2")
            bt_skip_hard = st.checkbox("戸田/江戸川/平和島を除外", value=False, key="bt_skip_hard")
        with qc2: bt_min_winrate = st.slider("1号艇勝率の下限", 0.0, 8.0, 5.0, 0.5, key="bt_min_wr")
        bt_use_exhibit = st.checkbox("展示タイムを反映 (精度向上・処理時間増)", value=False, key="bt_use_exhibit")

    st.markdown("**スコア差フィルター** (各順位の差で信頼度を厳格化)")
    mc1, mc2, mc3 = st.columns(3)
    with mc1: min_margin_12 = st.slider("1位-2位 ≥", 0.0, 3.0, 0.8, 0.1, key="bt_margin12")
    with mc2: min_margin_23 = st.slider("2位-3位 ≥", 0.0, 2.0, 0.0, 0.1, key="bt_margin23")
    with mc3: min_margin_34 = st.slider("3位-4位 ≥", 0.0, 2.0, 0.0, 0.1, key="bt_margin34")

    if bt_s > bt_e: st.warning("開始日 ≤ 終了日 にしてください。")
    else:
        n_days = (bt_e - bt_s).days + 1
        if st.button("🔍 1号艇1位を検索", type="primary", use_container_width=True, key="bt_run"):
            days = [bt_s + timedelta(days=i) for i in range(n_days)]
            prog, status, matches, hard_venues = st.progress(0.0), st.empty(), [], {"戸田", "江戸川", "平和島"}

            for idx, day in enumerate(days):
                dstr_bt, is_past = day.strftime("%Y%m%d"), day < today
                prog.progress((idx + 1) / n_days, text=f"[{idx+1}/{n_days}] {dstr_bt} 処理中{' (当日)' if not is_past else ''}...")

                if is_past:
                    status.caption(f"📡 {dstr_bt} — kyotei 払戻取得中...")
                    payouts = fetch_kyotei_day(dstr_bt)
                    if not payouts: continue
                    open_jcds = kyotei_venues(dstr_bt) or sorted({jcd for jcd, _ in payouts.keys()})
                else:
                    status.caption(f"📡 {dstr_bt} — 当日開催場取得中...")
                    payouts, open_jcds = {}, [j for j, _ in venues_for_date(day)]
                    if not open_jcds: continue

                for jcd_bt in open_jcds:
                    venue_bt = JCD_NAME.get(jcd_bt, "")
                    if not venue_bt or (bt_target_jcds and jcd_bt not in bt_target_jcds) or (bt_skip_hard and venue_bt in hard_venues): continue
                    
                    status.caption(f"📡 {dstr_bt} {venue_bt} — 選手データ取得中（並列処理中🚀）...")
                    races = fetch_racelist(jcd_bt, dstr_bt)
                    if not races: continue

                    for rno_bt, racers_bt in sorted(races.items()):
                        if len(racers_bt) < 6: continue
                        if bt_use_exhibit:
                            ex_times = fetch_exhibit_times(dstr_bt, jcd_bt, rno_bt)
                            for i, r_ in enumerate(racers_bt): r_.exhibit_rank = assign_exhibit_ranks(ex_times)[i]

                        ranked_bt = rank_all(racers_bt, venue_bt)
                        if ranked_bt[0]["lane"] != 1: continue

                        ichi = racers_bt[0]
                        if (bt_skip_b2 and ichi.cls == "B2") or (ichi.win_rate and ichi.win_rate < bt_min_winrate): continue

                        margin_12, margin_23, margin_34 = ranked_bt[0]["score"] - ranked_bt[1]["score"], ranked_bt[1]["score"] - ranked_bt[2]["score"], ranked_bt[2]["score"] - ranked_bt[3]["score"]
                        if margin_12 < min_margin_12 or margin_23 < min_margin_23 or margin_34 < min_margin_34: continue

                        bets_bt, top_score, inv_bt = make_bets(ranked_bt, strategy=bt_strategy), ranked_bt[0]["score"], len(make_bets(ranked_bt, strategy=bt_strategy)) * 100

                        if not is_past:
                            matches.append({"日付": dstr_bt, "場": venue_bt, "R": rno_bt, "スコア": top_score, "差12": margin_12, "差23": margin_23, "差34": margin_34, "_bets": bets_bt, "_inv": inv_bt, "結果": "未発走", "払戻": 0, "_hit": None, "_payout": 0, "_status": "pending"})
                            continue

                        pay_kyotei = payouts.get((jcd_bt, rno_bt))
                        if pay_kyotei is None:
                            matches.append({"日付": dstr_bt, "場": venue_bt, "R": rno_bt, "スコア": top_score, "差12": margin_12, "差23": margin_23, "差34": margin_34, "_bets": bets_bt, "_inv": inv_bt, "結果": "未確定", "払戻": 0, "_hit": None, "_payout": 0, "_status": "unresolved"})
                            continue

                        status.caption(f"📡 {dstr_bt} {venue_bt} {rno_bt}R — 着順確認...")
                        res_bt = fetch_result(dstr_bt, jcd_bt, rno_bt)
                        combo_bt, hit_bt, payout_bt, status_bt = (res_bt["combo"], res_bt["combo"] in bets_bt, pay_kyotei if res_bt["combo"] in bets_bt else 0, "resolved") if res_bt and res_bt["combo"] else ("取得失敗", None, 0, "unresolved")

                        matches.append({"日付": dstr_bt, "場": venue_bt, "R": rno_bt, "スコア": top_score, "差12": margin_12, "差23": margin_23, "差34": margin_34, "_bets": bets_bt, "_inv": inv_bt, "結果": combo_bt, "払戻": pay_kyotei, "_hit": hit_bt, "_payout": payout_bt, "_status": status_bt})

            prog.empty(); status.empty()
            st.session_state["bt_matches"] = matches

        if "bt_matches" in st.session_state:
            M = st.session_state["bt_matches"]
            if not M: st.warning("対象期間に条件を満たすレースが見つかりませんでした。")
            else:
                n_resv, n_hit, inv, ret = sum(1 for m in M if m["_status"] == "resolved"), len([m for m in M if m["_hit"] is True]), sum(m["_inv"] for m in M if m["_status"] == "resolved"), sum(m["_payout"] for m in M if m["_status"] == "resolved")
                
                st.success(f"✅ {len(M)}件 抽出 (結果確定: {n_resv} / 未発走: {sum(1 for m in M if m['_status'] == 'pending')} / 未確定: {sum(1 for m in M if m['_status'] == 'unresolved')})")
                ca, cb, cc, cd = st.columns(4)
                ca.metric("対象", f"{len(M)}件")
                cb.metric("的中", f"{n_hit}/{n_resv}", f"{round(n_hit/n_resv*100,1)}%" if n_resv else "-")
                cc.metric("回収率", f"{round(ret/inv*100,1)}%" if n_resv else "-", f"{ret-inv:+,}円" if n_resv else None, delta_color="normal" if inv>0 and ret/inv>=1 else "inverse")
                cd.metric("投資/回収", f"¥{inv:,} / ¥{ret:,}")

                if n_resv > 0:
                    venue_stats = {}
                    for m in [m for m in M if m["_status"] == "resolved"]:
                        s = venue_stats.setdefault(m["場"], {"n": 0, "hit": 0, "inv": 0, "pay": 0})
                        s["n"] += 1; s["hit"] += 1 if m["_hit"] else 0; s["inv"] += m["_inv"]; s["pay"] += m["_payout"]
                    st.dataframe(pd.DataFrame([{"場": v, "R数": s["n"], "的中": s["hit"], "的中率": f"{s['hit']/s['n']*100:.0f}%", "回収": f"¥{s['pay']:,}", "回収率": f"{s['pay']/s['inv']*100:.0f}%"} for v, s in sorted(venue_stats.items(), key=lambda kv: -kv[1]["n"])]), use_container_width=True, hide_index=True)

                st.markdown("### 📋 レース一覧")
                st.dataframe(pd.DataFrame([{"日付": m["日付"], "場": m["場"], "R": m["R"], "スコア": f"{m['スコア']:+.2f}", "差12": f"{m['差12']:+.2f}", "差23": f"{m['差23']:+.2f}", "差34": f"{m['差34']:+.2f}", "点数": len(m["_bets"]), "結果": m["結果"], "払戻": f"¥{m['払戻']:,}" if m["払戻"] else "-", "判定": "✅" if m["_hit"] is True else "✕" if m["_hit"] is False else "⏳" if m["_status"] == "pending" else "?"} for m in M]), use_container_width=True, hide_index=True)
