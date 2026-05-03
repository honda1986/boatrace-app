# -*- coding: utf-8 -*-
"""
v17.3 全艇スコア解析アプリ（タブ2に対象場マルチセレクト追加で時短）
=======================================================
v17.2 からの変更点:
  - タブ2に「対象場 (空=全場)」マルチセレクトを追加
  - 公式パーサー(v17系)は1日24場×12R=最大288リクエストとなり、
    全場対象だと処理時間が非常に長くなる問題に対処
  - 場を3〜5に絞ることで、リクエスト数を1/5〜1/8に削減
  - 推定リクエスト数を画面に表示

  使い方:
  - 「対象場」を空にすると従来通り全場対象 (時間がかかる)
  - 場を選択すると、その場のみで解析実行
  - 同じ場の傾向を継続的にバックテストする時に特に有効

タブ1: 個別レース解析
タブ2: 期間バックテスト + 当日予想スキャン (場選択可能)

起動: streamlit run v17_3_all_boats_app.py
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import lightgbm as lgb
import numpy as np


# ============================================================
# kyoteibiyori.com 場別コース別データ
# 集計期間: 2023年03月12日 - 2024年03月12日
# 出典: https://kyoteibiyori.com/blog/20240312001
# ============================================================

# 1着率(%) - インデックス[0..5] = [1C, 2C, 3C, 4C, 5C, 6C]
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

# 差し率(%) - インデックス[0..4] = [2C, 3C, 4C, 5C, 6C]
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

# まくり率(%) - インデックス[0..4] = [2C, 3C, 4C, 5C, 6C]
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

# まくり差し率(%) - インデックス[0..3] = [3C, 4C, 5C, 6C] (2Cは対象外)
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

# NEW: コース基礎点 (全国1着率に比例)
# 1C: 55% → +2.0、2C: 14% → +0.5、3C: 13% → 0、
# 4C: 11% → -0.5、5C: 6% → -1.0、6C: 2% → -1.5
COURSE_BASE_POINTS: Dict[int, float] = {
    1: 2.0, 2: 0.5, 3: 0.0, 4: -0.5, 5: -1.0, 6: -1.5,
}

# 係数: 場別1着率偏差に対する補正の強さ (v16.1:0.15 → v16.2:0.10に緩和)
VENUE_WIN_RATE_COEF = 0.10

# 係数: 攻めバイアス(差し+まくり+まくり差し)偏差の強さ (控えめ)
VENUE_ATTACK_COEF = 0.08


def venue_course_bonus(venue: str, lane: int) -> float:
    """場別コース別1着率の全国偏差をスコア補正に変換。"""
    if venue not in COURSE_WIN_RATE or not 1 <= lane <= 6:
        return 0.0
    nat = COURSE_WIN_RATE["全国"][lane - 1]
    ven = COURSE_WIN_RATE[venue][lane - 1]
    return round((ven - nat) * VENUE_WIN_RATE_COEF, 2)


def venue_attack_bonus(venue: str, lane: int) -> float:
    """攻めコース(3-6)のまくり+まくり差し率の全国偏差を補正。
    2Cの差しは1着率(場×コース)に既に反映されるためダブルカウント回避で除外。"""
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
    if diff_1c >= 5:
        return f"🟢 イン強烈({v[0]:.1f}%)"
    elif diff_1c >= 2:
        return f"🟢 インやや有利({v[0]:.1f}%)"
    elif diff_1c <= -5:
        return f"🔴 イン不利({v[0]:.1f}%)・荒れ水面"
    elif diff_1c <= -2:
        return f"🟡 インやや不利({v[0]:.1f}%)"
    else:
        return f"⚪ 標準({v[0]:.1f}%)"


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
    settle_avg_rank: Optional[float] = None  # v17.2: 今節2連率→今節平均順位
    motor_2rate: Optional[float] = None
    f_count: int = 0
    exhibit_rank: Optional[int] = None
    course5_avg_st: Optional[float] = None
    weight: Optional[float] = None
    makuri_rate: Optional[float] = None


def _band(v: Optional[float],
          bands: List[Tuple[float, float, float]],
          default: float = 0.0) -> float:
    if v is None:
        return default
    for lo, hi, pts in bands:
        if lo <= v < hi:
            return pts
    return default


def score_boat(r: Racer, venue: str, lane: int) -> Dict[str, float]:
    """全艇共通スコア。内訳をdictで返す。v16.3でバンド細分化。"""
    parts: Dict[str, float] = {}

    # コース基礎点(1着率比例)
    parts["コース基礎"] = COURSE_BASE_POINTS.get(lane, 0.0)

    parts["級別"] = {"A1": 2.5, "A2": 1.5, "B1": 0.0, "B2": -1.5}.get(r.cls, 0.0)
    parts["勝率"] = _band(r.win_rate, [
        (6.50, 99, 1.5), (5.50, 6.50, 1.0), (5.00, 5.50, 0.5),
        (4.00, 5.00, -0.5), (0.00, 4.00, -1.2),
    ])

    # v16.3: STバンドを5段階に細分化
    st_val = r.course5_avg_st if r.course5_avg_st is not None else r.avg_st
    att    = 1.0 if r.course5_avg_st is not None else 0.5
    parts["ST"] = att * _band(st_val, [
        (0.00, 0.14, 2.0),   # 超絶スタート
        (0.14, 0.16, 1.3),   # 優秀
        (0.16, 0.18, 0.5),   # 良
        (0.18, 0.20, -0.3),  # 凡
        (0.20, 9.99, -1.3),  # 遅い
    ])

    # v17.2: 節2率 → 節平均順位 に変更
    # 平均順位は1.0(全戦1着)〜6.0(全戦6着)の範囲。低いほど好調
    parts["節平順"] = _band(r.settle_avg_rank, [
        (0.99, 1.50, 2.0),    # 圧倒的好調 (平均ほぼ1着)
        (1.50, 2.50, 1.2),    # 好調 (平均2着前後)
        (2.50, 3.50, 0.3),    # 普通 (全国平均3.5)
        (3.50, 4.50, -0.5),   # やや不調
        (4.50, 6.01, -1.5),   # 不調
    ])

    # v16.3: 節ST改善を連続的に評価
    if r.settle_st is not None and r.avg_st is not None:
        delta = r.avg_st - r.settle_st  # 正 = 改善
        if delta >= 0.04:    parts["節ST改善"] = 1.5
        elif delta >= 0.02:  parts["節ST改善"] = 1.0
        elif delta >= 0.00:  parts["節ST改善"] = 0.3
        elif delta >= -0.02: parts["節ST改善"] = -0.3
        else:                parts["節ST改善"] = -1.0
    else:
        parts["節ST改善"] = 0.0

    parts["モーター"] = _band(r.motor_2rate, [
        (0.45, 1.01, 1.5), (0.35, 0.45, 0.8), (0.30, 0.35, 0.3),
        (0.25, 0.30, -0.3), (0.00, 0.25, -1.2),
    ])

    # v16.3: 展示順位を全6段階で評価
    exhibit_scores = {1: 1.5, 2: 0.8, 3: 0.3, 4: -0.2, 5: -0.6, 6: -1.0}
    parts["展示"] = exhibit_scores.get(r.exhibit_rank, 0.0)

    if r.weight is not None:
        if r.weight <= 52.0:   parts["体重"] = 0.5
        elif r.weight >= 57.0: parts["体重"] = -0.5
        else:                  parts["体重"] = 0.0
    else:
        parts["体重"] = 0.0

    # v16.3: F持ちペナルティを強化
    if r.f_count == 1:   parts["F持ち"] = -1.5
    elif r.f_count >= 2: parts["F持ち"] = -3.0
    else:                parts["F持ち"] = 0.0

    parts["場×コース"] = venue_course_bonus(venue, lane)
    parts["場×攻め"]   = venue_attack_bonus(venue, lane)

    total = round(sum(parts.values()), 2)
    parts["合計"] = total
    return parts


# ============================================================
# LightGBM AI予測の追加
# ============================================================
@st.cache_resource
def load_lgb_model():
    try:
        return lgb.Booster(model_file='lgb_model.txt')
    except Exception as e:
        return None

def get_lgb_features(r: Racer, venue: str, lane: int) -> list:
    return [
        float(lane),
        float(r.win_rate or 0.0),
        float(r.avg_st or 0.17),
        float(r.motor_2rate or 0.0)
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


def make_bets(ranked: List[Dict], strategy: str = "standard",
              odds_map: Optional[Dict[str, float]] = None,
              min_odds: float = 0.0) -> List[str]:
    """3連単買い目生成。strategyで点数を切替。
    
    - 'safe'(2点):     1-{2,3}-{3,2}
    - 'standard'(4点): 1-{2,3}-{2,3,4}  (従来)
    - 'wide'(9点):     1-{2,3,4}-{2,3,4,5}
    
    odds_map指定時、min_odds未満の買い目を除外。
    """
    if len(ranked) < 4:
        return []
    lanes = [x["lane"] for x in ranked]
    l1, l2, l3, l4, l5 = lanes[0], lanes[1], lanes[2], lanes[3], lanes[4]

    if strategy == "safe":
        # 2点: 1-2位-3位, 1-3位-2位
        raw = [f"{l1}-{l2}-{l3}", f"{l1}-{l3}-{l2}"]
    elif strategy == "wide":
        # 9点: 1-{2,3,4}-{2,3,4,5}
        raw = []
        for s in (l2, l3, l4):
            for t in (l2, l3, l4, l5):
                if t != s and t != l1 and s != l1:
                    c = f"{l1}-{s}-{t}"
                    if c not in raw:
                        raw.append(c)
    else:
        # standard (4点): 1-{2,3}-{2,3,4}
        raw = []
        for s in (l2, l3):
            for t in (l2, l3, l4):
                if t != s and t != l1 and s != l1:
                    c = f"{l1}-{s}-{t}"
                    if c not in raw:
                        raw.append(c)

    # オッズフィルター
    if odds_map and min_odds > 0:
        raw = [c for c in raw if odds_map.get(c, 0) >= min_odds]

    return raw


def strategy_label(strategy: str) -> str:
    return {"safe": "安全2点", "standard": "標準4点", "wide": "拡張9点"}.get(strategy, strategy)


# ============================================================
# 定数
# ============================================================
st.set_page_config(page_title="v17.3 全艇スコア解析", layout="centered")
st.title("🚤 v17.3 全艇スコア解析")
st.caption("タブ2に対象場マルチセレクト追加で時短")

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
# HTTP
# ============================================================
def get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def fnum(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"-?\d+\.\d+|-?\d+", s)
    return float(m.group()) if m else None


# ============================================================
# uchisankaku: 開催場一覧
# ============================================================
# ============================================================
# boatrace.jp公式: 開催場一覧 (最優先のソース)
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def boatrace_venues(date_str: str) -> List[int]:
    """boatrace.jp公式 https://www.boatrace.jp/owpc/pc/race/index?hd=YYYYMMDD
    から当該日付に開催される場のjcd一覧を取得。
    
    HTMLには各場の raceindex?jcd=XX&hd=YYYYMMDD 形式リンクが含まれており、
    日付を完全一致させた正規表現で確実に抽出可能。
    過去・当日・明日すべて対応(過去日も公式サイトはアーカイブ保持)。
    """
    html = get(f"{BOAT}/index?hd={date_str}")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    pat = re.compile(rf'raceindex\?jcd=(\d+)&hd={re.escape(date_str)}')
    jcds = set()
    for a in soup.find_all("a", href=True):
        m = pat.search(a["href"])
        if m:
            jcd = int(m.group(1))
            if jcd in JCD_NAME:
                jcds.add(jcd)
    return sorted(jcds)


# ============================================================
# uchisankaku: 開催場一覧 (フォールバック)
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def venues_for_date(d: datetime.date) -> List[Tuple[int, str]]:
    """指定日の開催場一覧を返す。優先順位:
       1. boatrace.jp公式 (最も信頼できる、過去〜明日まで対応)
       2. uchisankaku (当日・明日のみ)
       3. kyotei.sakura.ne.jp (過去日のみ)
       
    全ソースで取得失敗した場合は空リストを返す(全24場フォールバックは撤廃)。
    """
    today_ = datetime.now().date()
    date_str = d.strftime("%Y%m%d")

    # 1. 公式サイトを最優先
    jcds = boatrace_venues(date_str)
    if jcds:
        return [(j, JCD_NAME[j]) for j in jcds if j in JCD_NAME]

    # 2. uchisankaku (当日・明日)
    if d == today_:
        url = f"{UCHI}/raceindex.php"
    elif d == today_ + timedelta(days=1):
        url = f"{UCHI}/raceindex.php?date=tomorrow"
    else:
        # 3. kyotei.sakura.ne.jp フォールバック (過去日)
        kjcds = kyotei_venues(date_str)
        if kjcds:
            return [(j, JCD_NAME[j]) for j in kjcds if j in JCD_NAME]
        # 全ソース失敗時: 空リストを返す (全24場フォールバックは撤廃)
        return []

    html = get(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    seen, result = set(), []
    for a in soup.find_all("a", href=True):
        m = re.search(r"racelist\.php\?jcode=(\d+)", a["href"])
        if not m or "出走表" not in a.get_text():
            continue
        jcd = int(m.group(1))
        if jcd not in seen and jcd in JCD_NAME:
            seen.add(jcd)
            result.append((jcd, JCD_NAME[jcd]))
    result.sort(key=lambda x: x[0])
    return result


# ============================================================
# uchisankaku: 出走表パーサ
# ============================================================
# ============================================================
# boatrace.jp公式: 出走表パーサー (新方式)
# ============================================================
# 公式 racelist ページの構造:
#   - 各艇の主行 = "枠番号 写真 [登録番号 / 級別 氏名 支部/出身地 年齢/体重]
#                   F数 L数 平均ST | 全国勝率 全国2率 全国3率 |
#                   当地勝率 当地2率 当地3率 | モーターNo モーター2率 モーター3率 |
#                   ボートNo ボート2率 ボート3率"
#   - 続いて今節成績(初日〜最終日)が3行: 進入コース行 / ST行 / 着順行
#
# このパーサーは正規表現ベースで艇ごとの主行を抽出する方式。
# uchisankakuの不安定なテーブル構造依存をやめ、公式の安定構造に切り替え。

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_official_racelist_html(jcd: int, rno: int, date_str: str) -> Optional[str]:
    jcd_s = f"{jcd:02d}"
    return get(f"{BOAT}/racelist?rno={rno}&jcd={jcd_s}&hd={date_str}")


def _parse_official_racelist(html: str) -> List[Racer]:
    """boatrace.jp公式 racelist ページから6艇分のRacerを抽出。"""
    soup = BeautifulSoup(html, "html.parser")

    # 出走表テーブルを特定: 「ボートレーサー」と「全国」と「当地」と「モーター」を含むテーブル
    target = None
    for tbl in soup.find_all("table"):
        head = tbl.get_text(" ", strip=True)
        if all(k in head for k in ["ボートレーサー", "全国", "当地", "モーター"]):
            target = tbl
            break
    if not target:
        return []

    # 各艇の主行を抽出: 「profile?toban=N」リンクを持つ行のみ主行
    # (今節成績の進入コース/ST/着順の行には profileリンクが無いので確実に除外できる)
    racers: List[Racer] = []
    rows = target.find_all("tr")

    lane_map = {"１": 1, "２": 2, "３": 3, "４": 4, "５": 5, "６": 6,
                "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}

    main_rows: List[Tuple[int, "BeautifulSoup"]] = []
    seen_lanes = set()
    for tr in rows:
        # 主行は profile?toban=... へのリンクを必ず持つ
        a_test = tr.find("a", href=re.compile(r"profile\?toban=\d+"))
        if not a_test:
            continue
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        first_text = cells[0].get_text(strip=True)
        if first_text in lane_map and lane_map[first_text] not in seen_lanes:
            lane = lane_map[first_text]
            main_rows.append((lane, tr))
            seen_lanes.add(lane)
            if len(main_rows) >= 6:
                break

    if len(main_rows) < 6:
        return []

    main_rows.sort(key=lambda x: x[0])

    # 今節成績の3行(進入コース/ST/着順)を主行の直後から抽出する用に、
    # テーブル全体のtr配列上での主行のインデックスを取得しておく
    all_trs = list(target.find_all("tr"))
    main_tr_indices: Dict[int, int] = {}
    for lane, tr in main_rows:
        try:
            main_tr_indices[lane] = all_trs.index(tr)
        except ValueError:
            pass

    for lane, tr in main_rows:
        full_text = tr.get_text(" ", strip=True)
        full_text = re.sub(r"\s+", " ", full_text)

        cls_ = ""
        weight: Optional[float] = None

        cls_match = re.search(r"/\s*(A1|A2|B1|B2)\b", full_text)
        if cls_match:
            cls_ = cls_match.group(1)

        wt_match = re.search(r"(\d+\.\d+)\s*kg", full_text)
        if wt_match:
            weight = float(wt_match.group(1))

        a_tag = tr.find("a", href=re.compile(r"profile\?toban=\d+"))
        if a_tag:
            name = a_tag.get_text(strip=True).replace(" ", "").replace("　", "")
        else:
            name = f"選手{lane}"

        fl_match = re.search(r"F\s*(\d+)\s+L\s*(\d+)", full_text)
        f_count = int(fl_match.group(1)) if fl_match else 0

        avg_st: Optional[float] = None
        win_rate: Optional[float] = None
        motor_2rate: Optional[float] = None

        if fl_match:
            tail = full_text[fl_match.end():]
            nums = re.findall(r"-?\d+\.\d+|\d+", tail)
            try:
                avg_st = float(nums[0]) if "." in nums[0] else None
            except (IndexError, ValueError):
                pass
            try:
                win_rate = float(nums[1])
            except (IndexError, ValueError):
                pass
            try:
                m2v = float(nums[8])
                motor_2rate = m2v / 100.0 if m2v > 1.0 else m2v
            except (IndexError, ValueError):
                pass

        # 今節成績集計: 主行の直後に最大3行(進入コース/ST/着順)
        settle_st: Optional[float] = None
        settle_avg_rank: Optional[float] = None
        idx = main_tr_indices.get(lane)
        if idx is not None and idx + 3 < len(all_trs):
            cs_tr = all_trs[idx + 1]
            st_tr = all_trs[idx + 2]
            fn_tr = all_trs[idx + 3]

            def cells_text(t) -> List[str]:
                return [td.get_text(strip=True) for td in t.find_all(["td", "th"])]

            st_cells = cells_text(st_tr)
            fn_cells = cells_text(fn_tr)

            # ST行: ".12" 形式 (前ピリオド)、または「F.03」のようなF表記もある
            st_vals: List[float] = []
            for c in st_cells:
                # 「F.03」「L.05」「K」等の特殊表記は集計から除外
                if re.search(r"[FLK失]", c):
                    continue
                # ".12" → 0.12 / "0.12" → 0.12
                if re.fullmatch(r"\.\d+", c):
                    try:
                        st_vals.append(float("0" + c))
                    except ValueError:
                        pass
                elif re.fullmatch(r"0\.\d+", c):
                    try:
                        st_vals.append(float(c))
                    except ValueError:
                        pass

            if st_vals:
                settle_st = round(sum(st_vals) / len(st_vals), 3)

            # v17.2: 着順行から平均順位を算出
            # 公式サイトは全角数字(１〜６)で表示されるため変換
            # 失格/欠場/フライング(F0,F1,F2)は集計から除外
            zen_to_han = str.maketrans("１２３４５６", "123456")
            ranks: List[int] = []
            for c in fn_cells:
                # 全角→半角変換
                c_norm = c.translate(zen_to_han)
                # 1〜6の単一数字のみを着順として採用
                if re.fullmatch(r"[1-6]", c_norm):
                    ranks.append(int(c_norm))
                # それ以外(F/L/失/妨/転/落/エ等)は除外

            if ranks:
                settle_avg_rank = round(sum(ranks) / len(ranks), 2)

        racers.append(Racer(
            name=name, cls=cls_,
            win_rate=win_rate, avg_st=avg_st,
            settle_st=settle_st,
            settle_avg_rank=settle_avg_rank,
            motor_2rate=motor_2rate,
            f_count=f_count,
            weight=weight,
            course5_avg_st=avg_st,
        ))

    return racers


def fetch_race(jcd: int, rno: int, date_str: str) -> List[Racer]:
    """単一レースの出走表を取得。失敗時は空リスト。"""
    html = _fetch_official_racelist_html(jcd, rno, date_str)
    if not html:
        return []
    return _parse_official_racelist(html)


def fetch_racelist(jcd: int, date_str: str) -> Dict[int, List[Racer]]:
    """指定日・指定場の全12Rを公式から取得。
    Tab2バックテストで使われるため、12回HTTPするが、各レース独立で取得可能。
    """
    out: Dict[int, List[Racer]] = {}
    for rno in range(1, 13):
        racers = fetch_race(jcd, rno, date_str)
        if len(racers) == 6:
            out[rno] = racers
    return out





# ============================================================
# kyotei.sakura.ne.jp: 1日分の払戻を一括取得
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kyotei_day(date_str: str) -> Dict[Tuple[int, int], int]:
    html = get(f"{KYOTEI}/kako-{date_str}.html")
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    payouts: Dict[Tuple[int, int], int] = {}
    # v16.7: 正規表現に日付を直接埋め込み、他日のレースリンクを誤検出しないように
    pat = re.compile(rf'info-{re.escape(date_str)}-(\d+)-(\d+)\.html')
    for a in soup.find_all("a", href=True):
        if "race.kyotei.club" not in a["href"]:
            continue
        m = pat.search(a["href"])
        if not m:
            continue
        jcd = int(m.group(1))
        rno = int(m.group(2))
        if jcd not in JCD_NAME or rno not in range(1, 13):
            continue
        if (jcd, rno) in payouts:
            continue
        tr = a.find_parent("tr")
        if not tr:
            continue
        for a2 in tr.find_all("a", href=True):
            if "info.kyotei.fun" not in a2["href"]:
                continue
            txt = a2.get_text(strip=True).replace(",", "")
            if re.fullmatch(r"\d+", txt) and int(txt) > 0:
                payouts[(jcd, rno)] = int(txt)
                break
    return payouts


def kyotei_venues(date_str: str) -> List[int]:
    """kyotei.sakura.ne.jpのkakoページから、その日に実際に開催された場のjcd一覧を返す。
    
    v16.6 → v16.7 の修正:
    - 正規表現に日付(date_str)を直接埋め込み、他日のリンクを誤検出しないようにした
    - race.kyotei.club ドメインのリンクに限定 (他のリンクと混同しない)
    
    例: 2026/04/25のページに前日(2026/04/24)江戸川の結果リンクがあっても、
        date_str=20260425 と一致しないので除外される。
    """
    html = get(f"{KYOTEI}/kako-{date_str}.html")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    # 当日(date_str)のレース結果リンクのみマッチ
    pat = re.compile(rf'info-{re.escape(date_str)}-(\d+)-\d+\.html')
    jcds = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # race.kyotei.club ドメインに限定
        if "race.kyotei.club" not in href:
            continue
        m = pat.search(href)
        if m:
            jcd = int(m.group(1))
            if jcd in JCD_NAME:
                jcds.add(jcd)
    return sorted(jcds)


# ============================================================
# boatrace.jp: 個別レース着順・払戻
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_result(date_str: str, jcd: int, rno: int) -> Optional[Dict]:
    jcd_s = f"{jcd:02d}"
    html  = get(f"{BOAT}/raceresult?rno={rno}&jcd={jcd_s}&hd={date_str}")
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if "まだ結果がありません" in text or "発売中" in text:
        return None

    finish: List[int] = []
    for tbody in soup.select("div.table1 table tbody"):
        tds = [td.get_text(strip=True) for td in tbody.find_all("td")]
        if len(tds) >= 2 and re.fullmatch(r"\d+", tds[0]) and re.fullmatch(r"[1-6]", tds[1]):
            finish.append(int(tds[1]))
        if len(finish) >= 6:
            break
    if len(finish) < 3:
        return None

    combo, pay = None, 0
    m = re.search(r"3連単.*?([1-6])\s*[-ー‐]\s*([1-6])\s*[-ー‐]\s*([1-6]).*?¥?\s*([\d,]+)", text)
    if m:
        combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        pay   = int(m.group(4).replace(",", ""))

    kim = None
    km = re.search(r"(逃げ|まくり差し|まくり|差し|抜き|恵まれ)", text)
    if km:
        kim = km.group(1)

    return {"finish": finish, "combo": combo, "payout": pay, "kimarite": kim}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds_3t(date_str: str, jcd: int, rno: int) -> Dict[str, float]:
    """boatrace.jp から3連単オッズを取得。{'1-2-3': 5.6, ...} 形式で返す。
    
    boatrace.jpのテーブル構造:
      - 6つの1着グループ(列)が横並び
      - 各行に各グループのオッズが1つずつ
      - 1着グループ内で(2着,3着)の順序は: 2着=残りレーン昇順、各2着内で3着=残りレーン昇順
      - DOM順(行→列): 行r, 列c のセルは「1着=c+1」グループの第r位の組合せ
      - Standard順: 1着=1の全20通り → 1着=2の全20通り → ... の順
    """
    jcd_s = f"{jcd:02d}"
    html = get(f"{BOAT}/odds3t?rno={rno}&jcd={jcd_s}&hd={date_str}")
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if "発売前" in text or "まだ発売されていません" in text:
        return {}

    # 標準組合せ順 (1着→2着→3着、それぞれ昇順)
    combo_order: List[str] = []
    for a in range(1, 7):
        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c == a or c == b:
                    continue
                combo_order.append(f"{a}-{b}-{c}")

    # オッズらしいセル(decimal, または2桁以上の整数)を集める
    def collect_odds_cells(tbl) -> List[float]:
        out: List[float] = []
        for td in tbl.find_all("td"):
            txt = td.get_text(strip=True).replace(",", "").replace(" ", "")
            if not txt:
                continue
            # 形式1: X.X (オッズ典型)
            if re.fullmatch(r"\d+\.\d+", txt):
                out.append(float(txt))
            # 形式2: 2桁以上の整数 (高オッズ "2187" 等)
            elif re.fullmatch(r"\d+", txt) and len(txt) >= 2:
                out.append(float(txt))
            # 単一数字(1-6)はレーン番号なので無視
        return out

    target_cells: List[float] = []
    for tbl in soup.find_all("table"):
        cells = collect_odds_cells(tbl)
        if len(cells) == 120:
            target_cells = cells
            break

    # フォールバック: 120ぴったり無い場合、最大数のテーブルを採用
    if not target_cells:
        best_cells: List[float] = []
        for tbl in soup.find_all("table"):
            cells = collect_odds_cells(tbl)
            if len(cells) > len(best_cells) and len(cells) >= 60:
                best_cells = cells
        if len(best_cells) == 120:
            target_cells = best_cells
        else:
            return {}

    # DOM順 → Standard順 マッピング
    # DOM index = row * 6 + col  (row 0..19, col 0..5)
    # Standard index = col * 20 + row  (col=1着-1, row=1着内位置)
    odds_dict: Dict[str, float] = {}
    for dom_idx, val in enumerate(target_cells):
        row = dom_idx // 6
        col = dom_idx % 6
        std_idx = col * 20 + row
        if 0 <= std_idx < 120:
            odds_dict[combo_order[std_idx]] = val

    return odds_dict



# ============================================================
# boatrace.jp公式: 直前情報(展示タイム)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_exhibit_times(date_str: str, jcd: int, rno: int) -> List[Optional[float]]:
    """boatrace.jp の直前情報ページから展示タイムを取得。
    返り値は枠順(1〜6)に対応するList[Optional[float]]。
    取得失敗時や未発表時は[None]*6を返す。
    
    展示タイムは小さいほど速い(典型的に6.6〜6.9秒)。
    """
    jcd_s = f"{jcd:02d}"
    html = get(f"{BOAT}/beforeinfo?rno={rno}&jcd={jcd_s}&hd={date_str}")
    if not html:
        return [None] * 6
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if "情報がありません" in text or "発表前" in text:
        return [None] * 6

    # 展示タイムは選手テーブルの各枠の行に含まれる。
    # 構造: 体重(52.0kg)の次に展示タイム(6.67) が続くパターン
    # 全体から「6艇分」の行を順に走査する
    times: List[Optional[float]] = [None] * 6

    # ヘッダー行に "展示" と "タイム" を含むテーブルを探す
    target_table = None
    for tbl in soup.find_all("table"):
        head_text = tbl.get_text(" ", strip=True)
        if "展示" in head_text and "タイム" in head_text and "ボートレーサー" in head_text:
            target_table = tbl
            break
    if not target_table:
        return times

    # 各枠の行を抽出。各枠は複数tr (進入/着順等の行) にまたがるため、
    # 「枠番号(1-6) で始まる tr」を主行として識別
    rows = target_table.find_all("tr")
    lane_idx = 0  # 0..5
    for tr in rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        first = cells[0].get_text(strip=True)
        # 枠番号セル (1〜6) で始まる行が主行
        if first in ("1", "2", "3", "4", "5", "6") and lane_idx < 6:
            lane = int(first) - 1
            # 主行のテキストから X.XX 形式の数値を抽出
            row_text = tr.get_text(" ", strip=True)
            # 体重: "XX.Xkg" → 除外したい
            # 展示タイム: 6.50〜7.50 程度の範囲の数値
            for m in re.finditer(r"\b(\d+\.\d+)\b", row_text):
                v = float(m.group(1))
                # 体重(40〜60kg台)とチルト(-0.5〜+3.0)を除外し、
                # 展示タイムらしい範囲(5.50〜8.50)の値を採用
                if 5.50 <= v <= 8.50:
                    times[lane] = v
                    break
            lane_idx = lane + 1

    return times


def assign_exhibit_ranks(times: List[Optional[float]]) -> List[Optional[int]]:
    """展示タイム配列から各枠の順位(1〜6)を返す。Noneは順位もNone。"""
    valid = [(i, t) for i, t in enumerate(times) if t is not None]
    if len(valid) < 2:
        return [None] * 6
    # 小さいほど速い → 昇順
    sorted_pairs = sorted(valid, key=lambda x: x[1])
    ranks: List[Optional[int]] = [None] * 6
    for rank, (i, _) in enumerate(sorted_pairs, start=1):
        ranks[i] = rank
    return ranks



# ============================================================
# 場傾向の表示ヘルパー
# ============================================================
def render_venue_summary(venue: str):
    if venue not in COURSE_WIN_RATE:
        return
    st.markdown(f"### 🗺️ 場の傾向 — {venue} {venue_tendency_label(venue)}")
    st.caption("kyoteibiyori集計 2023/03/12〜2024/03/12")

    nat = COURSE_WIN_RATE["全国"]
    ven = COURSE_WIN_RATE[venue]
    rows = []
    for i in range(6):
        lane = i + 1
        diff = ven[i] - nat[i]
        attack = venue_attack_bonus(venue, lane)
        rows.append({
            "C": lane,
            "基礎": f"{COURSE_BASE_POINTS[lane]:+.1f}",
            "場1着率": f"{ven[i]:.1f}%",
            "場×C": f"{venue_course_bonus(venue, lane):+.2f}",
            "場×攻": f"{attack:+.2f}" if lane >= 2 else "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# UI
# ============================================================
today = datetime.now().date()
tab1, tab2 = st.tabs(["🔍 1レース解析", "📊 期間バックテスト/当日スキャン"])


# ──────────────────────────────────────────────────────────
# TAB 1
# ──────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        t1_date = st.date_input(
            "日付", value=today,
            min_value=datetime(2020,1,1).date(),
            max_value=today + timedelta(days=1),
            key="t1_date",
        )
    with c2:
        st.write("")
        st.write("")
        if st.button("🔄", key="t1_reload"):
            st.cache_data.clear()

    dstr = t1_date.strftime("%Y%m%d")

    with st.spinner("開催場を取得中..."):
        vlist = venues_for_date(t1_date)

    if not vlist:
        st.warning("開催場が見つかりません。")
    else:
        vname = st.selectbox("開催場", [n for _, n in vlist], key="t1_venue")
        jcd   = NAME_JCD[vname]
        rno   = st.selectbox("レース", list(range(1, 13)),
                             format_func=lambda r: f"{r}R", key="t1_rno")

        # NEW: 戦略プリセット
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            t1_strategy = st.radio(
                "戦略", ["safe", "standard", "wide"],
                index=1,
                format_func=strategy_label,
                horizontal=True,
                key="t1_strategy",
            )
        with sc2:
            t1_min_odds = st.slider(
                "最低オッズ (0=無効)",
                min_value=0.0, max_value=30.0, value=0.0, step=1.0,
                key="t1_min_odds",
                help="指定オッズ未満の買い目を除外。低オッズ乱撃を防ぎ回収率改善。",
            )

        with st.expander("🗺️ 場の傾向を見る", expanded=False):
            render_venue_summary(vname)

        if st.button("🎯 解析する", type="primary", use_container_width=True, key="t1_run"):
            with st.spinner("選手データ取得中..."):
                all_r = fetch_racelist(jcd, dstr)

            racers = all_r.get(rno)
            if not racers or len(racers) < 6:
                st.error("選手データを取得できませんでした。時間をおいて再試行してください。")
            else:
                # v16.9: 展示タイムを取得して順位を付与
                exhibit_times: List[Optional[float]] = [None] * 6
                exhibit_loaded = False
                if t1_date <= today + timedelta(days=1):
                    with st.spinner("展示タイム取得中..."):
                        exhibit_times = fetch_exhibit_times(dstr, jcd, rno)
                exhibit_ranks = assign_exhibit_ranks(exhibit_times)
                exhibit_loaded = any(t is not None for t in exhibit_times)
                for i, r in enumerate(racers):
                    r.exhibit_rank = exhibit_ranks[i]

                ranked = rank_all(racers, vname)

                # オッズ取得(未発走でも発売中なら取得可)
                odds_map: Dict[str, float] = {}
                if t1_date >= today:  # 当日/明日のみ取得試行
                    with st.spinner("オッズ取得中..."):
                        odds_map = fetch_odds_3t(dstr, jcd, rno)

                bets = make_bets(ranked, strategy=t1_strategy,
                                 odds_map=odds_map if odds_map else None,
                                 min_odds=t1_min_odds)

                res = None
                if t1_date <= today:
                    with st.spinner("レース結果取得中..."):
                        res = fetch_result(dstr, jcd, rno)

                st.markdown(f"### {vname} {rno}R {venue_tendency_label(vname)}")
                st.caption(f"戦略: **{strategy_label(t1_strategy)}**"
                           + (f" / オッズ≥{t1_min_odds:.0f}倍" if t1_min_odds > 0 else "")
                           + (" / オッズ取得成功" if odds_map else "")
                           + (" / 展示適用" if exhibit_loaded else " / 展示未発表"))

                # スコア差(信頼度)表示 — 1-2位 / 2-3位 / 3-4位
                if len(ranked) >= 4:
                    m12 = ranked[0]["score"] - ranked[1]["score"]
                    m23 = ranked[1]["score"] - ranked[2]["score"]
                    m34 = ranked[2]["score"] - ranked[3]["score"]
                    if m12 >= 1.5:    conf = "🟢 高信頼"
                    elif m12 >= 0.8:  conf = "🟡 中信頼"
                    else:             conf = "🔴 低信頼"
                    st.caption(
                        f"スコア差: 1-2={m12:+.2f} / 2-3={m23:+.2f} / 3-4={m34:+.2f} {conf}"
                    )

                df_rows = []
                for rk, x in enumerate(ranked, 1):
                    r = x["racer"]
                    bd = x["breakdown"]
                    win_r = f"{r.win_rate:.2f}" if r.win_rate is not None else "-"
                    c_st  = f"{r.avg_st:.2f}" if r.avg_st is not None else "-"
                    m2_r  = f"{r.motor_2rate*100:.0f}" if r.motor_2rate is not None else "-"
                    # v17.2: 節2率 → 節平順
                    sar_r = f"{r.settle_avg_rank:.1f}" if r.settle_avg_rank is not None else "-"
                    # 今節ST
                    sst_r = f"{r.settle_st:.2f}" if r.settle_st is not None else "-"
                    # 展示タイム & 順位
                    lane_idx = x["lane"] - 1
                    ex_time = exhibit_times[lane_idx]
                    ex_rank = r.exhibit_rank
                    ex_str = (f"{ex_time:.2f}({ex_rank})" if ex_time is not None and ex_rank
                              else "-")
                    venue_total = bd["コース基礎"] + bd["場×コース"] + bd["場×攻め"]
                    df_rows.append({
                        "順位": rk,
                        "艇":   x["lane"],
                        "名前": r.name,
                        "級":   r.cls or "-",
                        "勝率": win_r,
                        "ST":   c_st,
                        "M2率": m2_r,
                        "節平順": sar_r,
                        "節ST": sst_r,
                        "F":    r.f_count or 0,
                        "展示": ex_str,
                        "基礎+場": f"{venue_total:+.2f}",
                        "スコア": f"{x['score']:+.2f}",
                    })
                st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

                with st.expander("📋 スコア内訳(上位3艇)"):
                    br_rows = []
                    for rk, x in enumerate(ranked[:3], 1):
                        row = {"順位": rk, "艇": x["lane"], "名前": x["racer"].name}
                        for k, v in x["breakdown"].items():
                            if k == "合計":
                                continue
                            row[k] = f"{v:+.2f}"
                        row["合計"] = f"{x['score']:+.2f}"
                        br_rows.append(row)
                    st.dataframe(pd.DataFrame(br_rows), use_container_width=True, hide_index=True)

                if bets:
                    st.subheader(f"🎯 推奨買い目 ({strategy_label(t1_strategy)} / {len(bets)}点)")
                    st.markdown(
                        " / ".join(
                            f"**{i+1}位** {ranked[i]['lane']}号艇({ranked[i]['racer'].name})"
                            for i in range(min(4, len(ranked)))
                        )
                    )

                    # オッズ表示付きの買い目テーブル
                    if odds_map:
                        bet_rows = []
                        total_inv = len(bets) * 100
                        odds_values = [odds_map.get(b, 0) for b in bets]
                        for b, o in zip(bets, odds_values):
                            exp_pay = o * 100 if o > 0 else 0
                            bet_rows.append({
                                "買い目": b,
                                "オッズ": f"{o:.1f}倍" if o > 0 else "-",
                                "的中時回収": f"¥{int(exp_pay):,}" if o > 0 else "-",
                                "回収率": f"{exp_pay/total_inv*100:.0f}%" if o > 0 else "-",
                            })
                        st.dataframe(pd.DataFrame(bet_rows),
                                     use_container_width=True, hide_index=True)
                        # 平均オッズ
                        valid_odds = [o for o in odds_values if o > 0]
                        if valid_odds:
                            avg_o = sum(valid_odds) / len(valid_odds)
                            min_o = min(valid_odds)
                            max_o = max(valid_odds)
                            st.caption(
                                f"💡 オッズ範囲: {min_o:.1f}〜{max_o:.1f}倍 / 平均{avg_o:.1f}倍 / "
                                f"投資¥{total_inv} / 1点でも的中すれば¥{int(min_o*100):,}回収"
                            )
                    else:
                        st.code("\n".join(bets))
                        st.caption("オッズ未取得 (未発売または解析失敗)")
                elif t1_min_odds > 0:
                    st.warning(f"⚠️ オッズ≥{t1_min_odds:.0f}倍の買い目がありません。"
                               "戦略変更またはオッズ閾値を下げてください。")

                if res:
                    st.markdown("---")
                    st.subheader("🏁 レース結果")
                    c1r, c2r = st.columns(2)
                    c1r.markdown(f"**着順**: {'-'.join(str(n) for n in res['finish'][:3])}")
                    if res.get("kimarite"):
                        c2r.markdown(f"**決まり手**: {res['kimarite']}")
                    hit    = res["combo"] in bets if res["combo"] else False
                    inv_yen = len(bets) * 100
                    payout = res["payout"] if hit else 0
                    rr     = (payout / inv_yen * 100) if inv_yen > 0 else 0
                    profit = payout - inv_yen
                    if res.get("combo"):
                        st.metric("3連単 払戻", res["combo"], f"¥{res['payout']:,}")
                    st.markdown(f"### 💰 買い目収支（{len(bets)}点=¥{inv_yen}）")
                    ca, cb, cc = st.columns(3)
                    ca.metric("投資", f"¥{inv_yen}")
                    cb.metric("回収", f"¥{payout:,}")
                    cc.metric("回収率", f"{rr:.0f}%", f"{profit:+,}円",
                              delta_color="normal" if rr >= 100 else "inverse")
                    if hit:
                        st.success(f"✅ 的中: `{res['combo']}` → ¥{payout:,}")
                    else:
                        st.info("買い目不的中")
                else:
                    st.caption("🕓 結果未確定（未発走または取得失敗）")

                jcd_s = f"{jcd:02d}"
                st.markdown("---")
                st.markdown(
                    f"🔗 [uchisankaku]({UCHI}/racelist.php?jcode={jcd}&date={dstr})  "
                    f"[出走表]({BOAT}/racelist?rno={rno}&jcd={jcd_s}&hd={dstr})  "
                    f"[オッズ]({BOAT}/odds3t?rno={rno}&jcd={jcd_s}&hd={dstr})  "
                    f"[結果]({BOAT}/raceresult?rno={rno}&jcd={jcd_s}&hd={dstr})"
                )


# ──────────────────────────────────────────────────────────
# TAB 2: 期間バックテスト + 当日スキャン
# ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 期間バックテスト / 当日スキャン")
    st.caption("予想1位が1号艇のレースを抽出。過去日は結果取得、当日は予想のみ。")

    bc1, bc2 = st.columns(2)
    with bc1:
        bt_s = st.date_input(
            "開始日", value=today - timedelta(days=3),
            min_value=datetime(2020,1,1).date(),
            max_value=today,
            key="bt_s",
        )
    with bc2:
        bt_e = st.date_input(
            "終了日", value=today,
            min_value=datetime(2020,1,1).date(),
            max_value=today,
            key="bt_e",
        )

    # v17.3: 開催場を絞り込めるマルチセレクト
    # 公式パーサー(v17系)は1日24場×12R=最大288リクエストとなり時間がかかるため、
    # 場を絞ることで処理時間を大幅に短縮できる
    all_venue_names = [JCD_NAME[j] for j in sorted(JCD_NAME.keys())]
    bt_venues = st.multiselect(
        "対象場 (空=全場)",
        options=all_venue_names,
        default=[],
        key="bt_venues",
        help="選択した場のみ解析。空のままだと全24場が対象(時間がかかる)。"
             "1場あたり最大12レース×処理時間。3〜5場に絞ると現実的。",
    )
    bt_target_jcds: Optional[set] = None
    if bt_venues:
        bt_target_jcds = {NAME_JCD[v] for v in bt_venues if v in NAME_JCD}
        st.caption(f"🎯 対象 {len(bt_target_jcds)} 場 (推定リクエスト数: "
                   f"{len(bt_target_jcds) * 12} /日)")
    else:
        st.caption("⚠️ 全場対象 (推定最大288リクエスト/日)。"
                   "時間がかかる場合は対象場を絞ってください")

    # NEW: 戦略プリセット
    bt_strategy = st.radio(
        "戦略", ["safe", "standard", "wide"],
        index=1,
        format_func=strategy_label,
        horizontal=True,
        key="bt_strategy",
    )

    # NEW: 品質ゲート
    with st.expander("🔧 品質フィルター (デフォルト推奨)", expanded=True):
        qc1, qc2 = st.columns(2)
        with qc1:
            bt_skip_b2 = st.checkbox(
                "B2の1号艇を除外", value=True, key="bt_skip_b2",
                help="B2選手は1着率が大幅に低いため除外推奨",
            )
            bt_skip_hard = st.checkbox(
                "戸田/江戸川/平和島を除外", value=False, key="bt_skip_hard",
                help="荒れ水面は1号艇信頼度が低下",
            )
        with qc2:
            bt_min_winrate = st.slider(
                "1号艇勝率の下限", 0.0, 8.0, 5.0, 0.5, key="bt_min_wr",
                help="この勝率未満の1号艇は除外",
            )
        bt_use_exhibit = st.checkbox(
            "展示タイムを反映 (精度向上・処理時間増)", value=False, key="bt_use_exhibit",
            help="ONにすると各レースで展示タイムを公式から取得しスコアに反映。"
                 "Nレース×1リクエスト追加で時間がかかる。当日朝など展示前は意味なし。",
        )

    # スコア差フィルター (1-2位 / 2-3位 / 3-4位)
    st.markdown("**スコア差フィルター** (各順位の差で信頼度を厳格化)")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        min_margin_12 = st.slider(
            "1位-2位 ≥",
            min_value=0.0, max_value=3.0, value=0.8, step=0.1, key="bt_margin12",
            help="1着候補の優位性 (0.8=中信頼/1.5=高信頼)",
        )
    with mc2:
        min_margin_23 = st.slider(
            "2位-3位 ≥",
            min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="bt_margin23",
            help="2着候補が3着候補より明確に上か (買い目精度に直結)",
        )
    with mc3:
        min_margin_34 = st.slider(
            "3位-4位 ≥",
            min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="bt_margin34",
            help="3着候補が4着候補より明確に上か (拡張9点で重要)",
        )

    if bt_s > bt_e:
        st.warning("開始日 ≤ 終了日 にしてください。")
    else:
        n_days = (bt_e - bt_s).days + 1
        filters_desc = [
            f"戦略={strategy_label(bt_strategy)}",
            f"差12≥{min_margin_12:.1f}",
        ]
        if min_margin_23 > 0:
            filters_desc.append(f"差23≥{min_margin_23:.1f}")
        if min_margin_34 > 0:
            filters_desc.append(f"差34≥{min_margin_34:.1f}")
        filters_desc.append(f"勝率≥{bt_min_winrate:.1f}")
        if bt_skip_b2: filters_desc.append("B2除外")
        if bt_skip_hard: filters_desc.append("荒れ水面除外")
        if bt_use_exhibit: filters_desc.append("展示反映")
        if bt_target_jcds:
            filters_desc.append(f"場={len(bt_target_jcds)}場")
        st.caption(f"対象: {bt_s} 〜 {bt_e}（{n_days}日間） / " + " / ".join(filters_desc))

        if st.button("🔍 1号艇1位を検索", type="primary",
                     use_container_width=True, key="bt_run"):
            days    = [bt_s + timedelta(days=i) for i in range(n_days)]
            prog    = st.progress(0.0)
            status  = st.empty()
            matches: List[Dict] = []
            hard_venues = {"戸田", "江戸川", "平和島"}

            for idx, day in enumerate(days):
                dstr_bt = day.strftime("%Y%m%d")
                is_past = day < today
                prog.progress((idx + 1) / n_days,
                              text=f"[{idx+1}/{n_days}] {dstr_bt} 処理中"
                                   f"{'(当日)' if not is_past else ''}...")

                if is_past:
                    status.caption(f"📡 {dstr_bt} — kyotei 払戻取得中...")
                    payouts = fetch_kyotei_day(dstr_bt)
                    if not payouts:
                        continue
                    open_jcds = kyotei_venues(dstr_bt) or sorted({jcd for jcd, _ in payouts.keys()})
                else:
                    status.caption(f"📡 {dstr_bt} — 当日開催場取得中...")
                    payouts = {}
                    vtoday = venues_for_date(day)
                    open_jcds = [j for j, _ in vtoday]
                    if not open_jcds:
                        continue

                for jcd_bt in open_jcds:
                    venue_bt = JCD_NAME.get(jcd_bt, "")
                    if not venue_bt:
                        continue
                    # v17.3: ユーザー指定の対象場フィルター
                    if bt_target_jcds is not None and jcd_bt not in bt_target_jcds:
                        continue
                    # 荒れ水面スキップ
                    if bt_skip_hard and venue_bt in hard_venues:
                        continue
                    status.caption(f"📡 {dstr_bt} {venue_bt} — 選手データ取得中...")
                    races = fetch_racelist(jcd_bt, dstr_bt)
                    if not races:
                        continue

                    for rno_bt, racers_bt in sorted(races.items()):
                        if len(racers_bt) < 6:
                            continue

                        # v16.9: 展示タイム反映 (チェックボックスON時のみ)
                        if bt_use_exhibit:
                            ex_times = fetch_exhibit_times(dstr_bt, jcd_bt, rno_bt)
                            ex_ranks = assign_exhibit_ranks(ex_times)
                            for i, r_ in enumerate(racers_bt):
                                r_.exhibit_rank = ex_ranks[i]

                        ranked_bt = rank_all(racers_bt, venue_bt)
                        if ranked_bt[0]["lane"] != 1:
                            continue

                        # 品質ゲート: 1号艇の選手を取得
                        ichi = racers_bt[0]
                        if bt_skip_b2 and ichi.cls == "B2":
                            continue
                        if ichi.win_rate is not None and ichi.win_rate < bt_min_winrate:
                            continue

                        # スコア差フィルター (1-2 / 2-3 / 3-4)
                        margin_12 = ranked_bt[0]["score"] - ranked_bt[1]["score"]
                        margin_23 = ranked_bt[1]["score"] - ranked_bt[2]["score"]
                        margin_34 = ranked_bt[2]["score"] - ranked_bt[3]["score"]
                        if margin_12 < min_margin_12:
                            continue
                        if margin_23 < min_margin_23:
                            continue
                        if margin_34 < min_margin_34:
                            continue

                        bets_bt   = make_bets(ranked_bt, strategy=bt_strategy)
                        top_score = ranked_bt[0]["score"]
                        inv_bt    = len(bets_bt) * 100

                        # 当日: 未発走として記録
                        if not is_past:
                            matches.append({
                                "日付": dstr_bt, "場": venue_bt, "R": rno_bt,
                                "スコア": top_score,
                                "差12": margin_12, "差23": margin_23, "差34": margin_34,
                                "_bets": bets_bt, "_inv": inv_bt,
                                "結果": "未発走", "払戻": 0,
                                "_hit": None, "_payout": 0, "_status": "pending",
                            })
                            continue

                        pay_kyotei = payouts.get((jcd_bt, rno_bt))
                        if pay_kyotei is None:
                            matches.append({
                                "日付": dstr_bt, "場": venue_bt, "R": rno_bt,
                                "スコア": top_score,
                                "差12": margin_12, "差23": margin_23, "差34": margin_34,
                                "_bets": bets_bt, "_inv": inv_bt,
                                "結果": "未確定", "払戻": 0,
                                "_hit": None, "_payout": 0, "_status": "unresolved",
                            })
                            continue

                        status.caption(
                            f"📡 {dstr_bt} {venue_bt} {rno_bt}R — 着順確認...")
                        res_bt = fetch_result(dstr_bt, jcd_bt, rno_bt)
                        if res_bt and res_bt["combo"]:
                            combo_bt  = res_bt["combo"]
                            hit_bt    = combo_bt in bets_bt
                            payout_bt = pay_kyotei if hit_bt else 0
                            status_bt = "resolved"
                        else:
                            combo_bt  = "取得失敗"
                            hit_bt    = None
                            payout_bt = 0
                            status_bt = "unresolved"

                        matches.append({
                            "日付": dstr_bt, "場": venue_bt, "R": rno_bt,
                            "スコア": top_score,
                            "差12": margin_12, "差23": margin_23, "差34": margin_34,
                            "_bets": bets_bt, "_inv": inv_bt,
                            "結果": combo_bt, "払戻": pay_kyotei,
                            "_hit": hit_bt, "_payout": payout_bt,
                            "_status": status_bt,
                        })

            prog.empty()
            status.empty()
            st.session_state["bt_matches"] = matches

        if "bt_matches" in st.session_state:
            M = st.session_state["bt_matches"]
            if not M:
                st.warning("対象期間に条件を満たすレースが見つかりませんでした。"
                           "フィルターを緩めてみてください。")
            else:
                n_tot     = len(M)
                n_pending = sum(1 for m in M if m["_status"] == "pending")
                n_unres   = sum(1 for m in M if m["_status"] == "unresolved")
                n_resv    = sum(1 for m in M if m["_status"] == "resolved")
                hits      = [m for m in M if m["_hit"] is True]
                n_hit     = len(hits)
                inv       = sum(m["_inv"] for m in M if m["_status"] == "resolved")
                ret       = sum(m["_payout"] for m in M if m["_status"] == "resolved")
                rr        = round(ret / inv * 100, 1) if inv > 0 else 0
                hr        = round(n_hit / n_resv * 100, 1) if n_resv > 0 else 0

                st.success(
                    f"✅ {n_tot}件 抽出 "
                    f"(結果確定: {n_resv} / 未発走: {n_pending} / 未確定: {n_unres})"
                )

                st.markdown("### 📊 集計")
                ca, cb, cc, cd = st.columns(4)
                ca.metric("対象",    f"{n_tot}件")
                cb.metric("的中",    f"{n_hit}/{n_resv}", f"{hr}%" if n_resv else "-")
                cc.metric("回収率",  f"{rr}%" if n_resv else "-",
                          f"{ret-inv:+,}円" if n_resv else None,
                          delta_color="normal" if rr >= 100 else "inverse")
                cd.metric("投資/回収", f"¥{inv:,} / ¥{ret:,}")

                # 場別内訳
                if n_resv > 0:
                    st.markdown("### 🗺️ 場別内訳（結果確定分）")
                    venue_stats: Dict[str, Dict] = {}
                    for m in M:
                        if m["_status"] != "resolved":
                            continue
                        v = m["場"]
                        s = venue_stats.setdefault(v, {"n": 0, "hit": 0, "inv": 0, "pay": 0})
                        s["n"]   += 1
                        s["hit"] += 1 if m["_hit"] else 0
                        s["inv"] += m["_inv"]
                        s["pay"] += m["_payout"]
                    vrows = []
                    for v, s in sorted(venue_stats.items(), key=lambda kv: -kv[1]["n"]):
                        rr_v  = s["pay"] / s["inv"] * 100 if s["inv"] > 0 else 0
                        vrows.append({
                            "場":     v,
                            "R数":    s["n"],
                            "的中":   s["hit"],
                            "的中率": f"{s['hit']/s['n']*100:.0f}%" if s["n"] else "-",
                            "回収":   f"¥{s['pay']:,}",
                            "回収率": f"{rr_v:.0f}%",
                        })
                    if vrows:
                        st.dataframe(pd.DataFrame(vrows),
                                     use_container_width=True, hide_index=True)

                if n_pending > 0:
                    st.markdown("### 🕓 未発走予想（当日分）")
                    pending_rows = []
                    for m in M:
                        if m["_status"] != "pending":
                            continue
                        pending_rows.append({
                            "日付":   m["日付"],
                            "場":     m["場"],
                            "R":      m["R"],
                            "スコア": f"{m['スコア']:+.2f}",
                            "差12":   f"{m['差12']:+.2f}",
                            "差23":   f"{m['差23']:+.2f}",
                            "差34":   f"{m['差34']:+.2f}",
                            "点数":   len(m["_bets"]),
                            "買い目": " ".join(m["_bets"]),
                        })
                    st.dataframe(pd.DataFrame(pending_rows),
                                 use_container_width=True, hide_index=True)

                st.markdown("### 📋 レース一覧（全件）")
                rows_disp = []
                for m in M:
                    if m["_hit"] is True:    mk = "✅"
                    elif m["_hit"] is False: mk = "✕"
                    elif m["_status"] == "pending": mk = "⏳"
                    else:                    mk = "?"
                    rows_disp.append({
                        "日付":   m["日付"],
                        "場":     m["場"],
                        "R":      m["R"],
                        "スコア": f"{m['スコア']:+.2f}",
                        "差12":   f"{m['差12']:+.2f}",
                        "差23":   f"{m['差23']:+.2f}",
                        "差34":   f"{m['差34']:+.2f}",
                        "点数":   len(m["_bets"]),
                        "結果":   m["結果"],
                        "払戻":   f"¥{m['払戻']:,}" if m["払戻"] else "-",
                        "判定":   mk,
                    })
                st.dataframe(pd.DataFrame(rows_disp),
                             use_container_width=True, hide_index=True)
