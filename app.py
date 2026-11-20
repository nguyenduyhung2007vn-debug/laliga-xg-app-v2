# Khối 1: Khai báo thư viện & Cấu hình giao diện CSS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

st.set_page_config(
    page_title="LaLiga xG Predictor & Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #060913 !important;
}
.stApp p, .stApp label, .stApp li, .stApp small {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid #1e293b !important;
}

/* Fix hộp st.expander */
[data-testid="stExpander"] {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus {
    background-color: #0f172a !important;
    color: #00ffcc !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary p {
    color: #00ffcc !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #00ffcc !important;
    color: #00ffcc !important;
}
[data-testid="stExpanderDetails"] {
    background-color: #0b1120 !important;
    border-top: 1px solid #1e293b !important;
}

/* Fix hộp thông báo st.success & st.info */
[data-testid="stAlert"], div[role="alert"] {
    background-color: #0f172a !important;
    border: 1px solid #00ffcc !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] > div, div[role="alert"] > div {
    background-color: transparent !important;
}
[data-testid="stAlert"] p, div[role="alert"] p {
    color: #00ffcc !important;
}

/* Fix nút bấm */
[data-testid="stDownloadButton"] button, .stButton button, [data-testid="stFormSubmitButton"] button {
    background-color: #0f172a !important;
    color: #00ffcc !important;
    border: 1px solid #00ffcc !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
[data-testid="stDownloadButton"] button:hover, .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    background-color: #00ffcc !important;
    color: #060913 !important;
}
[data-testid="stDownloadButton"] button:hover p, .stButton button:hover p, [data-testid="stFormSubmitButton"] button:hover p {
    color: #060913 !important;
}

/* Thẻ KPI Metric */
[data-testid="stMetric"], .stMetric {
    background: linear-gradient(145deg, #0f172a, #1e293b) !important;
    border: 1px solid #334155 !important;
    padding: 18px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5) !important;
}
[data-testid="stMetricValue"] {
    color: #00ffcc !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}
h1, h2, h3, h1 *, h2 *, h3 * {
    color: #00f0ff !important;
    font-weight: 800 !important;
}
.stSelectbox div[data-baseweb="select"], .stNumberInput input, .stDateInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"] * {
    color: #ffffff !important;
}
.explain-box {
    background: linear-gradient(145deg, #0f172a, #162238);
    border-left: 4px solid #00ffcc;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 12px 0;
    color: #f8fafc !important;
    line-height: 1.6;
}
[data-testid="stMetricDelta"] {
    background-color: #1e293b !important;
    padding: 4px 10px !important;
    border-radius: 6px !important;
    width: fit-content !important;
}
[data-testid="stMetricDelta"] * {
    color: #fbbf24 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

def apply_dark_theme(fig, title=None):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#00ffcc", size=15)))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", size=12),
        legend=dict(font=dict(color="#f8fafc"), title=dict(font=dict(color="#f8fafc"))),
        coloraxis_colorbar=dict(tickfont=dict(color="#f8fafc"), title=dict(font=dict(color="#f8fafc")))
    )
    fig.update_xaxes(color="#f8fafc", tickfont=dict(color="#f8fafc"), title_font=dict(color="#f8fafc"), gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(color="#f8fafc", tickfont=dict(color="#f8fafc"), title_font=dict(color="#f8fafc"), gridcolor="#1e293b", zerolinecolor="#334155")
    return fig

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "matches_laliga_clean.csv"
NEW_DATA_FILE = BASE_DIR / "matches_laliga_new_clean.csv"
import urllib.request
import json

TEAM_NAME_MAP = {
    "Atlético Madrid": "Atletico Madrid",
    "Alavés": "Alaves",
    "Deportivo Alavés": "Alaves",
    "Leganés": "Leganes",
    "Cádiz": "Cadiz",
    "Almería": "Almeria",
    "Betis": "Real Betis",
    "Real Betis": "Real Betis",
    "Real Valladolid": "Valladolid",
    "RCD Espanyol": "Espanyol",
    "Real Oviedo": "Oviedo",
    "Athletic Club": "Athletic Club",
    "RC Celta": "Celta Vigo",
    "Celta Vigo": "Celta Vigo",
    "Rayo Vallecano": "Rayo Vallecano",
    "Real Sociedad": "Real Sociedad",
    "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Getafe CF": "Getafe",
    "CA Osasuna": "Osasuna",
    "Villarreal CF": "Villarreal",
    "FC Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Elche CF": "Elche",
    "Levante UD": "Levante"
}

@st.cache_data(ttl=3600)  # Tự động làm mới dữ liệu mỗi 1 giờ khi có người mở web
def auto_sync_laliga_data(old_csv_path, new_csv_path):
    """Tự động đồng bộ các trận La Liga mới nhất từ API ESPN mỗi khi mở web."""
    old_df = pd.read_csv(old_csv_path)
    old_df.columns = old_df.columns.str.strip().str.lower()
    old_df["date"] = pd.to_datetime(old_df["date"]).dt.strftime("%Y-%m-%d")
    last_date = old_df["date"].max()

    # Tính trung bình xG/bàn thắng lịch sử của từng đội để quy đổi xG chuẩn xác nếu API chỉ trả về số cú sút
    team_xg_avg = old_df.groupby("team")["xg"].mean().to_dict()
    league_xg_avg = float(old_df["xg"].mean())

    new_rows = []
    # Quét lịch thi đấu & kết quả từ 2025-10-01 đến ngày hiện tại
    date_ranges = ["20251001-20260630", "20260801-20270630"]

    for dr in date_ranges:
        url = f"https://espn.com{dr}&limit=500"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))

            for ev in data.get("events", []):
                status = ev.get("status", {}).get("type", {}).get("completed", False)
                if not status:
                    continue

                d_str = str(ev.get("date", ""))[:10]
                if d_str <= last_date:
                    continue

                yr = int(d_str[:4]) if int(d_str[5:7]) >= 8 else int(d_str[:4]) - 1
                comp = ev.get("competitions", [{}])
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors)
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors)

                raw_h = home_c.get("team", {}).get("displayName", "")
                raw_a = away_c.get("team", {}).get("displayName", "")
                ht = TEAM_NAME_MAP.get(raw_h, raw_h)
                at = TEAM_NAME_MAP.get(raw_a, raw_a)

                hg = int(float(home_c.get("score", 0)))
                ag = int(float(away_c.get("score", 0)))

                # Lấy xG nếu có trong statistics, nếu không thì ước lượng kết hợp bàn thắng & sức mạnh đội
                def extract_stat(comp_obj, stat_name):
                    for s in comp_obj.get("statistics", []):
                        if s.get("name") == stat_name:
                            try:
                                return float(s.get("displayValue", 0))
                            except Exception:
                                return None
                    return None

                hx = extract_stat(home_c, "expectedGoals")
                ax = extract_stat(away_c, "expectedGoals")

                if hx is None:
                    hx = round(0.55 * hg + 0.45 * team_xg_avg.get(ht, league_xg_avg), 2)
                if ax is None:
                    ax = round(0.55 * ag + 0.45 * team_xg_avg.get(at, league_xg_avg), 2)

                new_rows.append({
                    "season": yr, "date": d_str, "team": ht, "opponent": at, "venue": "Home",
                    "result": "W" if hg > ag else ("L" if hg < ag else "D"),
                    "gf": hg, "ga": ag, "xg": hx, "xga": ax, "sh": np.nan, "sot": np.nan
                })
                new_rows.append({
                    "season": yr, "date": d_str, "team": at, "opponent": ht, "venue": "Away",
                    "result": "W" if ag > hg else ("L" if ag < hg else "D"),
                    "gf": ag, "ga": hg, "xg": ax, "xga": hx, "sh": np.nan, "sot": np.nan
                })
        except Exception:
            continue

    if not new_rows:
        return False, 0

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["team"] = combined["team"].astype(str).str.strip().replace(TEAM_NAME_MAP)
    combined["opponent"] = combined["opponent"].astype(str).str.strip().replace(TEAM_NAME_MAP)
    combined = combined.drop_duplicates(subset=["date", "team", "opponent"], keep="last").sort_values(["date", "team"]).reset_index(drop=True)

    for col in ["sh", "sot"]:
        if col in combined.columns:
            combined[col] = combined.groupby("team")[col].transform(lambda s: s.fillna(s.median())).fillna(combined[col].median())

    combined.to_csv(new_csv_path, index=False)
    return True, (len(combined) - len(old_df)) // 2

# Khối 3: Huấn luyện mô hình ML & Hàm mô phỏng Poisson
@st.cache_resource
def train_ml_models(df, feature_cols):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    max_season = df_sorted["season"].max()

    if df_sorted["season"].nunique() > 1:
        train_df = df_sorted[df_sorted["season"] < max_season]
        test_df = df_sorted[df_sorted["season"] == max_season]
    else:
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["xg"]
    X_test, y_test = test_df[feature_cols], test_df["xg"]

    candidates = {}

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    candidates["Linear Regression"] = {"model": lr_model, "pred": lr_model.predict(X_test)}

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    candidates["Random Forest"] = {"model": rf_model, "pred": rf_model.predict(X_test)}

    if HAS_XGBOOST:
        xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        xgb_model.fit(X_train, y_train)
        candidates["XGBoost"] = {"model": xgb_model, "pred": xgb_model.predict(X_test)}

    metrics = {}
    for name, item in candidates.items():
        metrics[name] = {
            "MAE": mean_absolute_error(y_test, item["pred"]),
            "RMSE": np.sqrt(mean_squared_error(y_test, item["pred"])),
            "R2": r2_score(y_test, item["pred"])
        }

    best_model_name = min(metrics, key=lambda k: metrics[k]["MAE"])
    best_model = candidates[best_model_name]["model"]
    best_pred = candidates[best_model_name]["pred"]

    base_pred = X_test["xg_avg5"].values
    metrics["Historical Baseline"] = {
        "MAE": mean_absolute_error(y_test, base_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, base_pred)),
        "R2": r2_score(y_test, base_pred)
    }

    return best_model, best_model_name, metrics, X_test, y_test, best_pred, int(max_season), test_df

def build_latest_team_stats(df, team_name):
    team_matches = df[df["team"] == team_name].sort_values("date")
    last5 = team_matches.tail(5)
    last3 = team_matches.tail(3)
    matches_seen = len(team_matches)

    xg_avg5 = last5["xg"].mean()
    xga_avg5 = last5["xga"].mean()

    return {
        "xg_avg5": xg_avg5,
        "xga_avg5": xga_avg5,
        "goals_avg5": last5["gf"].mean(),
        "goals_against_avg5": last5["ga"].mean(),
        "sot_avg5": last5["sot"].mean(),
        "xg_avg3": last3["xg"].mean(),
        "net_xg_avg5": xg_avg5 - xga_avg5,
        "matches_seen": matches_seen,
        "short_history": int(matches_seen < 5)
    }

def poisson_matrix(home_xg, away_xg, max_goals=8):
    goals = np.arange(max_goals + 1)
    home_prob = poisson.pmf(goals, home_xg)
    away_prob = poisson.pmf(goals, away_xg)
    matrix = np.outer(home_prob, away_prob)
    return matrix / matrix.sum()

def calculate_outcomes(matrix):
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total
# Khối 4: Thanh Sidebar (Tự động đồng bộ dữ liệu khi mở web) & Trang 1
st.sidebar.title("⚽ LaLiga xG Predictor")
st.sidebar.caption("Machine Learning & Poisson Analytics")
st.sidebar.markdown("---")

# Tự động đồng bộ dữ liệu thực tế mới nhất ngay khi mở web
sync_ok, added_count = auto_sync_laliga_data(DATA_FILE, NEW_DATA_FILE)

dataset_options = []
if NEW_DATA_FILE.exists():
    dataset_options.append("🆕 Dữ liệu mới tự động cập nhật (Mở rộng)")
dataset_options.append("📁 Dữ liệu chuẩn Báo cáo (4.700 dòng gốc)")

selected_dataset = st.sidebar.selectbox("CHỌN BỘ DỮ LIỆU", dataset_options)
ACTIVE_DATA_FILE = NEW_DATA_FILE if selected_dataset.startswith("🆕") else DATA_FILE

page = st.sidebar.radio(
    "MENU CHỨC NĂNG",
    [
        "🏠 Tổng quan hệ thống",
        "🏟️ Match Prediction (Học máy)",
        "📊 Phân tích xG & Phong độ",
        "🤖 Đánh giá Mô hình Học máy"
    ]
)
st.sidebar.markdown("---")

try:
    if not ACTIVE_DATA_FILE.exists():
        st.error(f"Không tìm thấy tệp dữ liệu '{ACTIVE_DATA_FILE.name}'.")
        st.stop()

    with st.spinner("⚽ Đang nạp dữ liệu và huấn luyện mô hình..."):
        df, feature_cols = load_and_process_data(ACTIVE_DATA_FILE)
        best_model, best_model_name, model_metrics, X_test, y_test, test_preds, test_season, test_df = train_ml_models(df, feature_cols)
        teams = sorted(df["team"].unique().tolist())

    st.sidebar.success(f"Mô hình tối ưu: **{best_model_name}**")
    if sync_ok and added_count > 0:
        st.sidebar.info(f"🔄 Đã tự động đồng bộ thêm **{added_count}** trận mới nhất!")
    st.sidebar.caption(f"Tệp: `{ACTIVE_DATA_FILE.name}` | Test: **{test_season}**")


    # ------------------ TRANG 1: TỔNG QUAN ------------------
    if page == "🏠 Tổng quan hệ thống":
        st.title("🏆 HỆ THỐNG PHÂN TÍCH BÓNG ĐÁ LALIGA")
        st.markdown(f"Thuật toán tối ưu đang kích hoạt: **{best_model_name}** | Mùa giải kiểm thử: **{test_season}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng bản ghi đội–trận", f"{len(df):,}")
        c2.metric("Số CLB LaLiga", df["team"].nunique())
        c3.metric("Số mùa giải", df["season"].nunique())
        c4.metric("Bàn thắng TB/trận", f"{df['gf'].mean():.2f}")

        with st.expander("💡 Hiểu xG (Expected Goals) trong 30 giây — Dành cho người mới", expanded=True):
            g1, g2 = st.columns([1.4, 1])
            with g1:
                st.markdown("""
                * **xG (Bàn thắng kỳ vọng) là gì?** Đo chất lượng các cơ hội ghi bàn mà một đội tạo ra (từ `0.01` đến `1.00` cho mỗi cú sút).
                * **Ví dụ:** Một cú sút có `xG = 0.25` nghĩa là trung bình 100 cú sút tương tự sẽ có **25 bàn thắng**. Nếu đội A tạo ra 4 cơ hội như vậy, tổng **xG của đội A là 1.00**.
                * **Lưu ý:** `xG` đo chất lượng cơ hội tạo ra trước trận, không phải tỷ số chắc chắn của trận đấu.
                """)
            with g2:
                shot_case = st.selectbox(
                    "Thử nghiệm nhanh chất lượng tình huống sút:",
                    [
                        "Sút cận thành góc rộng (~0.65 xG)",
                        "Đá phạt đền Penalty 11m (~0.79 xG)",
                        "Sút xa ngoài vòng cấm >20m (~0.04 xG)"
                    ]
                )
                st.info(f"Đánh giá tình huống: **{shot_case}**")

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📊 Phân phối xG (Sân nhà vs Sân khách)")
            fig = px.histogram(
                df, x="xg", color="venue", barmode="overlay", marginal="box",
                color_discrete_map={"Home": "#00ffcc", "Away": "#3b82f6"},
                labels={"xg": "Bàn thắng kỳ vọng (xG)", "venue": "Địa điểm", "count": "Số trận"}
            )
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

        with col_g2:
            st.subheader("📈 Ma trận tương quan chỉ số chuyên môn")
            corr_matrix = df[["gf", "ga", "xg", "xga", "sh", "sot"]].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="Tealgrn")
            st.plotly_chart(apply_dark_theme(fig_corr), use_container_width=True)
# Khối 5: Trang 2 (Match Prediction & Ma trận tỷ số Poisson)
    # ------------------ TRANG 2: MATCH PREDICTION ------------------
    elif page == "🏟️ Match Prediction (Học máy)":
        st.title("🏟️ DỰ ĐOÁN TRẬN ĐẤU & MÔ PHỎNG POISSON")
        st.markdown(f"Thuật toán kích hoạt: **{best_model_name}** | Kết hợp mô phỏng xác suất tỷ số Poisson.")

        pred_mode = st.radio(
            "Chọn chế độ dự báo:",
            [
                "🎯 Mô phỏng cặp đấu tùy chọn (Dựa trên phong độ 5 trận gần nhất)",
                "📅 Kiểm chứng hồi cứu trận thực tế trong tập Test"
            ],
            horizontal=True
        )

        actual_home_xg, actual_away_xg, actual_gf, actual_ga = None, None, None, None

        if pred_mode.startswith("🎯"):
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("🏠 Chọn đội chủ nhà (Home)", teams, index=teams.index("Barcelona") if "Barcelona" in teams else 0)
            with col2:
                away_team = st.selectbox("🚀 Chọn đội khách (Away)", teams, index=teams.index("Real Madrid") if "Real Madrid" in teams else 1)

            if home_team == away_team:
                st.warning("⚠️ Vui lòng chọn 2 câu lạc bộ khác nhau để thiết lập cặp đấu.")
                st.stop()

            h_stats = build_latest_team_stats(df, home_team)
            a_stats = build_latest_team_stats(df, away_team)

            home_features = pd.DataFrame([{
                **h_stats,
                "opponent_xg_avg5": a_stats["xg_avg5"],
                "opponent_xga_avg5": a_stats["xga_avg5"],
                "is_home": 1,
                "opponent_matches_seen": a_stats["matches_seen"],
                "opponent_short_history": a_stats["short_history"],
                "attack_defense_index": (h_stats["xg_avg5"] + a_stats["xga_avg5"]) / 2
            }])

            away_features = pd.DataFrame([{
                **a_stats,
                "opponent_xg_avg5": h_stats["xg_avg5"],
                "opponent_xga_avg5": h_stats["xga_avg5"],
                "is_home": 0,
                "opponent_matches_seen": h_stats["matches_seen"],
                "opponent_short_history": h_stats["short_history"],
                "attack_defense_index": (a_stats["xg_avg5"] + h_stats["xga_avg5"]) / 2
            }])
            h_sot_disp, a_sot_disp = h_stats["sot_avg5"], a_stats["sot_avg5"]
        else:
            home_matches_test = test_df[test_df["venue"] == "Home"].sort_values("date", ascending=False).copy()
            home_matches_test["label"] = (
                home_matches_test["date"].dt.strftime("%Y-%m-%d") + " | " +
                home_matches_test["team"] + " vs " + home_matches_test["opponent"]
            )
            chosen_match = st.selectbox(f"Chọn trận đấu thực tế trong tập Test (Mùa {test_season}):", home_matches_test["label"].tolist())
            h_row = home_matches_test[home_matches_test["label"] == chosen_match].iloc
            home_team = h_row["team"]
            away_team = h_row["opponent"]

            a_row = test_df[(test_df["date"] == h_row["date"]) & (test_df["team"] == away_team) & (test_df["venue"] == "Away")].iloc
            home_features = pd.DataFrame([h_row[feature_cols].to_dict()])
            away_features = pd.DataFrame([a_row[feature_cols].to_dict()])

            h_sot_disp, a_sot_disp = float(h_row["sot_avg5"]), float(a_row["sot_avg5"])
            actual_home_xg, actual_away_xg = float(h_row["xg"]), float(a_row["xg"])
            actual_gf, actual_ga = int(h_row["gf"]), int(a_row["gf"])

        pred_home_xg = max(0.1, float(best_model.predict(home_features[feature_cols])[0]))
        pred_away_xg = max(0.1, float(best_model.predict(away_features[feature_cols])[0]))


        st.markdown("---")
        st.subheader("⚔️ CHỈ SỐ BÀN THẮNG KỲ VỌNG DỰ BÁO (xG)")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Dự báo xG {home_team}", f"{pred_home_xg:.2f}")
        k2.metric(f"Dự báo xG {away_team}", f"{pred_away_xg:.2f}")
        k3.metric(f"Sút trúng đích TB ({home_team})", f"{h_sot_disp:.1f}")
        k4.metric(f"Sút trúng đích TB ({away_team})", f"{a_sot_disp:.1f}")

        stronger_team = home_team if pred_home_xg > pred_away_xg else away_team
        xg_gap = abs(pred_home_xg - pred_away_xg)
        st.markdown(f"""
        <div class="explain-box">
            <b>🤖 AI Giải thích dự báo:</b> Mô hình ước tính <b>{home_team}</b> tạo ra chất lượng cơ hội tương đương <b>{pred_home_xg:.2f} xG</b> và <b>{away_team}</b> đạt <b>{pred_away_xg:.2f} xG</b>. 
            <b>{stronger_team}</b> nhỉnh hơn về khả năng tạo cơ hội ghi bàn (chênh lệch <b>{xg_gap:.2f} xG</b>) dựa trên lợi thế sân bãi và phong độ công – thủ 5 trận trước đó. 
            <i>(Lưu ý: xG phản ánh chất lượng cơ hội tạo ra, không phải tỷ số chắc chắn của trận đấu).</i>
        </div>
        """, unsafe_allow_html=True)

        if actual_home_xg is not None:
            st.subheader("⚖️ Đối chiếu thực tế sau trận đấu (xG Dự báo vs xG Thực tế vs Bàn thắng)")
            c_act1, c_act2, c_act3 = st.columns(3)
            c_act1.metric(f"xG Thực tế ({home_team})", f"{actual_home_xg:.2f}", f"Lệch: {pred_home_xg - actual_home_xg:+.2f}")
            c_act2.metric(f"xG Thực tế ({away_team})", f"{actual_away_xg:.2f}", f"Lệch: {pred_away_xg - actual_away_xg:+.2f}")
            c_act3.metric("Tỷ số thực tế trên sân", f"{actual_gf} - {actual_ga}", f"{home_team} vs {away_team}")

        matrix = poisson_matrix(pred_home_xg, pred_away_xg, max_goals=6)
        home_win, draw, away_win = calculate_outcomes(matrix)

        flat_scores = [(f"{home_team} {i} - {j} {away_team}", matrix[i, j] * 100) for i in range(5) for j in range(5)]
        top3_scores = sorted(flat_scores, key=lambda x: x, reverse=True)[:3]

        st.markdown("---")
        col_p1, col_p2 = st.columns([1.05, 1.15])

        with col_p1:
            st.subheader("🎯 Xác suất kết quả trận đấu (1X2)")
            fig_prob = go.Figure(go.Bar(
                y=[f"{away_team} Thắng", "Hòa", f"{home_team} Thắng"],
                x=[away_win * 100, draw * 100, home_win * 100],
                orientation='h',
                marker_color=["#3b82f6", "#fbbf24", "#00ffcc"],
                text=[f"{away_win * 100:.1f}%", f"{draw * 100:.1f}%", f"{home_win * 100:.1f}%"],
                textposition="inside"
            ))
            fig_prob.update_layout(xaxis_title="Xác suất (%)")
            st.plotly_chart(apply_dark_theme(fig_prob), use_container_width=True)

            st.markdown("**🔥 Top 3 tỷ số có xác suất cao nhất:**")
            for idx_s, (sc_label, sc_prob) in enumerate(top3_scores, 1):
                st.markdown(f"- Hạng {idx_s}: **{sc_label}** (`{sc_prob:.1f}%`)")

        with col_p2:
            st.subheader("🔢 Ma trận xác suất tỷ số chính xác (%)")
            score_df = pd.DataFrame(
                np.round(matrix[:5, :5] * 100, 1),
                index=[f"{home_team} {i}" for i in range(5)],
                columns=[f"{away_team} {j}" for j in range(5)]
            )
            fig_heat = px.imshow(score_df, text_auto=True, color_continuous_scale="Blues")
            st.plotly_chart(apply_dark_theme(fig_heat), use_container_width=True)
# Khối 6: Trang 3 (Phân tích phong độ), Trang 4 (Đánh giá mô hình) & Xử lý ngoại lệ
    # ------------------ TRANG 3: TEAM ANALYTICS ------------------
    elif page == "📊 Phân tích xG & Phong độ":
        st.title("📈 PHÂN TÍCH HỒ SƠ PHONG ĐỘ CÂU LẠC BỘ")
        selected_team = st.selectbox("Chọn câu lạc bộ cần tra cứu", teams)
        team_df = df[df["team"] == selected_team].sort_values("date", ascending=False)
        xg_diff = team_df["gf"].mean() - team_df["xg"].mean()

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Số trận đã đấu", len(team_df))
        t2.metric("xG trung bình/trận", f"{team_df['xg'].mean():.2f}")
        t3.metric("xGA trung bình/trận", f"{team_df['xga'].mean():.2f}")
        t4.metric("Bàn thắng TB/trận", f"{team_df['gf'].mean():.2f}")
        t5.metric("Hiệu suất (GF - xG)", f"{xg_diff:+.2f}", "Vượt kỳ vọng" if xg_diff > 0 else "Dưới kỳ vọng")

        st.markdown("---")
        st.subheader("⏳ Diễn biến chỉ số xG & xGA trung bình qua các mùa giải")
        seasonal_trend = team_df.groupby("season")[["xg", "xga", "gf", "ga"]].mean().reset_index()
        fig_trend = px.line(
            seasonal_trend, x="season", y=["xg", "xga"], markers=True,
            color_discrete_map={"xg": "#00ffcc", "xga": "#ef4444"},
            labels={"value": "Chỉ số trung bình", "season": "Mùa giải", "variable": "Chỉ số"}
        )
        st.plotly_chart(apply_dark_theme(fig_trend), use_container_width=True)

        st.subheader(f"📋 Lịch sử 15 trận gần nhất của {selected_team}")
        display_df = team_df[["date", "opponent", "venue", "result", "gf", "ga", "xg", "xga", "sh", "sot"]].head(15).copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True)

    # ------------------ TRANG 4: MODEL EVALUATION ------------------
    elif page == "🤖 Đánh giá Mô hình Học máy":
        st.title("🤖 THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH")
        st.markdown(f"Kết quả kiểm thử độc lập trên **mùa giải {test_season}** ({len(X_test)} bản ghi đội–trận, tương ứng {len(X_test)//2} trận đấu).")

        st.subheader("📊 Bảng đối chiếu sai số giữa các thuật toán")
        ordered_names = sorted(model_metrics, key=lambda k: model_metrics.get(k).get("MAE"))
        metrics_data = []
        for model_name in ordered_names:
            score = model_metrics.get(model_name)
            metrics_data.append({
                "Thuật toán mô hình": model_name,
                "MAE (Sai số tuyệt đối TB)": f"{score.get('MAE'):.4f}",
                "RMSE (Sai số bình phương TB)": f"{score.get('RMSE'):.4f}",
                "R² (Hệ số xác định)": f"{score.get('R2'):.4f}",
                "Trạng thái": "🏆 Tối ưu nhất" if model_name == best_model_name else "Đối chiếu"
            })
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)

        st.download_button(
            label="📥 Tải bảng kết quả đánh giá (.CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="laliga_xg_model_benchmark.csv",
            mime="text/csv"
        )

        st.markdown("---")
        col_g1, col_g2, col_g3 = st.columns(3)

        with col_g1:
            st.subheader("🎯 Trọng số đặc trưng")
            if hasattr(best_model, "feature_importances_"):
                values = best_model.feature_importances_
                label_title = f"Feature Importance ({best_model_name})"
            elif hasattr(best_model, "coef_"):
                values = np.abs(best_model.coef_)
                label_title = f"Độ lớn trọng số |Coef| ({best_model_name})"
            else:
                values = None

            if values is not None:
                indices = np.argsort(values)
                fig_importance = go.Figure(go.Bar(
                    y=[feature_cols[i] for i in indices],
                    x=[values[i] for i in indices],
                    orientation='h',
                    marker_color="#00f0ff"
                ))
                st.plotly_chart(apply_dark_theme(fig_importance, label_title), use_container_width=True)

        with col_g2:
            st.subheader("📉 Phân phối Phần dư")
            residuals = y_test - test_preds
            fig_res = px.histogram(
                x=residuals, nbins=30,
                labels={'x': 'Sai số (xG Thực tế - xG Dự báo)', 'count': 'Số trận'},
                color_discrete_sequence=["#00ffcc"]
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444")
            st.plotly_chart(apply_dark_theme(fig_res, f"Phân phối phần dư ({best_model_name})"), use_container_width=True)

        with col_g3:
            st.subheader("📐 Thực tế vs Dự báo")
            fig_scat = px.scatter(
                x=y_test, y=test_preds, opacity=0.75,
                labels={'x': 'xG Thực tế (Actual)', 'y': 'xG Dự báo (Predicted)'},
                color_discrete_sequence=["#00ffcc"]
            )
            max_v = max(float(y_test.max()), float(test_preds.max()))
            fig_scat.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", line=dict(color="#fbbf24", dash="dash"), name="Đường chuẩn y=x"))
            st.plotly_chart(apply_dark_theme(fig_scat, f"Actual vs Predicted ({best_model_name})"), use_container_width=True)

except Exception as e:
    st.error(f"Ứng dụng gặp sự cố kỹ thuật trong quá trình xử lý: {e}")
    st.info("Hãy đảm bảo tệp dữ liệu 'matches_laliga_clean.csv' nằm cùng thư mục với file 'app.py'.")
