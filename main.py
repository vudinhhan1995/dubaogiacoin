import streamlit as st
import pandas as pd
import requests
import time
import sqlite3
import os
from dudoangia import CoinGeckoPredictor
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timedelta

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Crypto AI Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SQLITE DATABASE & CACHING ---
DB_PATH = "portfolio.db"

def init_db():
    """Khởi tạo database và các bảng cần thiết."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                Coin TEXT PRIMARY KEY,
                'Số lượng' REAL NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS prediction_cache (
                cache_key TEXT PRIMARY KEY,
                predicted_price REAL NOT NULL,
                percent_change REAL NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_portfolio_to_db(portfolio_df):
    """Lưu danh mục vào SQLite bằng to_sql cho hiệu quả."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        portfolio_df.to_sql('portfolio', conn, if_exists='replace', index=False)

def load_portfolio_from_db():
    """Tải danh mục từ SQLite."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["Coin", "Số lượng"])
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM portfolio", conn)
        except pd.io.sql.DatabaseError:
            return pd.DataFrame(columns=["Coin", "Số lượng"])
    return df

def save_prediction_to_cache(cache_key, data):
    """Lưu kết quả dự báo vào cache."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("""
            INSERT OR REPLACE INTO prediction_cache (cache_key, predicted_price, percent_change, cached_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (cache_key, data['predicted_price'], data['percent_change']))

def load_prediction_from_cache(cache_key, max_age_hours=6):
    """Tải kết quả dự báo từ cache nếu nó còn hợp lệ."""
    if not os.path.exists(DB_PATH): return None
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT predicted_price, percent_change, cached_at FROM prediction_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        if row:
            cached_at_str = row[2]
            cached_at = datetime.strptime(cached_at_str, "%Y-%m-%d %H:%M:%S.%f")
            if (datetime.now() - cached_at) < timedelta(hours=max_age_hours):
                return {'predicted_price': row[0], 'percent_change': row[1]}
    return None

init_db()

# --- CSS & KHỞI TẠO CÁC ĐỐI TƯỢNG ---
st.markdown("""
<style>
    .total-asset-container .stMetric { background-color: #f0f2f6; border: 1px solid #dce0e6; border-radius: 10px; padding: 15px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    [data-testid="stDataFrame"] { border: 1px solid #f0f0f0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor(): return CoinGeckoPredictor()
predictor = get_predictor()

def get_current_prices_safe(coin_ids, currency='usd'):
    if not coin_ids: return {}
    api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies={currency}"
    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code == 200: return r.json()
    except requests.RequestException:
        return {}
    return {}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio_from_db()

# ==============================================================================
# --- SIDEBAR ---
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2272/2272825.png", width=60)
    st.title("Crypto AI Analyst")
    st.caption("v1.3 - Cache & Multi-thread")
    st.markdown("---")
    menu = st.radio("Menu Chính", ["📊 Dashboard Dự báo", "💼 Quản lý Danh mục"], index=1)
    st.markdown("---")
    st.info("💡 Dữ liệu danh mục và cache dự báo được lưu vào file `portfolio.db`.")

# ==============================================================================
# --- TRANG 1: DASHBOARD DỰ BÁO ---
# ==============================================================================
if menu == "📊 Dashboard Dự báo":
    st.header("🔮 Phân Tích & Dự Báo Giá")
    
    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with col1:
        coin_input = st.text_input("🔍 Nhập Coin ID", "bitcoin").strip().lower()
    with col2:
        prediction_days = st.number_input("⏳ Số ngày dự báo", min_value=1, max_value=365, value=7)
    with col3:
        btn_predict = st.button("🚀 Chạy Phân Tích", type="primary")

    if btn_predict and coin_input:
        with st.status(f"🤖 Đang phân tích dữ liệu {coin_input.upper()}...", expanded=True) as status:
            df = predictor.fetch_history(coin_input, days=max(365, prediction_days + 60))
            
            if df is None or df.empty:
                st.error(f"❌ Không tìm thấy dữ liệu cho '{coin_input}'. Kiểm tra lại ID coin!")
                status.update(label="Lỗi dữ liệu", state="error")
            else:
                current_price = df['price'].iloc[-1]
                st.metric("Giá hiện tại", f"${current_price:,.4f}")
                
                # --- LOGIC DỰ BÁO ---
                if prediction_days == 1:
                    # Dùng Linear Regression cho ngắn hạn
                    next_date, pred_price, model, score, _ = predictor.predict_linear(df)
                    change = ((pred_price - current_price) / current_price) * 100
                    
                    st.success(f"Dự báo ngày mai: ${pred_price:,.4f} ({change:+.2f}%)")
                    st.info(f"Độ tin cậy mô hình (R²): {score:.2f}")
                    
                    # Vẽ biểu đồ Plotly
                    fig = predictor.create_plotly_chart(df, [next_date], [pred_price], coin_id=coin_input, mode="Linear")
                    st.plotly_chart(fig, width='stretch')

                else:
                    # Dùng Prophet cho dài hạn
                    status.write("🧠 Đang chạy mô hình AI Prophet...")
                    dates, preds, bounds, model, mape = predictor.predict_prophet(df, days_ahead=prediction_days)
                    
                    final_price = preds[-1]
                    change = ((final_price - current_price) / current_price) * 100
                    
                    c1, c2 = st.columns(2)
                    c1.metric(f"Giá dự báo ({prediction_days} ngày)", f"${final_price:,.4f}", f"{change:+.2f}%")
                    c2.metric("Sai số trung bình (MAPE)", f"{mape:.2f}%", delta_color="inverse")
                    
                    if mape < 5: st.caption("✅ Mô hình rất đáng tin cậy.")
                    elif mape < 10: st.caption("⚠️ Độ chính xác trung bình.")
                    else: st.caption("❌ Thị trường biến động mạnh, tham khảo thận trọng.")

                    # Vẽ biểu đồ Plotly
                    status.write("🎨 Đang vẽ biểu đồ tương tác...")
                    fig = predictor.create_plotly_chart(df, dates, preds, bounds, coin_id=coin_input, mode="Prophet")
                    st.plotly_chart(fig, width='stretch')
                
                status.update(label="✅ Phân tích hoàn tất!", state="complete")

# ==============================================================================
# --- TRANG 2: QUẢN LÝ DANH MỤC ---
# ==============================================================================
elif menu == "💼 Quản lý Danh mục":
    st.header("📈 Portfolio & Smart Alerts")
    
    with st.expander("➕ Thêm Coin vào Danh mục"):
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with c1:
            new_coin = st.text_input("Coin ID (vd: bitcoin)", key="new_coin_input").strip().lower()
        with c2:
            new_qty = st.number_input("Số lượng", min_value=0.0, format="%.6f", key="new_qty_input")
        with c3:
            if st.button("Thêm / Cập nhật"):
                if new_coin and new_qty > 0:
                    df = st.session_state.portfolio
                    if new_coin in df["Coin"].values:
                        df.loc[df["Coin"] == new_coin, "Số lượng"] += new_qty
                    else:
                        new_row = pd.DataFrame([{"Coin": new_coin, "Số lượng": new_qty}])
                        df = pd.concat([df, new_row], ignore_index=True)
                    st.session_state.portfolio = df
                    save_portfolio_to_db(df)
                    st.success(f"Đã lưu {new_coin}!")
                    time.sleep(0.5); st.rerun()

    if not st.session_state.portfolio.empty:
        port_df = st.session_state.portfolio.copy()
        coin_ids = list(port_df["Coin"].unique())
        
        with st.spinner("Đang cập nhật giá thị trường..."):
            price_data = get_current_prices_safe(coin_ids, predictor.currency)

        port_df["Giá Hiện Tại"] = port_df["Coin"].apply(lambda cid: price_data.get(cid, {}).get(predictor.currency, 0))
        port_df["Tổng Giá Trị"] = port_df["Số lượng"] * port_df["Giá Hiện Tại"]
        total_net_worth = port_df["Tổng Giá Trị"].sum()
        
        st.markdown("### 💰 Tổng Tài Sản")
        st.markdown('<div class="total-asset-container">', unsafe_allow_html=True)
        st.metric("Net Worth", f"${total_net_worth:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        st.dataframe(port_df, use_container_width=True)
        st.markdown("---")

        st.subheader("🤖 Cố vấn Danh mục AI")
        advisor_cols = st.columns([1, 1, 2])
        with advisor_cols[0]:
            advisor_days = st.number_input("Số ngày dự báo", 1, 90, 7, key="adv_days")
        with advisor_cols[1]:
            st.write(""); btn_advise = st.button("⚡ Quét (Đa luồng + Cache)")

        def analyze_coin_thread(coin_row, days_ahead):
            cid, qty, current_price = coin_row["Coin"], coin_row["Số lượng"], coin_row["Giá Hiện Tại"]
            cache_key = f"{cid}_{days_ahead}"
            
            cached = load_prediction_from_cache(cache_key)
            if cached:
                cached['predicted_value'] = qty * cached['predicted_price']
                return (cached, True, cid)
            
            df_c = predictor.fetch_history(cid)
            if df_c is not None and not df_c.empty:
                _, preds, _, _, _ = predictor.predict_prophet(df_c, days_ahead)
                pred_p = preds[-1]
                change = ((pred_p - current_price) / current_price) * 100 if current_price > 0 else 0
                
                new_data = {"predicted_price": pred_p, "percent_change": change}
                save_prediction_to_cache(cache_key, new_data)
                
                new_data['predicted_value'] = qty * pred_p
                return (new_data, False, cid)
            return (None, False, cid)

        if btn_advise:
            tasks = [row for _, row in port_df.iterrows()]
            final_results, total_pred_value = [], 0
            
            with st.spinner("AI đang khởi động..."):
                with ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
                    worker = partial(analyze_coin_thread, days_ahead=advisor_days)
                    future_results = list(executor.map(worker, tasks))

            pbar = st.progress(0)
            for i, result_pack in enumerate(future_results):
                result_data, from_cache, coin_id = result_pack
                if result_data:
                    final_results.append({
                        "Coin": coin_id,
                        "Hiện Tại": port_df.loc[port_df['Coin'] == coin_id, "Giá Hiện Tại"].iloc[0],
                        f"Dự Báo ({advisor_days}d)": result_data['predicted_price'],
                        "% Thay Đổi": result_data['percent_change']
                    })
                    total_pred_value += result_data["predicted_value"]
                
                pbar.progress((i + 1) / len(tasks), text=f"Đã phân tích {coin_id.upper()} ({'CACHE' if from_cache else 'LIVE'})")
            
            pbar.empty()

            if final_results:
                res_df = pd.DataFrame(final_results).sort_values(by="% Thay Đổi", ascending=False)
                st.dataframe(res_df, use_container_width=True, column_config={"% Thay Đổi": st.column_config.NumberColumn(format="%.2f%%")})
                
                # ... (logic hiển thị lời khuyên giữ nguyên)
    else:
        st.info("👈 Danh mục trống. Hãy thêm coin mới!")

