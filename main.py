import streamlit as st
import pandas as pd
import requests
import time
import sqlite3
import os
from dudoangia import CoinGeckoPredictor

# --- CẤU HÌNH TRANG (Phải đặt đầu tiên) ---
st.set_page_config(
    page_title="Crypto AI Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SQLITE DATABASE PATH ---
DB_PATH = "portfolio.db"

# --- HÀM SQLITE ---
def init_db():
    """Khởi tạo database và bảng portfolio"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            coin_id TEXT PRIMARY KEY,
            quantity REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_portfolio_to_db(portfolio_df):
    """Lưu danh mục vào SQLite"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio") # Xóa cũ
    for _, row in portfolio_df.iterrows():
        c.execute("""
            INSERT OR REPLACE INTO portfolio (coin_id, quantity, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (row["Coin"], row["Số lượng"]))
    conn.commit()
    conn.close()

def load_portfolio_from_db():
    """Tải danh mục từ SQLite"""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["Coin", "Số lượng"])
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT coin_id as Coin, quantity as 'Số lượng' FROM portfolio", conn)
    except:
        df = pd.DataFrame(columns=["Coin", "Số lượng"])
    conn.close()
    return df

# Khởi tạo database ngay khi chạy app
init_db()

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .total-asset-container .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #dce0e6;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    /* Tùy chỉnh bảng dữ liệu cho đẹp hơn */
    [data-testid="stDataFrame"] {
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo bộ dự báo
@st.cache_resource
def get_predictor():
    return CoinGeckoPredictor()

predictor = get_predictor()

# --- HÀM HỖ TRỢ LẤY GIÁ AN TOÀN ---
def get_current_prices_safe(coin_ids, currency='usd'):
    if not coin_ids: return {}
    api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies={currency}"
    for i in range(3):
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200: return response.json()
            elif response.status_code == 429:
                time.sleep(2 * (i + 1))
                continue
            return {}
        except:
            time.sleep(1)
            continue
    return {}

# --- LOAD DATA TỪ DB VÀO SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio_from_db()

# ==============================================================================
# --- SIDEBAR ---
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2272/2272825.png", width=60)
    st.title("Crypto AI Analyst")
    st.caption("v1.2 - Database Integrated")
    st.markdown("---")
    menu = st.radio("Menu Chính", ["📊 Dashboard Dự báo", "💼 Quản lý Danh mục"], index=0)
    st.markdown("---")
    st.info("💡 **Mẹo:** Dữ liệu Portfolio đã được tự động lưu vào Database.")

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

    if btn_predict:
        coin_id = predictor.extract_coin_id(coin_input)
        if not coin_id:
            st.error("❌ Coin ID không hợp lệ!")
        else:
            with st.status(f"🤖 Đang phân tích dữ liệu {coin_id.upper()}...", expanded=True) as status:
                df = predictor.fetch_history(coin_id, days=max(365, prediction_days + 30))
                
                if df is None or df.empty:
                    status.update(label="❌ Lỗi dữ liệu!", state="error")
                    st.error(f"Không tìm thấy dữ liệu cho '{coin_id}'. Hãy kiểm tra lại tên ID trên CoinGecko.")
                else:
                    current_price = df['price'].iloc[-1]
                    
                    if prediction_days == 1:
                        st.write("3. Chạy Linear Regression...")
                        date, pred, model, score, mape = predictor.predict_linear(df, 1)
                        fig = predictor.visualize_linear(df, date, pred, model, coin_id)
                        delta = ((pred - current_price) / current_price) * 100
                        
                        status.update(label="✅ Hoàn tất!", state="complete", expanded=True)
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Giá Hiện Tại", f"${current_price:,.4f}")
                        m2.metric("Dự Báo (1 Ngày)", f"${pred:,.4f}", f"{delta:.2f}%")
                        m3.metric("R² Score", f"{score:.2f}")
                        st.pyplot(fig)
                        
                    else:
                        st.write("3. Chạy Prophet AI...")
                        dates, preds, bounds, _, mape = predictor.predict_prophet(df, prediction_days)
                        fig = predictor.visualize_prophet(df, dates, preds, bounds, coin_id)
                        last_pred = preds[-1]
                        delta = ((last_pred - current_price) / current_price) * 100
                        
                        status.update(label="✅ Hoàn tất!", state="complete", expanded=True)
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Giá Hiện Tại", f"${current_price:,.4f}")
                        m2.metric(f"Mục Tiêu ({prediction_days} Ngày)", f"${last_pred:,.4f}", f"{delta:.2f}%")
                        m3.metric("MAPE (Sai số)", f"{mape:.2f}%", delta_color="inverse")
                        st.pyplot(fig)

# ==============================================================================
# --- TRANG 2: QUẢN LÝ DANH MỤC ---
# ==============================================================================
elif menu == "💼 Quản lý Danh mục":
    st.header("📈 Portfolio & Smart Alerts")
    
    # --- FORM THÊM COIN ---
    with st.expander("➕ Thêm Coin vào Danh mục", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with c1:
            new_coin = st.text_input("Coin ID (vd: monad)", key="new_coin").strip().lower()
        with c2:
            new_qty = st.number_input("Số lượng", min_value=0.0, format="%.6f", key="new_qty")
        with c3:
            if st.button("Thêm / Cập nhật"):
                if new_coin and new_qty > 0:
                    if new_coin in st.session_state.portfolio["Coin"].values:
                        st.session_state.portfolio.loc[st.session_state.portfolio["Coin"] == new_coin, "Số lượng"] += new_qty
                    else:
                        new_row = pd.DataFrame([{"Coin": new_coin, "Số lượng": new_qty}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    
                    save_portfolio_to_db(st.session_state.portfolio) # Lưu DB
                    st.success(f"Đã lưu {new_coin} vào Database!")
                    time.sleep(0.5)
                    st.rerun()

    # --- HIỂN THỊ DANH MỤC ---
    if not st.session_state.portfolio.empty:
        currency = predictor.currency
        coin_ids = st.session_state.portfolio["Coin"].unique()
        
        with st.spinner("Đang cập nhật giá thị trường..."):
            price_data = get_current_prices_safe(list(coin_ids), currency)

        port_df = st.session_state.portfolio.copy()
        current_prices = []
        total_values = []
        
        for _, row in port_df.iterrows():
            cid = row["Coin"]
            price = price_data.get(cid, {}).get(currency, 0)
            current_prices.append(price)
            total_values.append(price * row["Số lượng"])
            
        port_df["Giá Hiện Tại"] = current_prices
        port_df["Tổng Giá Trị"] = total_values
        total_net_worth = sum(total_values)
        
        st.markdown("### 💰 Tổng Tài Sản")
        st.markdown('<div class="total-asset-container">', unsafe_allow_html=True)
        st.metric("Net Worth", f"${total_net_worth:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        # --- FIX CẢNH BÁO MÀU ĐỎ ---
        # Thay use_container_width=True bằng width="stretch" (cho bản mới) 
        # Hoặc xóa bỏ nếu bản cũ, ở đây dùng 'stretch' cho bản Streamlit mới nhất
        try:
            st.dataframe(port_df, width=None, use_container_width=True) 
        except:
            # Fallback nếu version quá cũ
            st.dataframe(port_df)
        
        st.markdown("---")

        # --- CỐ VẤN AI ---
        st.subheader("🤖 Cố vấn Danh mục AI")
        advisor_cols = st.columns([1, 1, 2])
        with advisor_cols[0]:
            advisor_days = st.number_input("Số ngày dự báo", 1, 90, 7, key="adv_days")
        with advisor_cols[1]:
            st.write("")
            btn_advise = st.button("🔍 Quét Danh Mục")

        if btn_advise:
            results = []
            total_pred = 0
            pbar = st.progress(0, text="AI đang phân tích...")

            for i, row in port_df.iterrows():
                cid = row["Coin"]
                qty = row["Số lượng"]
                pbar.progress((i+1)/len(port_df), text=f"Đang tính toán {cid.upper()}...")
                
                df_c = predictor.fetch_history(cid, days=365)
                if df_c is not None:
                    _, preds, _, _, _ = predictor.predict_prophet(df_c, days_ahead=advisor_days)
                    pred_p = preds[-1]
                    pred_v = qty * pred_p
                    change = ((pred_p - row["Giá Hiện Tại"]) / row["Giá Hiện Tại"]) * 100 if row["Giá Hiện Tại"] > 0 else 0
                    
                    results.append({
                        "Coin": cid,
                        "Hiện Tại": row["Giá Hiện Tại"],
                        f"Dự Báo ({advisor_days}d)": pred_p,
                        "% Thay Đổi": change
                    })
                    total_pred += pred_v
                time.sleep(1) # Tránh Rate Limit

            pbar.empty()
            if results:
                res_df = pd.DataFrame(results).sort_values(by="% Thay Đổi", ascending=False)
                
                # Hiển thị bảng kết quả (Fix cảnh báo đỏ)
                st.dataframe(
                    res_df, 
                    use_container_width=True,
                    column_config={"% Thay Đổi": st.column_config.NumberColumn(format="%.2f%%")}
                )

                top = res_df.iloc[0]
                total_change = ((total_pred - total_net_worth)/total_net_worth)*100 if total_net_worth > 0 else 0
                
                st.success(f"🌟 **Ngôi sao sáng:** {top['Coin'].upper()} (+{top['% Thay Đổi']:.2f}%)")
                if total_change > 0:
                    st.info(f"📈 Tổng tài sản dự kiến **TĂNG {total_change:.2f}%**.")
                else:
                    st.warning(f"📉 Tổng tài sản dự kiến **GIẢM {total_change:.2f}%**.")

        # --- XÓA COIN ---
        with st.expander("🗑 Xóa Coin khỏi danh mục"):
            del_coin = st.selectbox("Chọn coin để xóa", port_df["Coin"].unique())
            if st.button("Xác nhận xóa"):
                st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio["Coin"] != del_coin]
                save_portfolio_to_db(st.session_state.portfolio) # Lưu lại DB sau khi xóa
                st.rerun()
    else:
        st.info("👈 Danh mục trống. Hãy thêm coin mới!")