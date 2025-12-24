import streamlit as st
import pandas as pd
import requests
import time
from dudoangia import CoinGeckoPredictor

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Crypto AI Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .total-asset-container .stMetric {
        background-color: #e6f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo bộ dự báo
@st.cache_resource
def get_predictor():
    return CoinGeckoPredictor()

predictor = get_predictor()

# --- HÀM HỖ TRỢ LẤY GIÁ AN TOÀN (FIX LỖI CRASH) ---
def get_current_prices_safe(coin_ids, currency='usd'):
    """Lấy giá hiện tại với cơ chế thử lại để tránh lỗi 429 Rate Limit"""
    if not coin_ids:
        return {}
    
    api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies={currency}"
    
    for i in range(3): # Thử tối đa 3 lần
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2 * (i + 1)) # Đợi 2s, 4s, 6s...
                continue
            else:
                return {} # Lỗi khác thì trả về rỗng để không crash app
        except:
            time.sleep(1)
            continue
    return {}

# --- KHỞI TẠO SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Coin", "Số lượng"])

# ==============================================================================
# --- SIDEBAR ---
# ==============================================================================
with st.sidebar:
    st.title("Crypto AI Analyst")
    st.markdown("---")
    menu = st.radio("Menu Chính", ["📊 Dashboard Dự báo", "💼 Quản lý Danh mục"], index=0)
    st.markdown("---")
    st.info("💡 **Mẹo:** Nhập đúng Coin ID (ví dụ: `bitcoin`, `ethereum`)")

# ==============================================================================
# --- TRANG 1: DASHBOARD DỰ BÁO ---
# ==============================================================================
if menu == "📊 Dashboard Dự báo":
    st.header("🔮 Phân Tích & Dự Báo Giá")
    
    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    
    with col1:
        coin_input = st.text_input("🔍 Nhập Coin ID", "bitcoin").strip().lower() # .strip() để xóa khoảng trắng thừa
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
                    status.update(label="❌ Lỗi dữ liệu hoặc sai tên Coin!", state="error")
                    st.error("Không tải được dữ liệu. Kiểm tra lại tên Coin ID.")
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
    
    # --- PHẦN 1: THÊM COIN ---
    with st.expander("➕ Thêm Coin vào Danh mục", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with c1:
            new_coin = st.text_input("Coin ID (vd: monad)", key="new_coin").strip().lower()
        with c2:
            new_qty = st.number_input("Số lượng", min_value=0.0, format="%.6f", key="new_qty")
        with c3:
            if st.button("Thêm"):
                if new_coin and new_qty > 0:
                    if new_coin in st.session_state.portfolio["Coin"].values:
                        st.session_state.portfolio.loc[st.session_state.portfolio["Coin"] == new_coin, "Số lượng"] += new_qty
                    else:
                        new_row = pd.DataFrame([{"Coin": new_coin, "Số lượng": new_qty}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    st.success(f"Đã cập nhật {new_coin}")
                    time.sleep(0.5)
                    st.rerun()

    # --- PHẦN 2: HIỂN THỊ TỔNG QUAN ---
    if not st.session_state.portfolio.empty:
        currency = predictor.currency
        coin_ids = st.session_state.portfolio["Coin"].unique()
        
        # --- FIX: Gọi hàm an toàn thay vì gọi trực tiếp ---
        with st.spinner("Đang cập nhật giá thị trường..."):
            price_data = get_current_prices_safe(list(coin_ids), currency)

        port_df = st.session_state.portfolio.copy()
        current_prices = []
        total_values = []
        
        for _, row in port_df.iterrows():
            cid = row["Coin"]
            # Lấy giá an toàn, nếu lỗi trả về 0
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

        # Fix cảnh báo use_container_width
        st.dataframe(port_df, use_container_width=True)
        
        st.markdown("---")

        # --- PHẦN 3: CỐ VẤN DANH MỤC AI ---
        with st.container():
            st.subheader("🤖 Cố vấn Danh mục AI")
            st.info("AI sẽ phân tích toàn bộ danh mục để tìm ra coin tiềm năng.")
            
            advisor_cols = st.columns([1, 1, 2])
            with advisor_cols[0]:
                advisor_days = st.number_input("Số ngày dự báo", min_value=1, max_value=90, value=7)
            with advisor_cols[1]:
                st.write("")
                if st.button("🔍 Đưa ra lời khuyên"):
                    results = []
                    total_predicted_value = 0
                    progress_bar = st.progress(0, text="Bắt đầu phân tích...")

                    for i, row in port_df.iterrows():
                        coin_id = row["Coin"]
                        quantity = row["Số lượng"]
                        
                        progress_bar.progress((i+1)/len(port_df), text=f"Đang phân tích {coin_id.upper()}...")
                        
                        # Fetch history cũng có cơ chế retry nên an toàn
                        df_coin = predictor.fetch_history(coin_id, days=365)
                        if df_coin is not None:
                            dates, preds, _, _, _ = predictor.predict_prophet(df_coin, days_ahead=advisor_days)
                            predicted_price = preds[-1]
                            predicted_value = quantity * predicted_price
                            percent_change = ((predicted_price - row["Giá Hiện Tại"]) / row["Giá Hiện Tại"]) * 100 if row["Giá Hiện Tại"] > 0 else 0
                            
                            results.append({
                                "Coin": coin_id,
                                "Giá Hiện Tại": row["Giá Hiện Tại"],
                                f"Giá Dự Báo ({advisor_days} ngày)": predicted_price,
                                "Thay Đổi (%)": percent_change
                            })
                            total_predicted_value += predicted_value
                        
                        # Quan trọng: Nghỉ 1 chút để không bị block IP
                        time.sleep(1.0) 
                    
                    progress_bar.empty()

                    if results:
                        st.markdown("#### Bảng phân tích chi tiết")
                        result_df = pd.DataFrame(results).sort_values(by="Thay Đổi (%)", ascending=False)
                        st.dataframe(result_df, use_container_width=True, 
                                   column_config={"Thay Đổi (%)": st.column_config.NumberColumn(format="%.2f%%")})

                        top_gainer = result_df.iloc[0]
                        overall_change = ((total_predicted_value - total_net_worth) / total_net_worth) * 100 if total_net_worth > 0 else 0

                        st.success(f"**Coin tiềm năng nhất:** `{top_gainer['Coin'].upper()}` (+{top_gainer['Thay Đổi (%)']:.2f}%).")
                        if overall_change > 0:
                            st.info(f"Tổng danh mục dự kiến **TĂNG {overall_change:.2f}%**.")
                        else:
                            st.warning(f"Tổng danh mục dự kiến **GIẢM {overall_change:.2f}%**.")
        
        # --- PHẦN 4: SOI CHART RIÊNG ---
        st.markdown("---")
        st.subheader("🔬 Soi chart chi tiết")
        c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with c_sel1:
            selected_coin = st.selectbox("Chọn Coin:", port_df["Coin"].unique())
        with c_sel2:
            forecast_days = st.number_input("Ngày dự báo", 1, 365, 7)
        with c_sel3:
            if st.button("🔍 Phân tích"):
                with st.spinner(f"Đang soi {selected_coin}..."):
                    df_coin = predictor.fetch_history(selected_coin, days=max(90, forecast_days + 30))
                    if df_coin is not None:
                        dates, preds, bounds, _, mape = predictor.predict_prophet(df_coin, days_ahead=forecast_days)
                        st.pyplot(predictor.visualize_prophet(df_coin, dates, preds, bounds, selected_coin))
                    else:
                        st.error("Lỗi dữ liệu.")
            
        with st.expander("🗑 Xóa Coin khỏi danh mục"):
            del_coin = st.selectbox("Chọn để xóa", port_df["Coin"].unique())
            if st.button("Xác nhận xóa"):
                st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio["Coin"] != del_coin]
                st.rerun()
            
    else:
        st.info("👈 Danh mục trống. Hãy thêm coin mới!")