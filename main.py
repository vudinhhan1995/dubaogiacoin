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

# --- CSS TÙY CHỈNH (Làm đẹp giao diện) ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    h1, h2, h3 {
        color: #0e1117; 
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo bộ dự báo
@st.cache_resource
def get_predictor():
    return CoinGeckoPredictor()

predictor = get_predictor()

# --- KHỞI TẠO SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Coin", "Số lượng"])

# ==============================================================================
# --- SIDEBAR (THANH ĐIỀU HƯỚNG) ---
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2272/2272825.png", width=80)
    st.title("Crypto AI Analyst")
    st.markdown("---")
    
    menu = st.radio(
        "Menu Chính", 
        ["📊 Dashboard Dự báo", "💼 Quản lý Danh mục"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Mẹo:** Nhập đúng Coin ID (ví dụ: `bitcoin`, `ethereum`, `monad`) để có kết quả chính xác nhất.")

# ==============================================================================
# --- TRANG 1: DASHBOARD DỰ BÁO (Dành cho soi chart) ---
# ==============================================================================
if menu == "📊 Dashboard Dự báo":
    st.header("🔮 Phân Tích & Dự Báo Giá")
    
    # Chia cột cho Input (dùng vertical_alignment để căn đáy)
    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    
    with col1:
        coin_input = st.text_input("🔍 Nhập Coin ID", "bitcoin", help="Ví dụ: bitcoin, dogecoin, solana")
    with col2:
        prediction_days = st.selectbox("⏳ Khung thời gian", [1, 7, 30, 90], index=1)
    with col3:
        btn_predict = st.button("🚀 Chạy Phân Tích", type="primary")

    if btn_predict:
        coin_id = predictor.extract_coin_id(coin_input)
        if not coin_id:
            st.error("❌ Coin ID không hợp lệ!")
        else:
            with st.status(f"🤖 Đang phân tích dữ liệu {coin_id.upper()}...", expanded=True) as status:
                st.write("1. Kết nối API CoinGecko...")
                df = predictor.fetch_history(coin_id, days=max(365, prediction_days + 30))
                
                if df is None or df.empty:
                    status.update(label="❌ Lỗi dữ liệu!", state="error")
                    st.error("Không tải được dữ liệu.")
                else:
                    st.write("2. Làm sạch dữ liệu & Lọc nhiễu...")
                    current_price = df['price'].iloc[-1]
                    
                    # Xử lý dự báo
                    if prediction_days == 1:
                        st.write("3. Chạy mô hình Linear Regression...")
                        date, pred, model, score, mape = predictor.predict_linear(df, 1)
                        fig = predictor.visualize_linear(df, date, pred, model, coin_id)
                        
                        # Tính delta
                        delta = ((pred - current_price) / current_price) * 100
                        
                        status.update(label="✅ Hoàn tất!", state="complete", expanded=False)
                        
                        # HIỂN THỊ KẾT QUẢ
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Giá Hiện Tại", f"${current_price:,.4f}")
                        m2.metric("Dự Báo (1 Ngày)", f"${pred:,.4f}", f"{delta:.2f}%")
                        m3.metric("Độ Chính Xác (R²)", f"{score:.2f}")
                        
                        st.pyplot(fig)
                        
                    else:
                        st.write("3. Chạy mô hình Prophet AI (Facebook)...")
                        dates, preds, bounds, _, mape = predictor.predict_prophet(df, prediction_days)
                        fig = predictor.visualize_prophet(df, dates, preds, bounds, coin_id)
                        
                        # Tính delta ngày cuối
                        last_pred = preds[-1]
                        delta = ((last_pred - current_price) / current_price) * 100
                        
                        status.update(label="✅ Hoàn tất!", state="complete", expanded=False)
                        
                        # HIỂN THỊ KẾT QUẢ
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Giá Hiện Tại", f"${current_price:,.4f}")
                        m2.metric(f"Mục Tiêu ({prediction_days} Ngày)", f"${last_pred:,.4f}", f"{delta:.2f}%")
                        m3.metric("Sai Số (MAPE)", f"{mape:.2f}%", delta_color="inverse") # MAPE càng thấp càng tốt
                        
                        st.pyplot(fig)
                        
                        with st.expander("📄 Xem dữ liệu chi tiết"):
                            st.dataframe(pd.DataFrame({
                                "Ngày": dates,
                                "Dự đoán ($)": preds,
                                "Thấp nhất ($)": bounds[:, 0],
                                "Cao nhất ($)": bounds[:, 1]
                            }))

# ==============================================================================
# --- TRANG 2: QUẢN LÝ DANH MỤC (Tích hợp Dự báo) ---
# ==============================================================================
elif menu == "💼 Quản lý Danh mục":
    st.header("📈 Portfolio & Smart Alerts")
    
    # --- PHẦN 1: THÊM COIN ---
    with st.expander("➕ Thêm Coin vào Danh mục", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            new_coin = st.text_input("Coin ID (vd: monad)", key="new_coin")
        with c2:
            new_qty = st.number_input("Số lượng", min_value=0.0, format="%.6f", key="new_qty")
        with c3:
            st.write("##")
            if st.button("Thêm"):
                if new_coin and new_qty > 0:
                    # Logic thêm coin
                    if new_coin in st.session_state.portfolio["Coin"].values:
                        st.session_state.portfolio.loc[st.session_state.portfolio["Coin"] == new_coin, "Số lượng"] += new_qty
                    else:
                        new_row = pd.DataFrame([{"Coin": new_coin, "Số lượng": new_qty}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    st.success(f"Đã thêm {new_coin}")
                    st.rerun()

    # --- PHẦN 2: HIỂN THỊ TỔNG QUAN ---
    if not st.session_state.portfolio.empty:
        # Lấy giá hiện tại cho toàn bộ danh mục
        currency = predictor.currency
        coin_ids = st.session_state.portfolio["Coin"].unique()
        
        try:
            with st.spinner("Đang cập nhật giá thị trường..."):
                api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies={currency}"
                response = requests.get(api_url, timeout=10).json()

            # Tính toán bảng
            port_df = st.session_state.portfolio.copy()
            current_prices = []
            total_values = []
            
            for _, row in port_df.iterrows():
                cid = row["Coin"]
                price = response.get(cid, {}).get(currency, 0)
                current_prices.append(price)
                total_values.append(price * row["Số lượng"])
                
            port_df["Giá Hiện Tại"] = current_prices
            port_df["Tổng Giá Trị"] = total_values
            
            total_net_worth = sum(total_values)
            
            # Metric tổng quan
            st.markdown("### 💰 Tổng Tài Sản")
            st.metric("Net Worth", f"${total_net_worth:,.2f}", delta=None)
            
            # Bảng danh mục
            st.dataframe(port_df, use_container_width=True)
            
            st.markdown("---")
            
            # --- PHẦN 3: TÍNH NĂNG DỰ BÁO TÍCH HỢP (THÔNG MINH) ---
            st.subheader("🤖 AI Phân Tích Danh Mục (Dự báo 7 ngày)")
            st.info("Chọn một coin trong danh mục để AI chạy phân tích xu hướng.")
            
            selected_coin = st.selectbox("Chọn Coin để soi:", port_df["Coin"].unique())
            
            if st.button(f"🔍 Phân tích xu hướng {selected_coin.upper()}"):
                with st.spinner(f"AI đang tính toán đường đi của {selected_coin}..."):
                    # 1. Lấy dữ liệu
                    df_coin = predictor.fetch_history(selected_coin, days=180) # Lấy 6 tháng cho nhanh
                    
                    if df_coin is not None:
                        # 2. Chạy Prophet 7 ngày
                        dates, preds, bounds, _, _ = predictor.predict_prophet(df_coin, days_ahead=7)
                        
                        cur_p = df_coin['price'].iloc[-1]
                        fut_p = preds[-1]
                        percent_change = ((fut_p - cur_p) / cur_p) * 100
                        
                        # 3. Hiển thị Card thông tin
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            st.markdown(f"### {selected_coin.upper()}")
                            if percent_change > 0:
                                st.success(f"Xu hướng: TĂNG 📈")
                            else:
                                st.error(f"Xu hướng: GIẢM 📉")
                                
                            st.metric("Giá dự kiến (7 ngày)", f"${fut_p:,.4f}", f"{percent_change:.2f}%")
                            st.write(f"Khoảng giá: ${bounds[-1][0]:,.2f} - ${bounds[-1][1]:,.2f}")

                        with col_b:
                            # Vẽ biểu đồ nhỏ gọn
                            fig_mini = predictor.visualize_prophet(df_coin, dates, preds, bounds, selected_coin)
                            st.pyplot(fig_mini)
                    else:
                        st.error("Không đủ dữ liệu để phân tích coin này.")
            
            # Xóa coin (Đặt cuối cho gọn)
            with st.expander("🗑 Xóa Coin"):
                del_coin = st.selectbox("Chọn để xóa", port_df["Coin"].unique())
                if st.button("Xác nhận xóa"):
                    st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio["Coin"] != del_coin]
                    st.rerun()

        except Exception as e:
            st.error(f"Lỗi kết nối API: {e}")
            st.dataframe(st.session_state.portfolio)
            
    else:
        st.info("👈 Danh mục trống. Hãy thêm coin mới ở phần trên!")