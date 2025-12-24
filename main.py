import streamlit as st
import pandas as pd
import requests
from dudoangia import CoinGeckoPredictor

# Initialize the predictor
predictor = CoinGeckoPredictor()

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("Trình Quản lý Danh mục và Dự báo Giá Crypto")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Chọn một trang", ["Dự báo giá", "Quản lý danh mục"])

# ==============================================================================
# --- PRICE PREDICTION PAGE ---
# ==============================================================================
if page == "Dự báo giá":
    st.header("🔮 Công cụ Dự báo Giá")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        coin_input = st.text_input("Nhập tên hoặc ID của Coin (ví dụ: bitcoin, ethereum)", "bitcoin")
        
    with col2:
        prediction_days = st.selectbox(
            "Chọn số ngày dự đoán",
            [1, 7, 30, 90, 365],
            index=1  # Default to 7 days
        )

    if st.button("Bắt đầu Dự báo"):
        if not coin_input:
            st.warning("Vui lòng nhập tên một coin.")
        else:
            coin_id = predictor.extract_coin_id(coin_input)
            if not coin_id:
                st.error("Không thể xác định ID của coin. Vui lòng thử lại.")
            else:
                with st.spinner(f"Đang tải dữ liệu lịch sử cho {coin_id}..."):
                    df = predictor.fetch_history(coin_id, days=max(365, prediction_days + 1))

                if df is None or df.empty:
                    st.error(f"Không thể lấy được dữ liệu cho {coin_id}. Coin có thể không được hỗ trợ hoặc có lỗi API.")
                else:
                    st.success(f"Đã tải xong dữ liệu. Giá hiện tại: ${df['price'].iloc[-1]:,.4f}")

                    # --- Run Prediction ---
                    if prediction_days == 1:
                        # Use Linear Regression for 1 day
                        with st.spinner("Đang chạy mô hình Linear Regression..."):
                            date, pred, _, score, mape = predictor.predict_linear(df, 1)
                            fig = predictor.visualize_linear(df, date, pred, _, coin_id)
                        
                        st.subheader(f"Kết quả dự đoán cho {coin_id.upper()} (1 ngày)")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Giá dự đoán", f"${pred:,.4f}")
                        col2.metric("Độ phù hợp (R²)", f"{score:.4f}")
                        col3.metric("Sai số trung bình (MAPE)", f"{mape:.2f}%")
                        
                        st.pyplot(fig)

                    else:
                        # Use Prophet for > 1 day
                        with st.spinner(f"Đang chạy mô hình Prophet AI cho {prediction_days} ngày..."):
                            dates, preds, bounds, _, mape = predictor.predict_prophet(df, prediction_days)
                            fig = predictor.visualize_prophet(df, dates, preds, bounds, coin_id)
                        
                        st.subheader(f"Kết quả dự đoán cho {coin_id.upper()} ({prediction_days} ngày)")
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        col1.metric("Dự đoán ngày cuối", f"${preds[-1]:,.4f}")
                        col2.metric("Sai số trung bình (MAPE)", f"{mape:.2f}%")
                        
                        # Display chart
                        st.pyplot(fig)
                        
                        # Display data table
                        st.subheader("Dữ liệu dự báo chi tiết")
                        forecast_df = pd.DataFrame({
                            "Ngày": dates,
                            "Giá dự đoán (yhat)": preds,
                            "Biên dưới (yhat_lower)": bounds[:, 0],
                            "Biên trên (yhat_upper)": bounds[:, 1]
                        })
                        st.dataframe(forecast_df)


# ==============================================================================
# --- PORTFOLIO MANAGEMENT PAGE ---
# ==============================================================================
elif page == "Quản lý danh mục":
    st.header("📈 Quản lý Danh mục Đầu tư")

    # Initialize portfolio in session state if it doesn't exist
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=["Coin", "Số lượng"])

    st.subheader("Thêm Coin mới vào Danh mục")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_coin = st.text_input("Tên Coin (ID)", key="new_coin")
    with col2:
        new_quantity = st.number_input("Số lượng", min_value=0.0, format="%.6f", key="new_quantity")
    with col3:
        st.write("&#8203;") # Whitespace to align button
        if st.button("Thêm vào Danh mục"):
            if new_coin and new_quantity > 0:
                # Check if coin already exists
                if new_coin in st.session_state.portfolio["Coin"].values:
                    st.session_state.portfolio.loc[st.session_state.portfolio["Coin"] == new_coin, "Số lượng"] += new_quantity
                else:
                    new_row = pd.DataFrame([{"Coin": new_coin, "Số lượng": new_quantity}])
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                st.success(f"Đã thêm {new_quantity} {new_coin} vào danh mục.")
            else:
                st.warning("Vui lòng nhập tên coin và số lượng hợp lệ.")
    
    st.subheader("Danh mục Hiện tại")

    if not st.session_state.portfolio.empty:
        portfolio_df = st.session_state.portfolio.copy()
        
        # --- Fetch current prices for portfolio ---
        total_value = 0
        price_list = []
        
        # Create a unique list of coins to fetch
        coins_to_fetch = portfolio_df["Coin"].unique()
        
        try:
            # Efficiently fetch prices in one go
            currency = predictor.currency
            api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coins_to_fetch)}&vs_currencies={currency}"
            response = requests.get(api_url).json()

            for index, row in portfolio_df.iterrows():
                coin_id = row["Coin"]
                price = response.get(coin_id, {}).get(currency, 0)
                value = row["Số lượng"] * price
                price_list.append(price)
                total_value += value

            portfolio_df[f"Giá hiện tại ({currency.upper()})"] = price_list
            portfolio_df[f"Tổng giá trị ({currency.upper()})"] = portfolio_df["Số lượng"] * portfolio_df[f"Giá hiện tại ({currency.upper()})"]
            
            # --- Display Metrics ---
            st.metric(f"Tổng giá trị Danh mục ({currency.upper()})", f"${total_value:,.2f}")

            # --- Display Portfolio Table ---
            st.dataframe(portfolio_df)

        except Exception as e:
            st.error(f"Lỗi khi tải giá: {e}")
            st.dataframe(st.session_state.portfolio) # Show basic portfolio if API fails
    else:
        st.info("Danh mục của bạn đang trống. Thêm một coin để bắt đầu.")

    st.subheader("Xóa Coin khỏi Danh mục")
    if not st.session_state.portfolio.empty:
        coin_to_delete = st.selectbox("Chọn Coin để Xóa", st.session_state.portfolio["Coin"])
        if st.button("Xóa Coin"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio["Coin"] != coin_to_delete]
            st.experimental_rerun()
    else:
        st.write("Không có coin nào để xóa.")