import requests
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
import sys
import time

# Tắt các cảnh báo không cần thiết của Prophet và Pandas
warnings.filterwarnings('ignore')
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class CoinGeckoPredictor:
    def __init__(self):
        self.base_api = "https://api.coingecko.com/api/v3"
        self.currency = 'usd'
        # Cấu hình hiển thị biểu đồ đẹp hơn
        plt.style.use('bmh')

    def extract_coin_id(self, url):
        """Trích xuất coin_id từ URL CoinGecko."""
        try:
            match = re.search(r'/coins/([^/?]+)', url)
            if match:
                return match.group(1)
            if not "/" in url:
                return url.lower().strip()
            return None
        except Exception as e:
            print(f"⚠️ Lỗi phân tích URL: {e}")
            return None

    def fetch_history(self, coin_id, days=365, max_retries=3):
        """Lấy dữ liệu giá OHLC từ API CoinGecko với cơ chế retry."""
        url = f"{self.base_api}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': self.currency,
            'days': days,
            'interval': 'daily'
        }
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Đang tải dữ liệu '{coin_id}' ({days} ngày)... Lần thử {attempt + 1}/{max_retries}")
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 10  # Đợi 10s, 20s, 30s...
                    print(f"⏳ Rate Limit. Đang đợi {wait_time}s để thử lại...")
                    time.sleep(wait_time)
                    continue  # Thử lại
                
                if response.status_code == 200:
                    data = response.json()
                    prices = data.get('prices', [])
                    if not prices:
                        print("❌ Dữ liệu trống.")
                        return None
                    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
                    df = df.dropna()
                    return df
                
                if response.status_code == 404:
                    print(f"❌ Không tìm thấy coin ID: '{coin_id}'. Kiểm tra lại link/tên.")
                    return None
                
                print(f"⚠️ Lỗi API không mong muốn: {response.status_code}")
                return None

            except requests.exceptions.RequestException as e:
                print(f"❌ Lỗi mạng: {e}")
                return None
        
        print("❌ Đã thử lại nhiều lần nhưng thất bại trong việc lấy dữ liệu.")
        return None

    def remove_outliers(self, df, column='price', window=14, sigma=3.0): # Tăng sigma lên 3.0
        """Lọc nhiễu nhẹ nhàng hơn để giữ lại biến động thị trường quan trọng."""
        df_clean = df.copy()
        rolling_mean = df_clean[column].rolling(window=window).mean()
        rolling_std = df_clean[column].rolling(window=window).std()
        
        upper_bound = rolling_mean + (sigma * rolling_std)
        lower_bound = rolling_mean - (sigma * rolling_std)
        
        mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
        
        # Fix SettingWithCopyWarning: Giữ lại 14 ngày đầu tiên
        mask.iloc[:window] = True
        
        filtered_df = df_clean[mask]
        removed_count = len(df) - len(filtered_df)
        if removed_count > 0:
            print(f"🧹 Đã lọc bỏ {removed_count} điểm dữ liệu nhiễu (Outliers).")
        
        return filtered_df

    def predict_linear(self, df, days_ahead=1):
        """Dự đoán Linear Regression với train/test split cho MAPE."""
        df_clean = self.remove_outliers(df)
        
        X = df_clean[['date_ordinal']]
        y = df_clean['price']
        
        # --- Train/Test Split để tính MAPE ---
        split_size = int(len(df_clean) * 0.9)
        if split_size > 1:
            X_train, X_test = X[:split_size], X[split_size:]
            y_train, y_test = y[:split_size], y[split_size:]

            model_test = LinearRegression()
            model_test.fit(X_train, y_train)
            y_pred_test = model_test.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        else:
            mape = 0.0 # Không đủ dữ liệu để test

        # --- Retrain trên toàn bộ dữ liệu để dự đoán tương lai ---
        model_final = LinearRegression()
        model_final.fit(X, y)
        
        last_date = df['date'].iloc[-1]
        next_date = last_date + timedelta(days=days_ahead)
        next_date_ordinal = np.array([[next_date.toordinal()]])
        
        predicted_price = model_final.predict(next_date_ordinal)[0]
        predicted_price = max(0, predicted_price)
        score = model_final.score(X, y)
        
        return next_date, predicted_price, model_final, score, mape

    def predict_prophet(self, df, days_ahead=7):
        """Dự đoán Prophet với cross-validation cho MAPE và hyperparameter tuning."""
        prophet_df = df[['date', 'price']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['y'] = np.log(prophet_df['y'])
        
        use_yearly = (df['date'].max() - df['date'].min()).days > 300
        
        # --- BƯỚC 1: CROSS VALIDATION (Kiểm tra độ chính xác) ---
        mape = 0.0
        cut_off = len(prophet_df) - 30 
        if cut_off > 30: # Chỉ test nếu dữ liệu đủ dài (hơn 60 ngày)
            train_df = prophet_df.iloc[:cut_off]
            test_df = prophet_df.iloc[cut_off:]
            
            m_test = Prophet(
                daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=use_yearly,
                changepoint_prior_scale=0.15, changepoint_range=0.9
            )
            m_test.fit(train_df)
            forecast_test = m_test.predict(test_df)
            
            y_test_true = np.exp(test_df['y'])
            y_test_pred = np.exp(forecast_test['yhat'])
            mape = mean_absolute_percentage_error(y_test_true, y_test_pred) * 100
        else:
            print("ℹ️ Dữ liệu quá ngắn để thực hiện cross-validation, MAPE sẽ được báo cáo là 0.")

        # --- BƯỚC 2: TRAIN FULL ĐỂ DỰ ĐOÁN TƯƠNG LAI ---
        # Tinh chỉnh hyperparameter dựa trên độ dài dự đoán
        changepoint_scale = 0.05 if days_ahead >= 30 else 0.15
        
        model_final = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=use_yearly,
            changepoint_prior_scale=changepoint_scale,
            changepoint_range=0.9
        )
        model_final.fit(prophet_df)
        
        future = model_final.make_future_dataframe(periods=days_ahead)
        forecast = model_final.predict(future)
        
        future_forecast = forecast.tail(days_ahead).copy()
        
        # Inverse Log & Clipping
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            future_forecast[col] = np.exp(future_forecast[col])
            future_forecast[col] = future_forecast[col].clip(lower=0)

        ds_list = future_forecast['ds'].tolist()
        yhat_list = future_forecast['yhat'].tolist()
        bounds = future_forecast[['yhat_lower', 'yhat_upper']].values
        
        return ds_list, yhat_list, bounds, model_final, mape

    def visualize_prophet(self, df, future_dates, predictions, bounds, coin_id):
        fig = plt.figure(figsize=(14, 7))
        plt.plot(df['date'], df['price'], 'k-', label='Lịch sử giá', alpha=0.5, linewidth=1)
        
        dates = pd.to_datetime(future_dates)
        pred_arr = np.array(predictions)
        
        plt.plot(dates, pred_arr, color='#007acc', label='Dự đoán (Prophet)', linewidth=2)
        plt.fill_between(dates, bounds[:, 0], bounds[:, 1], color='#007acc', alpha=0.2, label='Vùng dao động (80%)')
        
        last_date = dates[-1]
        last_price = pred_arr[-1]
        plt.scatter([last_date], [last_price], color='red', s=100, zorder=5)
        plt.annotate(f"${last_price:,.4f}", (last_date, last_price), 
                     xytext=(10, 10), textcoords='offset points', fontweight='bold', color='red')

        plt.title(f"DỰ ĐOÁN GIÁ: {coin_id.upper()} (Mô hình Log-Prophet Tuned)", fontsize=16, fontweight='bold')
        plt.xlabel("Thời gian")
        plt.ylabel(f"Giá ({self.currency.upper()})")
        plt.legend()
        plt.tight_layout()
        return fig

    def visualize_linear(self, df, next_date, predicted_price, model, coin_id):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['price'], 'o', markersize=3, label='Giá thực tế', color='gray', alpha=0.5)
        
        trend_X = df[['date_ordinal']]
        trend_y = model.predict(trend_X)
        plt.plot(df['date'], trend_y, 'r--', linewidth=2, label='Đường xu hướng')
        
        plt.scatter([next_date], [predicted_price], color='green', s=150, marker='*', zorder=5)
        plt.annotate(f"${predicted_price:,.4f}", (next_date, predicted_price), 
                     xytext=(10, 10), textcoords='offset points', fontweight='bold', color='green')
        
        plt.title(f"DỰ ĐOÁN GIÁ: {coin_id.upper()} (Linear Regression)", fontsize=16)
        plt.xlabel("Thời gian")
        plt.ylabel(f"Giá ({self.currency.upper()})")
        plt.legend()
        plt.tight_layout()
        return fig