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

# Tắt các cảnh báo không cần thiết của Prophet và Pandas
warnings.filterwarnings('ignore')
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class CoinGeckoPredictor:
    def __init__(self):
        self.base_api = "https://api.coingecko.com/api/v3"
        # Cấu hình hiển thị biểu đồ đẹp hơn
        plt.style.use('bmh') # Sử dụng style 'bmh' cho biểu đồ chuyên nghiệp hơn

    def extract_coin_id(self, url):
        """Trích xuất coin_id từ URL CoinGecko."""
        try:
            # Hỗ trợ cả link có 'en/coins/' và link rút gọn nếu có
            match = re.search(r'/coins/([^/?]+)', url)
            if match:
                return match.group(1)
            # Fallback nếu user nhập trực tiếp tên coin (vd: bitcoin)
            if not "/" in url:
                return url.lower().strip()
            return None
        except Exception as e:
            print(f"⚠️ Lỗi phân tích URL: {e}")
            return None

    def fetch_history(self, coin_id, days=365):
        """Lấy dữ liệu giá OHLC từ API CoinGecko."""
        print(f"🔄 Đang tải dữ liệu '{coin_id}' ({days} ngày)...")
        url = f"{self.base_api}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        try:
            # Thêm timeout để tránh treo chương trình
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 429:
                print("⛔ Rate Limit: Bạn đang gửi quá nhiều yêu cầu. Vui lòng đợi 30s.")
                return None
            elif response.status_code == 404:
                print(f"❌ Không tìm thấy coin ID: '{coin_id}'. Kiểm tra lại link/tên.")
                return None
            elif response.status_code != 200:
                print(f"⚠️ Lỗi API: {response.status_code}")
                return None

            data = response.json()
            prices = data.get('prices', [])
            
            if not prices:
                print("❌ Dữ liệu trống.")
                return None

            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
            
            # Loại bỏ các dòng dữ liệu lỗi (nếu có)
            df = df.dropna()
            
            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi kết nối mạng: {e}")
            return None
        except Exception as e:
            print(f"❌ Lỗi xử lý dữ liệu: {e}")
            return None

    def remove_outliers(self, df, column='price', window=14, sigma=2.5):
        """
        TỐI ƯU: Loại bỏ nhiễu (outliers) bằng phương pháp Rolling Statistics.
        Giúp model không bị méo bởi các cú 'râu nến' (flash crash/pump) ảo.
        """
        df_clean = df.copy()
        # Tính trung bình trượt và độ lệch chuẩn trượt
        rolling_mean = df_clean[column].rolling(window=window).mean()
        rolling_std = df_clean[column].rolling(window=window).std()
        
        # Xác định biên trên/dưới (Band Bollinger)
        upper_bound = rolling_mean + (sigma * rolling_std)
        lower_bound = rolling_mean - (sigma * rolling_std)
        
        # Giữ lại giá nằm trong biên HOẶC dữ liệu 14 ngày đầu (chưa đủ rolling)
        mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
        mask.iloc[:window] = True # Luôn giữ dữ liệu gốc ban đầu
        
        filtered_df = df_clean[mask]
        
        removed_count = len(df) - len(filtered_df)
        if removed_count > 0:
            print(f"🧹 Đã lọc bỏ {removed_count} điểm dữ liệu nhiễu (Outliers) để tăng độ chính xác.")
            
        return filtered_df

    def predict_linear(self, df, days_ahead=1):
        """Dự đoán Linear Regression cơ bản có lọc nhiễu."""
        # Lọc nhiễu trước khi train
        df_clean = self.remove_outliers(df)
        
        X = df_clean[['date_ordinal']]
        y = df_clean['price']
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_date = df['date'].iloc[-1]
        next_date = last_date + timedelta(days=days_ahead)
        next_date_ordinal = np.array([[next_date.toordinal()]])
        
        predicted_price = model.predict(next_date_ordinal)[0]
        predicted_price = max(0, predicted_price)
        
        # Tính toán sai số trung bình (MAPE) trên tập train
        y_pred_train = model.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred_train) * 100
        score = model.score(X, y)
        
        return next_date, predicted_price, model, score, mape

    def predict_prophet(self, df, days_ahead=7):
        """
        Dự đoán Prophet tối ưu với Log Transform + Outlier Removal + Tuned Hyperparams.
        """
        # 1. Lọc nhiễu để đường xu hướng chuẩn hơn
        df_clean = self.remove_outliers(df)
        
        # 2. Chuẩn bị dữ liệu
        prophet_df = df_clean[['date', 'price']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Log Transform
        prophet_df['y'] = np.log(prophet_df['y'])
        
        # Cấu hình seasonality
        data_days = (df['date'].max() - df['date'].min()).days
        use_yearly = data_days > 300
        
        # 3. Tinh chỉnh Hyperparameters (Tối ưu cho Crypto)
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=use_yearly,
            # Tăng độ nhạy với xu hướng (mặc định 0.05 -> 0.15)
            changepoint_prior_scale=0.15,
            # Cho phép thay đổi xu hướng ở cả những ngày gần nhất (mặc định 0.8 -> 0.9)
            changepoint_range=0.9
        )
        
        model.fit(prophet_df)
        
        # Tính toán sai số mô hình (MAPE) trên dữ liệu lịch sử
        # (Lấy giá trị fit vs giá trị thực tế để xem model học tốt thế nào)
        forecast_history = model.predict(prophet_df)
        y_true = np.exp(prophet_df['y'])
        y_pred_history = np.exp(forecast_history['yhat'])
        mape = mean_absolute_percentage_error(y_true, y_pred_history) * 100
        
        # Dự đoán tương lai
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        future_forecast = forecast.tail(days_ahead).copy()
        
        # Inverse Log & Clipping
        future_forecast['yhat'] = np.exp(future_forecast['yhat'])
        future_forecast['yhat_lower'] = np.exp(future_forecast['yhat_lower'])
        future_forecast['yhat_upper'] = np.exp(future_forecast['yhat_upper'])
        
        cols = ['yhat', 'yhat_lower', 'yhat_upper']
        for col in cols:
            future_forecast[col] = future_forecast[col].clip(lower=0)

        ds_list = future_forecast['ds'].tolist()
        yhat_list = future_forecast['yhat'].tolist()
        bounds = future_forecast[['yhat_lower', 'yhat_upper']].values
        
        return ds_list, yhat_list, bounds, model, mape

    def visualize_prophet(self, df, future_dates, predictions, bounds, coin_id):
        plt.figure(figsize=(14, 7))
        
        plt.plot(df['date'], df['price'], 'k-', label='Lịch sử giá', alpha=0.5, linewidth=1)
        
        dates = pd.to_datetime(future_dates)
        pred_arr = np.array(predictions)
        lower_arr = bounds[:, 0]
        upper_arr = bounds[:, 1]
        
        plt.plot(dates, pred_arr, color='#007acc', label='Dự đoán (Prophet)', linewidth=2)
        plt.fill_between(dates, lower_arr, upper_arr, color='#007acc', alpha=0.2, label='Vùng dao động (80%)')
        
        last_date = dates[-1]
        last_price = pred_arr[-1]
        plt.scatter([last_date], [last_price], color='red', s=100, zorder=5)
        plt.annotate(f"${last_price:,.4f}", (last_date, last_price), 
                     xytext=(10, 10), textcoords='offset points', fontweight='bold', color='red')

        plt.title(f"DỰ ĐOÁN GIÁ: {coin_id.upper()} (Mô hình Log-Prophet Tuned)", fontsize=16, fontweight='bold')
        plt.xlabel("Thời gian")
        plt.ylabel("Giá (USD)")
        plt.legend()
        plt.tight_layout()
        print("\n📊 Đang hiển thị biểu đồ...")
        plt.show()

    def visualize_linear(self, df, next_date, predicted_price, model, coin_id):
        plt.figure(figsize=(12, 6))
        
        plt.plot(df['date'], df['price'], 'o', markersize=3, label='Giá thực tế', color='gray', alpha=0.5)
        
        trend_X = df[['date_ordinal']]
        trend_y = model.predict(trend_X)
        plt.plot(df['date'], trend_y, 'r--', linewidth=2, label='Đường xu hướng')
        
        plt.scatter([next_date], [predicted_price], color='green', s=150, marker='*', zorder=5)
        plt.annotate(f"${predicted_price:,.4f}", (next_date, predicted_price), 
                     xytext=(10, 10), textcoords='offset points', fontweight='bold', color='green')
        
        plt.title(f"DỰ ĐOÁN GIÁ: {coin_id.upper()} (Linear Regression)", fontsize=16)
        plt.xlabel("Thời gian")
        plt.ylabel("Giá (USD)")
        plt.legend()
        plt.tight_layout()
        print("\n📊 Đang hiển thị biểu đồ...")
        plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    bot = CoinGeckoPredictor()
    
    print("\n" + "="*50)
    print("💎 CÔNG CỤ DỰ ĐOÁN GIÁ CRYPTO PRO (V3.0) 💎")
    print("="*50)
    print("• Tính năng mới: Tự động lọc nhiễu (Remove Outliers).")
    print("• Tính năng mới: Hiển thị sai số trung bình (MAPE).")
    print("• Tối ưu Log-Prophet cho độ chính xác cao nhất.")
    print("-" * 50)
    
    while True:
        try:
            url_input = input("\n👉 Nhập link/tên Coin (vd: bitcoin, monad): ").strip()
            
            if url_input.lower() in ['exit', 'quit', 'thoat']:
                print("👋 Tạm biệt!")
                break
                
            if not url_input: continue

            coin_id = bot.extract_coin_id(url_input)
            if not coin_id: continue

            df = bot.fetch_history(coin_id)
            if df is None: continue

            print("\n🔮 Chọn chế độ dự đoán:")
            print("  1. Ngắn hạn (Linear Regression - 1 ngày)")
            print("  2. Trung hạn (Prophet AI - 7 ngày)")
            print("  3. Dài hạn (Prophet AI - 30 ngày)")
            print("  4. Tùy chỉnh số ngày")
            
            choice = input("👉 Lựa chọn (Mặc định 2): ").strip()
            
            days_ahead = 7
            model_type = "prophet"
            
            if choice == "1":
                days_ahead = 1
                model_type = "linear"
            elif choice == "3":
                days_ahead = 30
            elif choice == "4":
                try:
                    d = int(input("   Nhập số ngày (1-365): "))
                    days_ahead = max(1, min(365, d))
                    model_type = "prophet" if days_ahead > 1 else "linear"
                except:
                    print("⚠️ Số ngày không hợp lệ, dùng mặc định 7 ngày.")
            
            current_price = df['price'].iloc[-1]
            print(f"\n💵 Giá hiện tại: ${current_price:,.4f}")
            
            if model_type == "prophet":
                print(f"🧠 Đang training & backtesting mô hình ({days_ahead} ngày)...")
                dates, preds, bounds, _, mape = bot.predict_prophet(df, days_ahead)
                
                print(f"\n🎯 ĐỘ CHÍNH XÁC MÔ HÌNH (MAPE): {mape:.2f}%")
                if mape < 5: print("   (Đánh giá: Rất tốt ✅)")
                elif mape < 10: print("   (Đánh giá: Tốt 🆗)")
                else: print("   (Đánh giá: Biến động mạnh ⚠️)")

                print(f"\n📋 KẾT QUẢ DỰ BÁO ({dates[0].strftime('%d/%m')} - {dates[-1].strftime('%d/%m')}):")
                print("-" * 65)
                print(f"{'NGÀY':<12} | {'GIÁ DỰ ĐOÁN':<15} | {'THAY ĐỔI':<10} | {'VÙNG GIÁ (MIN-MAX)':<20}")
                print("-" * 65)
                
                for d, p, b in zip(dates, preds, bounds):
                    change = ((p - current_price) / current_price) * 100
                    date_str = d.strftime('%d/%m/%Y')
                    change_str = f"{change:+.2f}%"
                    
                    print(f"{date_str:<12} | ${p:<14,.4f} | {change_str:<10} | ${b[0]:,.2f} - ${b[1]:,.2f}")
                
                print("-" * 65)
                bot.visualize_prophet(df, dates, preds, bounds, coin_id)
                
            else:
                print(f"🧠 Đang tính toán Linear Regression...")
                date, pred, _, score, mape = bot.predict_linear(df, days_ahead)
                change = ((pred - current_price) / current_price) * 100
                
                print(f"\n🎯 Dự đoán ngày {date.strftime('%d/%m/%Y')}:")
                print(f"   Giá: ${pred:,.4f} ({change:+.2f}%)")
                print(f"   Sai số trung bình (MAPE): {mape:.2f}%")
                print(f"   Độ phù hợp (R²): {score:.4f}")
                
                bot.visualize_linear(df, date, pred, _, coin_id)

        except KeyboardInterrupt:
            print("\n👋 Đã dừng chương trình.")
            break
        except Exception as e:
            print(f"\n❌ Lỗi không xác định: {e}")