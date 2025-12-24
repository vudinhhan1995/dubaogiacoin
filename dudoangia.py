import requests
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import logging

# Tắt cảnh báo
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class CoinGeckoPredictor:
    def __init__(self):
        self.base_api = "https://api.coingecko.com/api/v3"
        self.currency = 'usd'

    def extract_coin_id(self, url):
        try:
            match = re.search(r'/coins/([^/?]+)', url)
            if match: return match.group(1)
            if "/" not in url: return url.lower().strip()
            return None
        except Exception:
            return None

    def fetch_history(self, coin_id, days=365, max_retries=3):
        # Lấy tối đa dữ liệu để train model tốt hơn
        if days < 180: days = 365 
        
        url = f"{self.base_api}/coins/{coin_id}/market_chart"
        params = {'vs_currency': self.currency, 'days': days, 'interval': 'daily'}
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 429: # Rate limit
                    time.sleep((attempt + 1) * 5)
                    continue
                if response.status_code == 200:
                    data = response.json()
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', [])
                    
                    if not prices: return None
                    
                    # Xử lý Price
                    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Xử lý Volume (đôi khi độ dài mảng không khớp)
                    df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    df_vol['date'] = pd.to_datetime(df_vol['timestamp'], unit='ms')
                    
                    # Merge Price và Volume
                    df = pd.merge(df, df_vol[['date', 'volume']], on='date', how='inner')
                    
                    df = df.dropna()
                    # Loại bỏ bản ghi trùng lặp (CoinGecko đôi khi trả về 2 bản ghi cùng ngày)
                    df = df.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
                    
                    return df
                if response.status_code == 404: return None
                return None
            except requests.RequestException:
                continue
        return None

    def calculate_indicators(self, df):
        """Tính toán các chỉ báo kỹ thuật nâng cao làm Feature cho AI"""
        df = df.copy()
        
        # 1. Trend Indicators
        df['SMA_20'] = df['price'].rolling(window=20).mean()
        df['SMA_50'] = df['price'].rolling(window=50).mean()
        
        # 2. Momentum Indicators (RSI)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 4. Volatility (Bollinger Bands)
        df['BB_Middle'] = df['price'].rolling(window=20).mean()
        df['BB_Std'] = df['price'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        
        # 5. Lag Features (Quan trọng cho Machine Learning: Giá hôm qua, Giá tuần trước)
        df['Lag_1'] = df['price'].shift(1)
        df['Lag_7'] = df['price'].shift(7)
        
        # Fill NaN ban đầu bằng phương pháp Backward Fill
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df

    def predict_machine_learning(self, df, days_ahead=1):
        """
        Thay thế Linear Regression bằng Random Forest.
        Phương pháp: Dùng dữ liệu T để dự đoán T+1 (Next Day Prediction)
        """
        df_full = self.calculate_indicators(df)
        df_ml = df_full.dropna().copy()
        
        # Feature Selection: Không dùng ngày tháng, dùng chỉ báo kỹ thuật
        features = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'volume', 'Lag_1', 'Lag_7', 'BB_Upper', 'BB_Lower']
        target = 'price'
        
        # Chuẩn bị dữ liệu: Target của dòng hiện tại là Price của days_ahead ngày sau
        # Lưu ý: Với days_ahead > 1, độ chính xác ML thuần túy sẽ giảm nhanh.
        # Ở đây ta tối ưu cho dự báo ngắn hạn (1 ngày) hoặc trend ngắn.
        
        X = df_ml[features]
        y = df_ml[target].shift(-days_ahead) # Shift ngược để dòng T chứa target T+days_ahead
        
        # Loại bỏ các dòng cuối cùng bị NaN do shift
        X = X.iloc[:-days_ahead]
        y = y.iloc[:-days_ahead]
        
        # Chia train/test theo thời gian (Không được shuffle)
        train_size = int(len(X) * 0.9)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Model Random Forest (Mạnh hơn Linear Regression nhiều)
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Đánh giá
        score = model.score(X_test, y_test)
        
        # Dự báo tương lai
        # Lấy dữ liệu mới nhất để dự đoán
        last_row_features = df_ml[features].iloc[-1].values.reshape(1, -1)
        predicted_price = model.predict(last_row_features)[0]
        
        last_date = df['date'].iloc[-1]
        next_date = last_date + timedelta(days=days_ahead)
        
        return next_date, max(0, predicted_price), model, score, 0

    def predict_prophet(self, df, days_ahead=7):
        """
        Cải tiến Prophet:
        1. Tự động điều chỉnh changepoint_prior_scale dựa trên biến động giá.
        2. Chuyển sang Log scale để xử lý tăng trưởng theo cấp số nhân của Crypto.
        """
        prophet_df = df[['date', 'price']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Log Transform: Giúp ổn định phương sai cho chuỗi thời gian tài chính
        prophet_df['y'] = np.log(prophet_df['y'])
        
        # Tính độ biến động (Volatility) để chỉnh tham số mô hình
        # Nếu giá biến động mạnh -> Tăng changepoint_scale để bắt trend nhanh hơn
        volatility = prophet_df['y'].diff().std()
        
        if volatility > 0.05: # Biến động rất mạnh (Crypto rác/Meme)
            cps = 0.5
        elif volatility > 0.02: # Biến động vừa (Altcoin)
            cps = 0.3
        else: # Ổn định (BTC/ETH sideway)
            cps = 0.15
            
        use_yearly = (df['date'].max() - df['date'].min()).days > 365
        
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=True, 
            yearly_seasonality=use_yearly,
            changepoint_prior_scale=cps, # Linh hoạt theo volatility
            changepoint_range=0.9,
            seasonality_mode='multiplicative' # Crypto thường theo mô hình nhân (biến động tăng theo giá)
        )
        
        # Thêm country holidays nếu cần (tùy chọn)
        # model.add_country_holidays(country_name='US')
        
        model.fit(prophet_df)
        
        # Cross Validation thủ công để lấy MAPE
        mape = 0
        try:
            cutoff = int(len(prophet_df) * 0.9)
            train = prophet_df.iloc[:cutoff]
            test = prophet_df.iloc[cutoff:]
            
            m_test = Prophet(
                daily_seasonality=True, weekly_seasonality=True, 
                changepoint_prior_scale=cps, seasonality_mode='multiplicative'
            ).fit(train)
            
            future_test = m_test.make_future_dataframe(periods=len(test))
            fc_test = m_test.predict(future_test)
            
            y_true = np.exp(test['y'].values)
            y_pred = np.exp(fc_test['yhat'].tail(len(test)).values)
            
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except Exception as e:
            print(f"Lỗi tính MAPE: {e}")
            pass

        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        future_forecast = forecast.tail(days_ahead)
        
        # Inverse Log Transform
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            future_forecast[col] = np.exp(future_forecast[col])

        return future_forecast['ds'].tolist(), future_forecast['yhat'].tolist(), future_forecast[['yhat_lower', 'yhat_upper']].values, model, mape

    def create_plotly_chart(self, df, pred_dates, pred_prices, bounds=None, coin_id="COIN", mode="Prophet"):
        df = self.calculate_indicators(df)
        
        # Layout 3 dòng: Giá (60%), Volume (20%), RSI (20%)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f"Phân tích giá {coin_id.upper()}", "Volume", "RSI & Momentum"))

        # --- 1. Main Price Chart ---
        fig.add_trace(go.Scatter(x=df['date'], y=df['price'], mode='lines', name='Lịch sử', line=dict(color='gray', width=1)), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Lower'], mode='lines', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', line=dict(width=0), name='Bollinger Bands'), row=1, col=1)

        dates = pd.to_datetime(pred_dates)
        if mode == "Prophet":
            fig.add_trace(go.Scatter(x=dates, y=pred_prices, mode='lines', name='AI Dự báo', line=dict(color='#00ff00', width=3)), row=1, col=1)
            if bounds is not None:
                fig.add_trace(go.Scatter(
                    x=pd.concat([pd.Series(dates), pd.Series(dates)[::-1]]),
                    y=pd.concat([pd.Series(bounds[:, 1]), pd.Series(bounds[:, 0])[::-1]]),
                    fill='toself', fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(0,0,0,0)'),
                    name='Vùng rủi ro', showlegend=False
                ), row=1, col=1)
        else: # ML Short-term
            fig.add_trace(go.Scatter(x=dates, y=pred_prices, mode='markers+text', 
                                     text=[f"${p:,.2f}" for p in pred_prices], textposition="top center",
                                     marker=dict(color='orange', size=15, symbol='star'), name='Giá mục tiêu'), row=1, col=1)

        # --- 2. Volume Chart ---
        colors = ['red' if row['price'] < row['open_price'] else 'green' for i, row in df.iterrows()] if 'open_price' in df else 'blue'
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color='rgba(0, 0, 255, 0.3)'), row=2, col=1)

        # --- 3. RSI Chart ---
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI_14'], mode='lines', name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_shape(type="line", x0=df['date'].iloc[0], x1=dates[-1] if isinstance(dates, list) else dates, y0=70, y1=70, line=dict(color="red", width=1, dash="dot"), row=3, col=1)
        fig.add_shape(type="line", x0=df['date'].iloc[0], x1=dates[-1] if isinstance(dates, list) else dates, y0=30, y1=30, line=dict(color="green", width=1, dash="dot"), row=3, col=1)

        fig.update_layout(height=800, template="plotly_white", hovermode="x unified", xaxis_rangeslider_visible=False)
        return fig