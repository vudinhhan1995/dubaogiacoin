import sqlite3
import random
import datetime
import requests
import time
import json
import math
import pandas as pd
import numpy as np
import os
import logging

# --- TENSORFLOW CONFIGURATION ---
# Tắt log TensorFlow và AutoGraph warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=All, 1=Filter, 2=Error, 3=Fatal
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, render_template_string, request, redirect, url_for, flash

# --- AI & Data Science Libraries ---
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf

# Tắt AutoGraph warning cụ thể
tf.autograph.set_verbosity(0)

app = Flask(__name__)
app.secret_key = 'crypto_super_secret_key'
DB_NAME = 'portfolio.db'
MODEL_DIR = 'models' # Thư mục lưu model

# Cấu hình Cache
CACHE_TIMEOUT_PRICE = 600      # 10 phút
CACHE_TIMEOUT_HISTORY = 86400  # 24 giờ
MODEL_RETRAIN_INTERVAL = 86400 # 24 giờ train lại LSTM 1 lần

# ------------------------------------------------------------------
# 1. DATABASE & INIT
# ------------------------------------------------------------------
def init_db():
    # Tạo thư mục models nếu chưa có
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        quantity REAL NOT NULL,
        buy_price REAL NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS price_cache (
        symbol TEXT PRIMARY KEY, price REAL NOT NULL, updated_at REAL DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history_cache (
        symbol TEXT PRIMARY KEY, labels_json TEXT NOT NULL,
        prices_json TEXT NOT NULL, updated_at REAL DEFAULT 0)''')
    conn.commit()
    conn.close()

# ------------------------------------------------------------------
# 2. DATA FETCHING (COINGECKO)
# ------------------------------------------------------------------
COIN_MAP = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'BNB': 'binancecoin',
    'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin', 'DOT': 'polkadot',
    'USDT': 'tether', 'USDC': 'usd-coin', 'LINK': 'chainlink', 'LTC': 'litecoin',
    'SHIB': 'shiba-inu', 'TRX': 'tron', 'AVAX': 'avalanche-2', 'UNI': 'uniswap'
}

def get_coin_id(symbol):
    return COIN_MAP.get(symbol.upper(), symbol.lower())

def parse_updated_at(val):
    if val is None: return 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        try: return datetime.datetime.strptime(val, "%Y-%m-%d %H:%M:%S").timestamp()
        except: return 0.0
    return 0.0

def get_current_prices_bulk(symbols):
    if not symbols: return {}
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    prices = {}
    to_fetch = []
    now = time.time()
    
    for s in symbols:
        c.execute("SELECT price, updated_at FROM price_cache WHERE symbol = ?", (s,))
        row = c.fetchone()
        ts = parse_updated_at(row[1]) if row else 0
        if row and (now - ts < CACHE_TIMEOUT_PRICE):
            prices[s] = row[0]
        else:
            to_fetch.append(s)
            if row: prices[s] = row[0]
    conn.close()
    
    if not to_fetch: return prices

    ids = ",".join([get_coin_id(s) for s in to_fetch])
    try:
        resp = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            for s in to_fetch:
                cid = get_coin_id(s)
                p = data.get(cid, {}).get('usd') or data.get(cid.lower(), {}).get('usd')
                if p:
                    prices[s] = p
                    c.execute("INSERT OR REPLACE INTO price_cache (symbol, price, updated_at) VALUES (?, ?, ?)", (s, p, now))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"API Error: {e}")
    return prices

def get_historical_data(symbol, days=365):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = time.time()
    try:
        c.execute("SELECT labels_json, prices_json, updated_at FROM history_cache WHERE symbol = ?", (symbol,))
        row = c.fetchone()
        ts = parse_updated_at(row[2]) if row else 0
        if row and (now - ts < CACHE_TIMEOUT_HISTORY):
            conn.close()
            return json.loads(row[0]), json.loads(row[1])
    except: pass
    conn.close()

    cid = get_coin_id(symbol)
    try:
        resp = requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart?vs_currency=usd&days={days}", timeout=10)
        data = resp.json()
        if 'prices' not in data: return [], []
        labels, prices = [], []
        for p in data['prices']:
            labels.append(datetime.datetime.fromtimestamp(p[0]/1000).strftime('%d/%m/%Y'))
            prices.append(p[1])
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO history_cache VALUES (?, ?, ?, ?)", 
                 (symbol, json.dumps(labels), json.dumps(prices), now))
        conn.commit()
        conn.close()
        return labels, prices
    except: return [], []

# ------------------------------------------------------------------
# 3. AI MODELS (ARIMA & LSTM - with SAVE/LOAD)
# ------------------------------------------------------------------

def calculate_forecast_arima(history_prices, days_to_predict=7):
    """Mô hình thống kê ARIMA (Nhanh, Hiệu quả ngắn hạn)"""
    try:
        if len(history_prices) < 30: return [], [], [], []
        
        # Tinh chỉnh tham số theo độ dài dự báo
        order = (5, 1, 0) if days_to_predict <= 7 else (21, 1, 1)
        
        model = ARIMA(pd.Series(history_prices), order=order)
        res = model.fit()
        forecast = res.get_forecast(steps=days_to_predict)
        
        mean = [round(x, 4) for x in forecast.predicted_mean.tolist()]
        conf = forecast.conf_int(alpha=0.05)
        lower = [max(0, round(x, 4)) for x in conf.iloc[:, 0].tolist()]
        upper = [round(x, 4) for x in conf.iloc[:, 1].tolist()]
        
        labels = []
        curr = datetime.datetime.now()
        for i in range(1, days_to_predict + 1):
            labels.append((curr + datetime.timedelta(days=i)).strftime('%d/%m'))
            
        return labels, mean, lower, upper, "ARIMA (Live Calc)"
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return [], [], [], [], "Error"

def calculate_forecast_lstm(symbol, history_prices, days_to_predict=7):
    """
    Mô hình Deep Learning LSTM với cơ chế Save/Load
    """
    try:
        if len(history_prices) < 60: return [], [], [], [], "Not Enough Data"

        # Đường dẫn file model
        model_filename = f"{symbol.upper()}_lstm.keras"
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        # 1. Chuẩn bị dữ liệu (Scaling)
        data = np.array(history_prices).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        prediction_days = 60
        
        model = None
        model_status = "LSTM (Training...)"

        # 2. Kiểm tra Model đã lưu chưa
        should_train = True
        if os.path.exists(model_path):
            file_age = time.time() - os.path.getmtime(model_path)
            # Nếu model tồn tại và mới (< 24h), load lại dùng luôn
            if file_age < MODEL_RETRAIN_INTERVAL:
                try:
                    model = load_model(model_path)
                    should_train = False
                    model_status = "LSTM (Cached Model)"
                    print(f"Loaded cached model for {symbol}")
                except Exception as e:
                    print(f"Load model failed, retraining: {e}")
                    should_train = True
        
        # 3. Huấn luyện (nếu cần)
        if should_train:
            x_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i-prediction_days:i, 0])
                y_train.append(scaled_data[i, 0])
                
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train nhanh 5-10 epochs
            model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # Lưu model để dùng lần sau
            model.save(model_path)
            model_status = "LSTM (New Training)"
            print(f"Trained and saved new model for {symbol}")

        # 4. Dự báo đệ quy (Luôn dùng dữ liệu mới nhất để predict)
        # Lấy 60 ngày cuối cùng của dữ liệu HIỆN TẠI làm đầu vào
        test_inputs = scaled_data[len(scaled_data) - prediction_days:].reshape(1, -1)
        temp_input = list(test_inputs[0])
        lst_output = []
        
        for i in range(days_to_predict):
            if len(temp_input) > prediction_days:
                x_input = np.array(temp_input[-prediction_days:])
                x_input = x_input.reshape((1, prediction_days, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
            else:
                x_input = np.array(temp_input).reshape((1, prediction_days, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])

        predicted_prices = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
        
        # --- FIX: NumPy Deprecation Warning ---
        # Sử dụng .item() thay vì float(x) để chuyển đổi scalar array
        predicted_prices = [round(x.item(), 4) for x in predicted_prices.flatten()]

        labels = []
        curr = datetime.datetime.now()
        for i in range(1, days_to_predict + 1):
            labels.append((curr + datetime.timedelta(days=i)).strftime('%d/%m'))
            
        return labels, predicted_prices, [], [], model_status

    except Exception as e:
        print(f"LSTM Error: {e}")
        return [], [], [], [], "Error"

def get_market_sentiment(pnl):
    if pnl < -15: return "Thị trường Oversold (Quá bán). Vùng mua tiềm năng.", "danger"
    if pnl > 25: return "Thị trường Overbought (Quá mua). Cân nhắc chốt lời.", "success"
    return "Thị trường Sideway (Đi ngang). Tiếp tục quan sát.", "info"

# ------------------------------------------------------------------
# 4. TEMPLATE HTML
# ------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Crypto Portfolio</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .sidebar { background: #1a1d21; color: #fff; min-height: 100vh; padding: 20px; }
        .card { border: none; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .crypto-icon { width: 32px; height: 32px; border-radius: 50%; background: #f8f9fa; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: 600; font-size: 0.8rem; }
        .positive { color: #198754; font-weight: 600; }
        .negative { color: #dc3545; font-weight: 600; }
        
        /* Loading Overlay */
        #loading-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(255, 255, 255, 0.9); z-index: 9999;
            display: none; flex-direction: column;
            justify-content: center; align-items: center;
        }
        .spinner-ai {
            width: 3rem; height: 3rem; border: 4px solid #f3f3f3;
            border-top: 4px solid #0d6efd; border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

<!-- Loading Screen -->
<div id="loading-overlay">
    <div class="spinner-ai mb-3"></div>
    <h5 class="text-primary fw-bold">AI Processing...</h5>
    <p class="text-muted">Đang tải mô hình hoặc huấn luyện lại nếu cần...</p>
</div>

<div class="container-fluid">
    <div class="row">
        <!-- Sidebar -->
        <div class="col-md-2 sidebar d-none d-md-block">
            <h4 class="mb-4 text-center"><i class="fas fa-brain text-primary"></i> NeuroCrypto</h4>
            <ul class="nav flex-column">
                <li class="nav-item mb-2"><a href="{{ url_for('index') }}" class="nav-link text-white-50 active"><i class="fas fa-home me-2"></i> Dashboard</a></li>
                <li class="nav-item mb-2"><a href="#addModal" data-bs-toggle="modal" class="nav-link text-white-50"><i class="fas fa-plus me-2"></i> Thêm Tài Sản</a></li>
            </ul>
            <div class="mt-auto pt-5 text-center small text-secondary">
                <p>Powered by TensorFlow <br> & Statsmodels</p>
            </div>
        </div>

        <!-- Main -->
        <div class="col-md-10 p-4">
            <!-- Stats -->
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="card bg-primary text-white p-3">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="text-white-50">Tổng Tài Sản</h6>
                                <h3>${{ "{:,.2f}".format(total_value) }}</h3>
                            </div>
                            <div class="fs-1 text-white-50"><i class="fas fa-wallet"></i></div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card {{ 'bg-success' if total_pnl >= 0 else 'bg-danger' }} text-white p-3">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="text-white-50">Lãi/Lỗ (PnL)</h6>
                                <h3>{{ "+" if total_pnl >= 0 else "" }}{{ "{:,.2f}".format(total_pnl) }}$</h3>
                            </div>
                            <div class="fs-1 text-white-50"><i class="fas fa-chart-line"></i></div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-dark text-white p-3">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h6 class="text-white-50">Số Coin</h6>
                                <h3>{{ portfolio|length }}</h3>
                            </div>
                            <div class="fs-1 text-white-50"><i class="fas fa-coins"></i></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Portfolio Table -->
                <div class="col-lg-8">
                    <div class="card">
                        <div class="card-header bg-white py-3 d-flex justify-content-between align-items-center">
                            <h5 class="m-0 fw-bold">Danh Mục Đầu Tư</h5>
                            <button class="btn btn-sm btn-primary" data-bs-toggle="modal" data-bs-target="#addModal">
                                <i class="fas fa-plus"></i> Thêm
                            </button>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-hover align-middle mb-0">
                                <thead class="table-light">
                                    <tr>
                                        <th>Token</th>
                                        <th>Số lượng</th>
                                        <th>Giá TB</th>
                                        <th>Giá Hiện Tại</th>
                                        <th>Tổng</th>
                                        <th>PnL</th>
                                        <th>AI Forecast</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for item in portfolio %}
                                    <tr>
                                        <td>
                                            <div class="d-flex align-items-center">
                                                <span class="crypto-icon">{{ item.symbol[0] }}</span>
                                                <span class="fw-bold">{{ item.symbol }}</span>
                                            </div>
                                        </td>
                                        <td>{{ item.quantity }}</td>
                                        <td>${{ "{:,.2f}".format(item.buy_price) }}</td>
                                        <td>${{ "{:,.2f}".format(item.current_price) }}</td>
                                        <td><strong>${{ "{:,.2f}".format(item.total_val) }}</strong></td>
                                        <td class="{{ 'positive' if item.pnl >= 0 else 'negative' }}">
                                            {{ "+" if item.pnl >= 0 else "" }}{{ "{:,.2f}".format(item.pnl) }}%
                                        </td>
                                        <td>
                                            <div class="btn-group btn-group-sm">
                                                <a href="{{ url_for('predict', symbol=item.symbol, model='arima') }}" 
                                                   class="btn btn-outline-secondary" onclick="showLoading()">ARIMA</a>
                                                <a href="{{ url_for('predict', symbol=item.symbol, model='lstm') }}" 
                                                   class="btn btn-outline-primary" onclick="showLoading()">LSTM</a>
                                            </div>
                                            <a href="{{ url_for('delete_coin', id=item.id) }}" class="btn btn-sm text-danger ms-1">
                                                <i class="fas fa-times"></i>
                                            </a>
                                        </td>
                                    </tr>
                                    {% else %}
                                    <tr><td colspan="7" class="text-center py-4 text-muted">Chưa có dữ liệu</td></tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Chart Section -->
                    {% if chart_data_history %}
                    <div class="card mt-4" id="chart-section">
                        <div class="card-header bg-white py-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="m-0 fw-bold">{{ chart_symbol }} - Dự báo AI</h5>
                                    <span class="badge {{ 'bg-success' if 'Cached' in model_status else 'bg-primary' }}">
                                        Status: {{ model_status }}
                                    </span>
                                    <span class="badge bg-light text-dark border ms-1">365 ngày lịch sử</span>
                                </div>
                                <div class="btn-group btn-group-sm">
                                    {% for d in [7, 30, 90] %}
                                    <a href="{{ url_for('predict', symbol=chart_symbol, days=d, model=model_selected) }}" 
                                       class="btn btn-outline-dark {{ 'active' if days_selected == d else '' }}"
                                       onclick="showLoading()">{{ d }} ngày</a>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <div style="height: 400px;">
                                <canvas id="analysisChart"></canvas>
                            </div>
                            <div class="alert alert-{{ advice_color }} d-flex align-items-center mt-3">
                                <i class="fas fa-robot fs-4 me-3"></i>
                                <div>
                                    <strong>Nhận định AI:</strong> {{ advice_text }}
                                </div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>

                <!-- Right Panel -->
                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header bg-white fw-bold">Phân Bổ Vốn</div>
                        <div class="card-body">
                            <canvas id="allocationChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Add Modal -->
<div class="modal fade" id="addModal" tabindex="-1">
    <div class="modal-dialog">
        <form method="POST" action="{{ url_for('add_coin') }}">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Thêm Coin Mới</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label>Mã Coin (VD: BTC)</label>
                        <input type="text" name="symbol" class="form-control" required style="text-transform: uppercase">
                    </div>
                    <div class="mb-3">
                        <label>Số lượng</label>
                        <input type="number" step="any" name="quantity" class="form-control" required>
                    </div>
                    <div class="mb-3">
                        <label>Giá mua ($)</label>
                        <input type="number" step="any" name="buy_price" class="form-control" required>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="submit" class="btn btn-primary">Lưu</button>
                </div>
            </div>
        </form>
    </div>
</div>

<script>
    function showLoading() {
        document.getElementById('loading-overlay').style.display = 'flex';
    }

    // Allocation Chart
    const ctxAlloc = document.getElementById('allocationChart').getContext('2d');
    new Chart(ctxAlloc, {
        type: 'doughnut',
        data: {
            labels: [{% for i in portfolio %}"{{ i.symbol }}",{% endfor %}],
            datasets: [{
                data: [{% for i in portfolio %}{{ i.total_val }},{% endfor %}],
                backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b']
            }]
        }
    });

    // Forecast Chart
    {% if chart_data_history %}
    const ctxMain = document.getElementById('analysisChart').getContext('2d');
    const histData = {{ chart_data_history | tojson }};
    const predData = {{ chart_data_forecast | tojson }};
    const lower = {{ chart_data_lower | tojson }};
    const upper = {{ chart_data_upper | tojson }};
    
    // Stitch data
    const nulls = new Array(histData.length - 1).fill(null);
    const connect = histData[histData.length - 1];
    
    const predSet = nulls.concat([connect]).concat(predData);
    const lowSet = lower.length ? nulls.concat([connect]).concat(lower) : [];
    const upSet = upper.length ? nulls.concat([connect]).concat(upper) : [];
    
    const labels = {{ chart_labels_history | tojson }}.concat({{ chart_labels_forecast | tojson }});
    const modelName = "{{ model_selected }}";

    new Chart(ctxMain, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Lịch sử',
                    data: histData,
                    borderColor: '#858796',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Dự báo (' + modelName.toUpperCase() + ')',
                    data: predSet,
                    borderColor: modelName === 'lstm' ? '#4e73df' : '#e74a3b',
                    backgroundColor: modelName === 'lstm' ? 'rgba(78, 115, 223, 0.05)' : 'rgba(231, 74, 59, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                // Chỉ vẽ vùng tin cậy nếu là ARIMA (LSTM cơ bản ko có)
                ...(upSet.length ? [{
                    label: 'Upper Bound',
                    data: upSet,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(231, 74, 59, 0.1)',
                    pointRadius: 0,
                    fill: '+1'
                }, {
                    label: 'Lower Bound',
                    data: lowSet,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    fill: false
                }] : [])
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { labels: { filter: i => !i.text.includes('Bound') } } },
            scales: { x: { grid: { display: false } }, y: { beginAtZero: false } }
        }
    });
    document.getElementById('chart-section').scrollIntoView({behavior: 'smooth'});
    {% endif %}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ------------------------------------------------------------------
# 5. ROUTES
# ------------------------------------------------------------------
@app.route('/')
def index():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio")
    rows = c.fetchall()
    conn.close()

    port_list = []
    symbols = [r['symbol'] for r in rows]
    prices = get_current_prices_bulk(symbols)
    
    total_val = 0
    total_pnl = 0
    total_inv = 0

    for r in rows:
        sym = r['symbol']
        qty = r['quantity']
        bp = r['buy_price']
        cp = prices.get(sym, 0.0)
        
        val = qty * cp
        inv = qty * bp
        pnl_pct = ((cp - bp) / bp * 100) if bp > 0 else 0
        
        total_val += val
        total_inv += inv
        
        port_list.append({
            'id': r['id'], 'symbol': sym, 'quantity': qty, 'buy_price': bp,
            'current_price': cp, 'total_val': val, 'pnl': pnl_pct
        })
    
    total_pnl = total_val - total_inv

    return render_template_string(HTML_TEMPLATE, portfolio=port_list, total_value=total_val, total_pnl=total_pnl)

@app.route('/add', methods=['POST'])
def add_coin():
    s = request.form.get('symbol').upper().strip()
    q = float(request.form.get('quantity'))
    p = float(request.form.get('buy_price'))
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (symbol, quantity, buy_price) VALUES (?,?,?)", (s,q,p))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/delete/<int:id>')
def delete_coin(id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/predict/<symbol>')
def predict(symbol):
    days = request.args.get('days', 7, type=int)
    model_type = request.args.get('model', 'arima')
    
    # Re-fetch data for background UI (same as index)
    init_db()
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio")
    rows = c.fetchall()
    conn.close()
    
    # ... (Simplified logic for background data calc similar to index)
    port_list = []
    symbols = [r['symbol'] for r in rows]
    prices = get_current_prices_bulk(symbols)
    total_val = 0; total_inv = 0
    target_pnl = 0
    
    for r in rows:
        cp = prices.get(r['symbol'], 0.0)
        val = r['quantity'] * cp
        total_val += val
        total_inv += r['quantity'] * r['buy_price']
        pnl = ((cp - r['buy_price'])/r['buy_price']*100) if r['buy_price'] else 0
        if r['symbol'] == symbol: target_pnl = pnl
        port_list.append({'id': r['id'], 'symbol': r['symbol'], 'quantity': r['quantity'], 
                          'buy_price': r['buy_price'], 'current_price': cp, 'total_val': val, 'pnl': pnl})
    
    total_pnl = total_val - total_inv
    
    # --- FORECAST LOGIC ---
    hist_lbl, hist_data = get_historical_data(symbol, 365)
    
    if not hist_data:
        flash("Không đủ dữ liệu lịch sử.", "warning")
        return redirect(url_for('index'))

    pred_lbl, pred_mean, low, up = [], [], [], []
    model_status = "ARIMA"

    if model_type == 'lstm':
        pred_lbl, pred_mean, low, up, model_status = calculate_forecast_lstm(symbol, hist_data, days)
    else:
        pred_lbl, pred_mean, low, up, model_status = calculate_forecast_arima(hist_data, days)
        
    advice_txt, advice_col = get_market_sentiment(target_pnl)

    return render_template_string(
        HTML_TEMPLATE,
        portfolio=port_list, total_value=total_val, total_pnl=total_pnl,
        chart_symbol=symbol,
        chart_data_history=hist_data,
        chart_labels_history=hist_lbl,
        chart_data_forecast=pred_mean,
        chart_data_lower=low,
        chart_data_upper=up,
        chart_labels_forecast=pred_lbl,
        days_selected=days,
        model_selected=model_type,
        model_status=model_status,
        advice_text=advice_txt,
        advice_color=advice_col
    )

if __name__ == '__main__':
    init_db()
    print("AI Crypto App running on http://127.0.0.1:5005")
    app.run(debug=True, port=5005)