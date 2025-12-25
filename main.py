import sqlite3
import random
import datetime
import requests
import time
import json
import math  # Thêm thư viện toán học
from flask import Flask, render_template_string, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'crypto_super_secret_key'
DB_NAME = 'portfolio.db'

# Cấu hình thời gian Cache (Giây)
CACHE_TIMEOUT_PRICE = 600      # 10 phút cho giá hiện tại
CACHE_TIMEOUT_HISTORY = 86400  # 24 giờ cho biểu đồ lịch sử

# ------------------------------------------------------------------
# 1. CẤU HÌNH DATABASE (SQLite)
# ------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Bảng Portfolio
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            buy_price REAL NOT NULL
        )
    ''')
    
    # Bảng Cache Giá Hiện Tại
    c.execute('''
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol TEXT PRIMARY KEY,
            price REAL NOT NULL,
            updated_at REAL DEFAULT 0
        )
    ''')

    # Bảng Cache Lịch Sử
    c.execute('''
        CREATE TABLE IF NOT EXISTS history_cache (
            symbol TEXT PRIMARY KEY,
            labels_json TEXT NOT NULL,
            prices_json TEXT NOT NULL,
            updated_at REAL DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

# ------------------------------------------------------------------
# 2. LOGIC COINGECKO API & CACHING
# ------------------------------------------------------------------
COIN_MAP = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'BNB': 'binancecoin',
    'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin', 'DOT': 'polkadot',
    'USDT': 'tether', 'USDC': 'usd-coin', 'LINK': 'chainlink', 'LTC': 'litecoin',
    'SHIB': 'shiba-inu', 'TRX': 'tron', 'AVAX': 'avalanche-2', 'UNI': 'uniswap'
}

def get_coin_id(symbol):
    return COIN_MAP.get(symbol.upper(), symbol.lower())

def parse_updated_at(updated_at_val):
    if updated_at_val is None:
        return 0.0
    if isinstance(updated_at_val, (int, float)):
        return float(updated_at_val)
    if isinstance(updated_at_val, str):
        try:
            dt = datetime.datetime.strptime(updated_at_val, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            return 0.0
    return 0.0

def get_current_prices_bulk(symbols):
    if not symbols: return {}
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    prices = {}
    symbols_to_fetch = []
    current_time = time.time()
    
    for sym in symbols:
        c.execute("SELECT price, updated_at FROM price_cache WHERE symbol = ?", (sym,))
        row = c.fetchone()
        last_updated = 0.0
        cached_price = 0.0
        if row:
            cached_price = row[0]
            last_updated = parse_updated_at(row[1])
        
        if row and (current_time - last_updated < CACHE_TIMEOUT_PRICE):
            prices[sym] = cached_price
        else:
            symbols_to_fetch.append(sym)
            if row: prices[sym] = cached_price

    conn.close()
    
    if not symbols_to_fetch:
        return prices

    print(f"🌐 Đang tải lại giá mới cho: {', '.join(symbols_to_fetch)}")
    ids = [get_coin_id(s) for s in symbols_to_fetch]
    ids_str = ",".join(ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=usd"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            for sym in symbols_to_fetch:
                cid = get_coin_id(sym)
                price = None
                if cid in data and 'usd' in data[cid]:
                    price = data[cid]['usd']
                elif cid.lower() in data and 'usd' in data[cid.lower()]:
                    price = data[cid.lower()]['usd']
                
                if price is not None:
                    prices[sym] = price
                    c.execute("INSERT OR REPLACE INTO price_cache (symbol, price, updated_at) VALUES (?, ?, ?)", 
                             (sym, price, current_time))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"❌ Lỗi API: {e}")
    
    return prices

def get_historical_data(symbol, days=365):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    current_time = time.time()
    
    try:
        c.execute("SELECT labels_json, prices_json, updated_at FROM history_cache WHERE symbol = ?", (symbol,))
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = None

    last_updated = 0.0
    if row:
        last_updated = parse_updated_at(row[2])
    
    if row and (current_time - last_updated < CACHE_TIMEOUT_HISTORY):
        conn.close()
        try:
            return json.loads(row[0]), json.loads(row[1])
        except json.JSONDecodeError:
            pass
    
    conn.close()
    
    print(f"🌐 Tải dữ liệu lịch sử mới cho {symbol}...")
    coin_id = get_coin_id(symbol)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'prices' not in data: return [], []
        prices_data = data['prices']
        labels = []
        prices = []
        for point in prices_data:
            ts = point[0] / 1000
            price = point[1]
            date_str = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y')
            labels.append(date_str)
            prices.append(round(price, 4))
            
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO history_cache (symbol, labels_json, prices_json, updated_at) VALUES (?, ?, ?, ?)", 
                 (symbol, json.dumps(labels), json.dumps(prices), current_time))
        conn.commit()
        conn.close()
        return labels, prices
    except Exception as e:
        print(f"❌ Lỗi lấy lịch sử: {e}")
        if row:
            try: return json.loads(row[0]), json.loads(row[1])
            except: pass
        return [], []

def calculate_forecast(history_prices, days_to_predict=7):
    """
    Sử dụng Mô hình Geometric Brownian Motion (GBM) + Monte Carlo Simulation.
    Phù hợp cho Crypto: Mô phỏng xu hướng ngẫu nhiên dựa trên độ biến động (Volatility) thực tế.
    """
    if not history_prices or len(history_prices) < 2:
        return [], []

    # Dùng 90 ngày gần nhất để tính độ biến động (Volatility)
    lookback = 90
    recent_data = history_prices[-lookback:] if len(history_prices) > lookback else history_prices
    
    # 1. Tính lợi nhuận logarit (Log Returns)
    returns = []
    for i in range(1, len(recent_data)):
        if recent_data[i-1] == 0: continue
        r = math.log(recent_data[i] / recent_data[i-1])
        returns.append(r)
    
    if not returns: return [], []

    # 2. Tính Drift (Xu hướng) và Volatility (Biến động)
    avg_return = sum(returns) / len(returns)
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    stdev = math.sqrt(variance) # Độ lệch chuẩn (Sigma)
    
    # Drift điều chỉnh cho GBM: mu - 0.5 * sigma^2
    drift = avg_return - (0.5 * variance)
    
    # 3. Monte Carlo Simulation (Chạy 100 kịch bản và lấy trung bình)
    num_simulations = 100
    future_paths = [[0.0] * days_to_predict for _ in range(num_simulations)]
    last_price = recent_data[-1]
    
    for sim in range(num_simulations):
        current_sim_price = last_price
        for day in range(days_to_predict):
            # Tạo số ngẫu nhiên từ phân phối chuẩn (Standard Normal Distribution)
            z_score = random.normalvariate(0, 1)
            
            # Công thức GBM: Pt = Pt-1 * e^(drift + sigma * Z)
            daily_return = drift + stdev * z_score
            current_sim_price = current_sim_price * math.exp(daily_return)
            
            future_paths[sim][day] = current_sim_price

    # 4. Tính giá trung bình của các kịch bản (Expected Price)
    final_forecast_prices = []
    future_labels = []
    current_date = datetime.datetime.now()
    
    for day in range(days_to_predict):
        daily_sum = sum(future_paths[sim][day] for sim in range(num_simulations))
        avg_price = daily_sum / num_simulations
        
        final_forecast_prices.append(round(avg_price, 4))
        date_part = (current_date + datetime.timedelta(days=day+1)).strftime('%d/%m')
        future_labels.append(date_part)
        
    return future_labels, final_forecast_prices

def get_market_sentiment_advanced(pnl_percent, rsi_simulated=None):
    if rsi_simulated is None: rsi_simulated = random.randint(30, 70) 
    advice = ""; color = "primary"
    if pnl_percent < -15:
        advice = "Giá đã giảm sâu (Oversold). Vùng tích lũy DCA tiềm năng."; color = "danger"
    elif pnl_percent > 25:
        advice = "Lợi nhuận cao. Cân nhắc chốt lời từng phần (Take Profit)."; color = "success"
    else:
        if rsi_simulated > 70: advice = "Thị trường hưng phấn (Overbought). Cẩn thận điều chỉnh."; color = "warning"
        elif rsi_simulated < 30: advice = "Thị trường sợ hãi. Cơ hội mua ngắn hạn."; color = "info"
        else: advice = "Thị trường trung tính (Sideway). Tiếp tục quan sát.";
    return advice, color

# ------------------------------------------------------------------
# 3. HTML TEMPLATES
# ------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quản Lý Crypto Portfolio</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #f8f9fa; }
        .card { border-radius: 15px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .crypto-icon { width: 30px; height: 30px; border-radius: 50%; background: #eee; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: bold; }
        .positive { color: #28a745; font-weight: bold; }
        .negative { color: #dc3545; font-weight: bold; }
        .sidebar { background: #343a40; color: white; min-height: 100vh; padding: 20px; }
        .main-content { padding: 20px; }
        .api-badge { font-size: 0.7em; background: #2ecc71; color: white; padding: 2px 6px; border-radius: 4px; vertical-align: middle; }
        .toast-container { position: fixed; top: 20px; right: 20px; z-index: 9999; }
    </style>
</head>
<body>

<div class="container-fluid">
    <div class="row">
        <!-- Sidebar -->
        <div class="col-md-3 col-lg-2 sidebar d-none d-md-block">
            <h3 class="text-center mb-4"><i class="fas fa-coins"></i> CoinManager</h3>
            <ul class="nav flex-column">
                <li class="nav-item mb-2"><a href="{{ url_for('index') }}" class="nav-link text-white active"><i class="fas fa-tachometer-alt"></i> Tổng quan</a></li>
                <li class="nav-item mb-2"><a href="#addModal" data-bs-toggle="modal" class="nav-link text-white"><i class="fas fa-plus-circle"></i> Thêm Coin</a></li>
            </ul>
            <div class="mt-5 text-center small text-muted">
                <p>Smart Caching Enabled <br> Model: GBM Monte Carlo</p>
            </div>
        </div>

        <!-- Main Content -->
        <div class="col-md-9 col-lg-10 main-content">
            <div class="toast-container"></div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                <script>
                  document.addEventListener('DOMContentLoaded', function() {
                    {% for category, message in messages %}
                    showToast('{{ message }}', '{{ category }}');
                    {% endfor %}
                  });
                </script>
              {% endif %}
            {% endwith %}
            
            <!-- Header Stats -->
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="card bg-primary text-white">
                        <div class="card-body">
                            <h5 class="card-title">Tổng Tài Sản</h5>
                            <h2>${{ "{:,.2f}".format(total_value) }}</h2>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card {{ 'bg-success' if total_pnl >= 0 else 'bg-danger' }} text-white">
                        <div class="card-body">
                            <h5 class="card-title">Tổng Lãi/Lỗ (PnL)</h5>
                            <h2>
                                {{ "+" if total_pnl >= 0 else "" }}{{ "{:,.2f}".format(total_pnl) }} $
                                <small style="font-size: 0.6em">({{ "{:,.2f}".format(total_pnl_percent) }}%)</small>
                            </h2>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-info text-white">
                        <div class="card-body">
                            <h5 class="card-title">Số Coin Nắm Giữ</h5>
                            <h2>{{ portfolio|length }}</h2>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Danh sách Portfolio -->
                <div class="col-lg-8">
                    <div class="card">
                        <div class="card-header bg-white d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">Danh Mục Đầu Tư</h5>
                            <div>
                                <button class="btn btn-sm btn-primary" data-bs-toggle="modal" data-bs-target="#addModal">+ Thêm</button>
                            </div>
                        </div>
                        <div class="card-body table-responsive">
                            <table class="table table-hover align-middle">
                                <thead>
                                    <tr>
                                        <th>Coin</th>
                                        <th>Số lượng</th>
                                        <th>Giá TB Mua</th>
                                        <th>Giá Hiện Tại</th>
                                        <th>Giá trị</th>
                                        <th>PnL</th>
                                        <th>Hành động</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for item in portfolio %}
                                    <tr>
                                        <td>
                                            <div class="d-flex align-items-center">
                                                <span class="crypto-icon">{{ item.symbol[0] }}</span>
                                                <strong>{{ item.symbol }}</strong>
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
                                            <a href="{{ url_for('delete_coin', id=item.id) }}" class="btn btn-sm btn-outline-danger" onclick="return confirm('Bạn chắc chắn muốn xóa?')"><i class="fas fa-trash"></i></a>
                                            <a href="{{ url_for('predict', symbol=item.symbol) }}" class="btn btn-sm btn-outline-info" title="Dự báo giá"><i class="fas fa-chart-line"></i></a>
                                        </td>
                                    </tr>
                                    {% else %}
                                    <tr><td colspan="7" class="text-center text-muted">Chưa có coin nào. Hãy thêm mới!</td></tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        <div class="card-footer text-muted small">
                            <i class="fas fa-info-circle"></i> Giá lưu cache 10p. Lịch sử lưu 24h.
                        </div>
                    </div>
                    
                    <!-- Khu vực Biểu đồ -->
                    {% if chart_data_history %}
                    <div class="card mt-4" id="prediction-section">
                        <div class="card-header bg-white">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h5 class="mb-0">Phân tích giá: {{ chart_symbol }}</h5>
                                <span class="badge bg-secondary">Dữ liệu 365 ngày</span>
                            </div>
                            
                            <div class="btn-group w-100" role="group">
                                <a href="{{ url_for('predict', symbol=chart_symbol, days=7) }}" class="btn btn-outline-primary {{ 'active' if days_selected == 7 else '' }}">Ngắn hạn (7d)</a>
                                <a href="{{ url_for('predict', symbol=chart_symbol, days=30) }}" class="btn btn-outline-primary {{ 'active' if days_selected == 30 else '' }}">Trung hạn (30d)</a>
                                <a href="{{ url_for('predict', symbol=chart_symbol, days=90) }}" class="btn btn-outline-primary {{ 'active' if days_selected == 90 else '' }}">Dài hạn (90d)</a>
                            </div>
                        </div>
                        <div class="card-body">
                            <div style="height: 350px;">
                                <canvas id="analysisChart"></canvas>
                            </div>
                            <div class="alert alert-{{ advice_color }} mt-3 mb-0">
                                <i class="fas fa-robot"></i> <strong>AI Advice ({{ days_selected }} ngày):</strong> {{ advice_text }}
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>

                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header bg-white">
                            <h5 class="mb-0">Phân Bổ Tài Sản</h5>
                        </div>
                        <div class="card-body">
                            <canvas id="allocationChart"></canvas>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header bg-white">
                            <h5 class="mb-0">Trạng Thái Danh Mục</h5>
                        </div>
                        <div class="card-body">
                            <ul class="list-group list-group-flush">
                                {% if total_pnl_percent < -10 %}
                                    <li class="list-group-item text-danger"><i class="fas fa-exclamation-triangle"></i> Cảnh báo: Lỗ >10%.</li>
                                {% elif total_pnl_percent > 20 %}
                                    <li class="list-group-item text-success"><i class="fas fa-check-circle"></i> Tốt: Lãi >20%.</li>
                                {% else %}
                                    <li class="list-group-item text-muted"><i class="fas fa-info-circle"></i> Danh mục ổn định.</li>
                                {% endif %}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Modal Thêm Coin -->
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
                        <label class="form-label">Mã Coin (Symbol)</label>
                        <input type="text" name="symbol" class="form-control" placeholder="VD: BTC, ETH" required style="text-transform: uppercase">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Số lượng</label>
                        <input type="number" step="any" name="quantity" class="form-control" placeholder="0.0" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Giá mua trung bình ($)</label>
                        <input type="number" step="any" name="buy_price" class="form-control" placeholder="0.0" required>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Đóng</button>
                    <button type="submit" class="btn btn-primary">Lưu</button>
                </div>
            </div>
        </form>
    </div>
</div>

<script>
    function showToast(message, type = 'warning') {
        const toastContainer = document.querySelector('.toast-container');
        const toastId = 'toast-' + Date.now();
        const bgColor = type === 'warning' ? 'bg-warning' : type === 'danger' ? 'bg-danger' : type === 'success' ? 'bg-success' : 'bg-info';
        const icon = type === 'warning' ? 'fa-exclamation-triangle' : type === 'danger' ? 'fa-times-circle' : type === 'success' ? 'fa-check-circle' : 'fa-info-circle';
        
        const toastHtml = `
            <div id="${toastId}" class="toast ${bgColor} text-white" role="alert">
                <div class="toast-header ${bgColor} text-white">
                    <i class="fas ${icon} me-2"></i>
                    <strong class="me-auto">Thông báo</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">${message}</div>
            </div>
        `;
        toastContainer.insertAdjacentHTML('insertAdjacentHTML', toastHtml);
        new bootstrap.Toast(document.getElementById(toastId), { delay: 5000 }).show();
    }
    
    const ctxAlloc = document.getElementById('allocationChart').getContext('2d');
    new Chart(ctxAlloc, {
        type: 'doughnut',
        data: {
            labels: [{% for item in portfolio %}"{{ item.symbol }}",{% endfor %}],
            datasets: [{
                data: [{% for item in portfolio %}{{ item.total_val }},{% endfor %}],
                backgroundColor: ['#f6e58d', '#ffbe76', '#ff7979', '#badc58', '#dff9fb', '#7ed6df', '#e056fd', '#686de0'],
            }]
        }
    });

    {% if chart_data_history %}
    const ctxAnalysis = document.getElementById('analysisChart').getContext('2d');
    const historyData = {{ chart_data_history | tojson }};
    const forecastData = {{ chart_data_forecast | tojson }};
    const historyLabels = {{ chart_labels_history | tojson }};
    const forecastLabels = {{ chart_labels_forecast | tojson }};
    
    const allLabels = historyLabels.concat(forecastLabels);
    const nullPadding = new Array(historyData.length - 1).fill(null);
    const connectionPoint = historyData[historyData.length - 1]; 
    const dataSet2 = nullPadding.concat([connectionPoint]).concat(forecastData);

    new Chart(ctxAnalysis, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Lịch sử giá',
                    data: historyData,
                    borderColor: '#2980b9',
                    backgroundColor: 'rgba(41, 128, 185, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Dự báo (GBM Monte Carlo)',
                    data: dataSet2,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 1,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: { y: { beginAtZero: false } }
        }
    });
    document.getElementById('prediction-section').scrollIntoView({behavior: 'smooth', block: 'center'});
    {% endif %}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ------------------------------------------------------------------
# 4. ROUTES & CONTROLLERS
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

    portfolio_data = []
    symbols_list = [row['symbol'] for row in rows]
    
    real_prices = get_current_prices_bulk(symbols_list)

    total_value = 0
    total_invested = 0

    for row in rows:
        sym = row['symbol']
        qty = row['quantity']
        buy_price = row['buy_price']
        
        current_price = real_prices.get(sym) or 0.0
        
        val = qty * current_price
        invested = qty * buy_price
        pnl_percent = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
        
        total_value += val
        total_invested += invested
        
        portfolio_data.append({
            'id': row['id'], 'symbol': sym, 'quantity': qty, 'buy_price': buy_price,
            'current_price': current_price, 'total_val': val, 'pnl': pnl_percent
        })

    total_pnl = total_value - total_invested
    total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    return render_template_string(
        HTML_TEMPLATE,
        portfolio=portfolio_data,
        total_value=total_value,
        total_pnl=total_pnl,
        total_pnl_percent=total_pnl_percent,
        chart_data_history=None
    )

@app.route('/add', methods=['POST'])
def add_coin():
    symbol = request.form.get('symbol').strip().upper()
    quantity = float(request.form.get('quantity'))
    buy_price = float(request.form.get('buy_price'))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (symbol, quantity, buy_price) VALUES (?, ?, ?)", 
              (symbol, quantity, buy_price))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/delete/<int:id>')
def delete_coin(id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/predict/<symbol>')
def predict(symbol):
    days_to_predict = request.args.get('days', 7, type=int)

    init_db()
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio")
    rows = c.fetchall()
    conn.close()

    symbols_list = [r['symbol'] for r in rows]
    real_prices = get_current_prices_bulk(symbols_list)
    
    portfolio_data = []
    total_value = 0; total_invested = 0
    target_coin_pnl = 0

    for row in rows:
        sym = row['symbol']
        qty = row['quantity']
        buy_price = row['buy_price']
        current_price = real_prices.get(sym) or 0.0
        val = qty * current_price
        invested = qty * buy_price
        pnl = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
        total_value += val
        total_invested += invested
        
        if sym == symbol: target_coin_pnl = pnl

        portfolio_data.append({
            'id': row['id'], 'symbol': sym, 'quantity': qty, 'buy_price': buy_price,
            'current_price': current_price, 'total_val': val, 'pnl': pnl
        })
    
    total_pnl = total_value - total_invested
    total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    hist_labels, hist_data = get_historical_data(symbol, days=365)
    
    # Dùng hàm dự báo mới
    forecast_labels, forecast_data = calculate_forecast(hist_data, days_to_predict=days_to_predict)
    advice_text, advice_color = get_market_sentiment_advanced(target_coin_pnl)

    if not hist_data:
        flash(f"Không tìm thấy dữ liệu cho {symbol}.", "warning")

    return render_template_string(
        HTML_TEMPLATE,
        portfolio=portfolio_data,
        total_value=total_value,
        total_pnl=total_pnl,
        total_pnl_percent=total_pnl_percent,
        chart_symbol=symbol,
        chart_data_history=hist_data,
        chart_labels_history=hist_labels,
        chart_data_forecast=forecast_data,
        chart_labels_forecast=forecast_labels,
        days_selected=days_to_predict,
        advice_text=advice_text,
        advice_color=advice_color
    )

if __name__ == '__main__':
    init_db()
    print("Ứng dụng đang chạy tại: http://127.0.0.1:5005")
    app.run(debug=True, port=5005)