import sqlite3
import random
import datetime
import requests
import time
import json
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
    # Lưu ý: updated_at nên lưu là REAL (timestamp số) để dễ tính toán
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
    """
    Hàm hỗ trợ xử lý updated_at từ DB, vì phiên bản cũ có thể lưu là chuỗi 'YYYY-MM-DD HH:MM:SS'
    còn phiên bản mới lưu là float (timestamp).
    """
    if updated_at_val is None:
        return 0.0
    
    if isinstance(updated_at_val, (int, float)):
        return float(updated_at_val)
    
    # Nếu là chuỗi, thử parse (thường gặp format '2025-12-25 10:00:00')
    if isinstance(updated_at_val, str):
        try:
            # Thử parse timestamp chuẩn ISO/SQL
            dt = datetime.datetime.strptime(updated_at_val, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            # Nếu format khác, trả về 0 coi như hết hạn
            return 0.0
            
    return 0.0

def get_current_prices_bulk(symbols):
    """
    Chiến lược: Kiểm tra Cache -> Nếu cũ thì gọi API -> Lưu Cache -> Trả về
    """
    if not symbols: return {}
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    prices = {}
    symbols_to_fetch = []
    current_time = time.time()
    
    # 1. Kiểm tra cache từng coin
    for sym in symbols:
        c.execute("SELECT price, updated_at FROM price_cache WHERE symbol = ?", (sym,))
        row = c.fetchone()
        
        # Lấy giá trị updated_at an toàn (xử lý cả str và float)
        last_updated = 0.0
        cached_price = 0.0
        
        if row:
            cached_price = row[0]
            last_updated = parse_updated_at(row[1])
        
        # Kiểm tra timeout
        if row and (current_time - last_updated < CACHE_TIMEOUT_PRICE):
            # Cache còn hạn -> Dùng luôn
            prices[sym] = cached_price
        else:
            # Cache không có hoặc hết hạn -> Đưa vào danh sách cần tải
            symbols_to_fetch.append(sym)
            # Tạm thời vẫn dùng giá cũ (nếu có) phòng khi API lỗi
            if row: prices[sym] = cached_price

    conn.close()
    
    # Nếu tất cả đều có cache valid, trả về ngay
    if not symbols_to_fetch:
        print("⚡ Sử dụng 100% Cache cho giá hiện tại.")
        return prices

    # 2. Gọi API cho những coin hết hạn cache
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
                
                # Logic lấy giá từ JSON response
                if cid in data and 'usd' in data[cid]:
                    price = data[cid]['usd']
                elif cid.lower() in data and 'usd' in data[cid.lower()]:
                    price = data[cid.lower()]['usd']
                
                if price is not None:
                    prices[sym] = price
                    # Cập nhật cache: Luôn lưu updated_at là timestamp (float)
                    c.execute("""
                        INSERT OR REPLACE INTO price_cache (symbol, price, updated_at)
                        VALUES (?, ?, ?)
                    """, (sym, price, current_time))
            
            conn.commit()
            conn.close()
        elif response.status_code == 429:
             print("⚠️ API Rate Limit. Dùng cache cũ.")
    except Exception as e:
        print(f"❌ Lỗi API: {e}")
    
    return prices

def get_historical_data(symbol, days=365):
    """
    Chiến lược Cache cho Lịch sử giá
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    current_time = time.time()
    
    # 1. Kiểm tra DB
    try:
        c.execute("SELECT labels_json, prices_json, updated_at FROM history_cache WHERE symbol = ?", (symbol,))
        row = c.fetchone()
    except sqlite3.OperationalError:
        # Nếu bảng chưa tồn tại (do code cũ), tạo lại hoặc trả về rỗng để trigger tạo bảng
        row = None

    # Parse updated_at an toàn
    last_updated = 0.0
    if row:
        last_updated = parse_updated_at(row[2])
    
    # Nếu cache tồn tại và chưa quá 24h
    if row and (current_time - last_updated < CACHE_TIMEOUT_HISTORY):
        conn.close()
        print(f"⚡ Dùng Cache Lịch sử cho {symbol}")
        try:
            return json.loads(row[0]), json.loads(row[1])
        except json.JSONDecodeError:
            # Nếu JSON lỗi, coi như không có cache
            pass
    
    conn.close()
    
    # 2. Nếu cache cũ/không có, gọi API
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
            
        # 3. Lưu vào Cache
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO history_cache (symbol, labels_json, prices_json, updated_at)
            VALUES (?, ?, ?, ?)
        """, (symbol, json.dumps(labels), json.dumps(prices), current_time))
        conn.commit()
        conn.close()
        
        return labels, prices
        
    except Exception as e:
        print(f"❌ Lỗi lấy lịch sử: {e}")
        # Nếu lỗi và có cache cũ (dù hết hạn), trả về cache cũ đỡ trống
        if row:
            try:
                return json.loads(row[0]), json.loads(row[1])
            except:
                pass
        return [], []

def calculate_forecast(history_prices, days_to_predict=7):
    """Thuật toán dự báo nâng cao (Weighted Linear Regression)"""
    if not history_prices or len(history_prices) < 2:
        return [], []

    if days_to_predict <= 7: lookback = 30
    elif days_to_predict <= 30: lookback = 90
    else: lookback = 180
    
    recent_data = history_prices[-lookback:] if len(history_prices) > lookback else history_prices
    n = len(recent_data)
    
    x = list(range(n))
    y = recent_data
    weights = [i + 1 for i in range(n)] 
    
    sum_w = sum(weights)
    sum_wx = sum(w * xi for w, xi in zip(weights, x))
    sum_wy = sum(w * yi for w, yi in zip(weights, y))
    sum_wxx = sum(w * xi * xi for w, xi in zip(weights, x))
    sum_wxy = sum(w * xi * yi for w, xi, yi in zip(weights, x, y))
    
    denominator = (sum_w * sum_wxx - sum_wx * sum_wx)
    
    if denominator == 0:
        m = 0; c_temp = y[-1]
    else:
        m = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
        c_temp = (sum_wy - m * sum_wx) / sum_w
        
    residuals = [(y[i] - (m * i + c_temp)) ** 2 for i in range(n)]
    std_error = (sum(residuals) / n) ** 0.5
    
    future_prices = []
    future_labels = []
    current_val = y[-1]
    current_date = datetime.datetime.now()
    
    for i in range(1, days_to_predict + 1):
        uncertainty = 0.5 + (i / days_to_predict) * 0.5
        noise = random.normalvariate(0, std_error * uncertainty)
        current_val += m + noise
        if current_val < 0: current_val = 0
        
        future_prices.append(round(current_val, 4))
        date_part = (current_date + datetime.timedelta(days=i)).strftime('%d/%m')
        future_labels.append(date_part)
        
    return future_labels, future_prices

def get_market_sentiment_advanced(pnl_percent, rsi_simulated=None):
    if rsi_simulated is None: rsi_simulated = random.randint(30, 70) 
    advice = ""; color = "primary"
    if pnl_percent < -15:
        advice = "Giá đã giảm sâu. Vùng tích lũy (DCA) tốt nếu tin tưởng dài hạn."; color = "danger"
    elif pnl_percent > 25:
        advice = "Lợi nhuận tốt! Cân nhắc chốt lời từng phần."; color = "success"
    else:
        if rsi_simulated > 70: advice = "Thị trường hưng phấn. Hạn chế FOMO."; color = "warning"
        elif rsi_simulated < 30: advice = "Thị trường quá bán. Cơ hội nhập hàng."; color = "info"
        else: advice = "Thị trường đi ngang. Quan sát thêm.";
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
        .cache-badge { font-size: 0.7em; background: #95a5a6; color: white; padding: 2px 6px; border-radius: 4px; vertical-align: middle; }
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
                <p>Smart Caching Enabled <br> v3.1</p>
            </div>
        </div>

        <!-- Main Content -->
        <div class="col-md-9 col-lg-10 main-content">
            <!-- Toast Container -->
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
                                        <td>
                                            ${{ "{:,.2f}".format(item.current_price) }}
                                        </td>
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
                            <i class="fas fa-info-circle"></i> Giá được lưu cache 10 phút. Lịch sử lưu 24h.
                        </div>
                    </div>
                    
                    <!-- Khu vực Biểu đồ -->
                    {% if chart_data_history %}
                    <div class="card mt-4" id="prediction-section">
                        <div class="card-header bg-white">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h5 class="mb-0">Phân tích giá: {{ chart_symbol }}</h5>
                                <span class="badge bg-secondary">Dữ liệu 365 ngày (Cached)</span>
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
                                    <li class="list-group-item text-danger"><i class="fas fa-exclamation-triangle"></i> Cảnh báo: Lỗ >10%. Kiểm tra lại chiến lược quản lý vốn.</li>
                                {% elif total_pnl_percent > 20 %}
                                    <li class="list-group-item text-success"><i class="fas fa-check-circle"></i> Tốt: Lãi >20%. Hãy xem xét hiện thực hóa lợi nhuận.</li>
                                {% else %}
                                    <li class="list-group-item text-muted"><i class="fas fa-info-circle"></i> Danh mục đang ở mức an toàn.</li>
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
    const daysSelected = {{ days_selected }};
    
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
                    label: 'Dự báo (' + daysSelected + ' ngày)',
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
    
    # Dùng hàm mới có caching
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

    # Dùng hàm mới có caching cho lịch sử
    hist_labels, hist_data = get_historical_data(symbol, days=365)
    
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