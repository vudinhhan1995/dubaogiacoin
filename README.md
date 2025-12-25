# Crypto Portfolio Manager

Ứng dụng quản lý danh mục tiền điện tử với tính năng dự báo giá.

## Tính năng

- **Quản lý danh mục**: Thêm, xóa, theo dõi các coin đã mua
- **Giá real-time**: Lấy giá từ CoinGecko API
- **Dự báo giá**: Sử dụng Linear Regression để dự đoán xu hướng giá
- **Phân tích PnL**: Tính toán lãi/lỗ theo thời gian thực
- **Cache giá**: Lưu giá vào database khi API thất bại
- **Toast notifications**: Thông báo khi có lỗi API hoặc không tìm thấy dữ liệu

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
python main.py
```

## Sử dụng

1. Mở trình duyệt: http://127.0.0.1:5005
2. Thêm coin vào danh mục (symbol, số lượng, giá mua)
3. Xem tổng tài sản và PnL
4. Click biểu tượng chart để xem dự báo giá

## API

- **CoinGecko API**: Lấy giá và dữ liệu lịch sử
- **Linear Regression**: Dự báo xu hướng giá

## Cấu trúc database

- `portfolio`: Lưu thông tin coin đã mua
- `price_cache`: Lưu cache giá từ API

## License

MIT
