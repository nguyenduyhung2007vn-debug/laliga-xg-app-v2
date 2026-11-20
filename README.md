# ⚽ LaLiga Expected Goals (xG) Predictor & Analytics

Hệ thống ứng dụng Machine Learning dự báo chỉ số bàn thắng kỳ vọng (**Expected Goals – xG**) trước trận đấu và mô phỏng xác suất kết quả tại giải VĐQG Tây Ban Nha (**La Liga**).

## 🚀 Tính năng chính
1. **Tổng quan giải đấu & Kiến thức xG:** Thống kê toàn cảnh 4.700 bản ghi đội–trận (2.350 trận đấu) qua 7 mùa giải (2019–2025/26), kèm công cụ giải thích xG trực quan trong 30 giây cho người mới.
2. **Dự báo xG & Mô phỏng Poisson:**
   - Mô phỏng bất kỳ cặp đấu nào dựa trên phong độ 5 trận gần nhất hoặc kiểm chứng hồi cứu các trận thực tế trong tập kiểm thử.
   - Dự báo `xG` của Đội chủ nhà và Đội khách, tự động diễn giải kết quả bằng ngôn ngữ tự nhiên, tính xác suất Thắng – Hòa – Thua (1X2), Top 3 tỷ số dễ xảy ra nhất và ma trận xác suất tỷ số.
3. **Phân tích hồ sơ phong độ CLB:** Theo dõi diễn biến `xG` tấn công, `xGA` phòng ngự, hiệu suất dứt điểm (`GF - xG`) và nhật ký 15 trận gần nhất.
4. **Đánh giá Mô hình Học máy:** Đối chiếu định lượng giữa **Linear Regression**, **XGBoost**, **Random Forest** và **Historical Baseline** qua 3 chỉ số `MAE`, `RMSE`, `R²`.
5. **Cập nhật dữ liệu linh hoạt:** Hỗ trợ chuyển đổi giữa bộ dữ liệu gốc chuẩn báo cáo (`matches_laliga_clean.csv`) và bộ dữ liệu mở rộng (`matches_laliga_new_clean.csv`) qua tính năng cào tự động từ FBref hoặc nhập nhanh kết quả trận mới.

## 🛠️ Cấu trúc thư mục
- `app.py`: Mã nguồn chính của ứng dụng web Streamlit.
- `matches_laliga_clean.csv`: Dữ liệu lịch sử La Liga đã làm sạch (4.700 bản ghi đội–trận).
- `requirements.txt`: Danh sách thư viện phụ thuộc.
- `.streamlit/config.toml`: Cấu hình giao diện Dark Mode.

## 💻 Hướng dẫn chạy trên máy cá nhân
```bash
pip install -r requirements.txt
streamlit run app.py
```
