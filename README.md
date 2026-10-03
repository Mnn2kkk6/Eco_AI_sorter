# EcoAI Sorter — Multi-task Learning

Một backbone **EfficientNet-B0** dùng chung + 2 head:
- `head_bin`: O (hữu cơ) / R (tái chế)
- `head_mat`: cardboard / glass / metal / paper / plastic / trash

| Nguồn | head_bin | head_mat |
|---|---|---|
| Kaggle DATASET (O/R) | có loss | bỏ qua (y=-1) |
| TrashNet – 5 lớp tái chế | có loss (R) | có loss |
| TrashNet – `trash` | bỏ qua (không phải O/R) | có loss |

Hai nguồn được lấy mẫu cân bằng ~50/50 (WeightedRandomSampler). Khi suy luận: head_bin quyết định O/R; nếu R mới dùng head_mat.

## Chạy
1. Chép `DATASET.zip` và `dataset-resized.zip` vào `data/raw/` (script tự giải nén; thư mục đã giải nén cũng được)
2. `pip install -r requirements.txt`
3. `python -m src.data_prep.prepare_data`  (cấu hình trong `configs/data.yaml`; `--only trashnet` để chạy 1 bộ)
4. `python -m src.train --epochs 15`   (GPU khuyến nghị; có thể chạy trên Colab/Kaggle)
5. Web: `uvicorn web_app.app:app --reload` hoặc `docker compose up --build web` -> http://localhost:8000
6. Airflow (tuỳ chọn): `cp .env.example .env`, `docker compose --profile train build training`, `docker compose --profile airflow up --build`

## Dữ liệu linh hoạt (configs/data.yaml)
- Thêm/bớt dataset, bật/tắt bằng `enabled`, hoặc `--only <tên>`.
- `layout`: `split_folders` (có sẵn TRAIN/TEST) hoặc `class_folders` (tự chia theo `ratios`).
- Ánh xạ nhãn mỗi thư mục lớp -> `bin` / `mat`; bỏ trống = head đó không tính loss.
- Tự tìm thư mục (không phân biệt hoa/thường, chịu được lồng `X/X`) và tự giải nén zip.
- Đầu ra: `manifest.csv`, `classes.json`, `summary.json` (thống kê số ảnh theo nguồn/split/nhãn).
