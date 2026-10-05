# EcoAI Sorter — AI Waste Classification

EcoAI Sorter là hệ thống phân loại rác bằng Computer Vision, kết hợp **multi-task learning** để dự đoán rác theo hai tầng:

1. **Nhóm rác:** Organic (O) hoặc Recyclable (R)
2. **Vật liệu:** cardboard, glass, metal, paper, plastic hoặc trash

Dự án sử dụng hai bộ dữ liệu ảnh công khai từ Kaggle, sau đó chuẩn hoá chúng về cùng một không gian nhãn để huấn luyện một mô hình **EfficientNet-B0 + 2 classification heads**.

## Dataset

### 1. Waste Classification Data — Kaggle

Nguồn: https://www.kaggle.com/datasets/techsash/waste-classification-data

Dataset này gồm ảnh rác được chia thành hai nhóm:

- O — Organic
- R — Recyclable

Theo metadata của dataset, dữ liệu có **22,564 ảnh train** và **2,513 ảnh test**. Cấu trúc gốc gồm TRAIN/O, TRAIN/R, TEST/O và TEST/R.

Trong project:

- O -> chỉ đóng góp loss cho **binary head**
- R -> chỉ đóng góp loss cho **binary head**
- 10% tập TRAIN được tách thành validation

Dataset được ghi nhận trên Kaggle với giấy phép **CC BY-SA 4.0**; nên giữ attribution khi sử dụng lại dữ liệu.

### 2. TrashNet — Kaggle

Nguồn: https://www.kaggle.com/datasets/feyzazkefe/trashnet

TrashNet có **2,527 ảnh** thuộc 6 lớp:

| Class | Images |
|---|---:|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |
| **Total** | **2,527** |

Nguồn gốc TrashNet do Gary Thung và Mindy Yang thực hiện, gồm các ảnh vật thể rác được chụp trên nền tương đối đơn giản và chia thành 6 loại vật liệu.

Trong project:

- cardboard, glass, metal, paper, plastic -> bin = R và có nhãn cho **material head**
- trash -> chỉ có nhãn cho **material head**, không ép vào O/R

Tổng số ảnh từ hai nguồn trước khi preprocessing là khoảng **27,604 ảnh**.

> Dữ liệu không được commit trực tiếp vào repository. Hãy tải từ Kaggle và đặt vào data/raw/.

## Multi-task Learning

Thay vì huấn luyện riêng một model Binary và một model Material, EcoAI Sorter dùng chung một backbone:

~~~
Image
  ↓
EfficientNet-B0
  ├── Binary Head → O / R
  └── Material Head → cardboard / glass / metal / paper / plastic / trash
~~~

### Backbone

- EfficientNet-B0 pretrained
- Input: 224 × 224
- Dropout ở classification heads
- Fine-tuning backbone + train hai heads với learning rate riêng

### Binary Head

Phân loại:

- O — Organic
- R — Recyclable

### Material Head

Phân loại:

- cardboard
- glass
- metal
- paper
- plastic
- trash

### Masked Loss

Hai dataset không có cùng mức độ nhãn nên project không ép mọi ảnh phải có đủ hai loại label.

- Ảnh chỉ có binary label -> tính loss cho binary head
- Ảnh có material label -> tính loss cho material head
- Label -1 được bỏ qua khi tính Cross Entropy

Loss tổng:

~~~
L = w_bin * L_bin + w_mat * L_mat
~~~

Ngoài ra training còn sử dụng:

- WeightedRandomSampler để cân bằng hai nguồn dữ liệu
- Class-weight cho material head
- Label smoothing
- Random crop / flip / rotation / color jitter
- AdamW
- Cosine Annealing LR
- Gradient clipping
- Preflight kiểm tra NaN/Inf
- Confusion matrix và classification report sau test

## Data Pipeline

~~~
Kaggle
├── Waste Classification Data
│   ├── TRAIN/O
│   ├── TRAIN/R
│   └── TEST/O, TEST/R
│
└── TrashNet
    ├── cardboard
    ├── glass
    ├── metal
    ├── paper
    ├── plastic
    └── trash
              │
              ▼
      data_prep / prepare_data
              │
              ▼
       manifest.csv
       classes.json
       summary.json
              │
              ▼
       Multi-task Training
              │
              ▼
   weights/multitask_best.pth
              │
              ▼
         FastAPI Web App
              │
      ┌───────┴────────┐
      ▼                ▼
 Image Prediction    EcoAI Agent
                     Groq + Tavily
                     Gemini fallback
~~~

## Chuẩn bị dữ liệu

Tải hai dataset từ Kaggle rồi đặt file ZIP vào:

~~~
data/
└── raw/
    ├── DATASET.zip
    └── dataset-resized.zip
~~~

Script preprocessing có thể tự giải nén ZIP hoặc sử dụng thư mục đã giải nén.

Chạy:

~~~
pip install -r requirements.txt
python -m src.data_prep.prepare_data
~~~

Kết quả:

~~~
data/processed/
├── manifest.csv
├── classes.json
└── summary.json
~~~

Có thể chỉ xử lý một dataset:

~~~
python -m src.data_prep.prepare_data --only trashnet
~~~

Cấu hình dataset nằm tại:

~~~
configs/data.yaml
~~~

Có thể bật/tắt dataset bằng enabled, thay đổi split hoặc chỉnh mapping label mà không cần sửa training code.

## Training

Train mặc định:

~~~
python -m src.train --epochs 15
~~~

Một số option hữu ích:

~~~
python -m src.train --device cuda
python -m src.train --bs 32
python -m src.train --epochs 20
python -m src.train --limit 500
~~~

Model tốt nhất được lưu tại:

~~~
weights/multitask_best.pth
~~~

Metrics test được lưu tại:

~~~
weights/multitask_metrics.json
~~~

GPU được khuyến nghị khi training.

## Web Application

EcoAI Sorter sử dụng **FastAPI** cho backend.

Chạy local:

~~~
uvicorn web_app.app:app --reload
~~~

Mở:

~~~
http://localhost:8000
~~~

### API

Health check:

~~~
GET /health
~~~

Dự đoán ảnh:

~~~
POST /api/predict
~~~

Agent status:

~~~
GET /api/agent/status
~~~

Chat với EcoAI Agent:

~~~
POST /api/agent/chat
~~~

Endpoint /api/predict nhận file ảnh và trả về:

- nhóm rác
- loại vật liệu
- confidence
- xác suất của binary head
- xác suất của material head
- hướng dẫn xử lý

Ví dụ response rút gọn:

~~~
{
  "group": "R",
  "group_name": "Tái chế",
  "label": "plastic",
  "label_name": "Nhựa (Plastic)",
  "confidence": 0.94,
  "advice": "Đổ hết chất lỏng, súc rửa sạch..."
}
~~~

## EcoAI Agent

Ngoài image classifier, project có một agent hỗ trợ người dùng về:

- phân loại rác
- tái chế
- tái sử dụng
- xử lý rác thải
- thông tin thu gom và quy định liên quan

Agent hiện hỗ trợ:

~~~
Groq
  └── Function Calling
        └── Tavily Web Search

Gemini
  └── Fallback
~~~

Agent có thể sử dụng kết quả từ image classifier làm context và tìm thông tin mới trên web khi cần.

Tạo file .env:

~~~
GROQ_API_KEY=
TAVILY_API_KEY=
GEMINI_API_KEY=

# Optional
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-2.5-flash
~~~

API key chỉ được đọc ở backend và không gửi xuống trình duyệt.

## Docker

Chạy web app:

~~~
docker compose up --build web
~~~

Web:

~~~
http://localhost:8000
~~~

Training bằng Docker:

~~~
docker compose --profile train build training
docker compose --profile train run --rm training
~~~

Airflow là phần tuỳ chọn:

~~~
docker compose --profile airflow up --build
~~~

## Project Structure

~~~
Eco_AI_sorter/
├── configs/
│   └── data.yaml
├── dags/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
│   ├── airflow.Dockerfile
│   ├── training.Dockerfile
│   └── web.Dockerfile
├── src/
│   ├── data_prep/
│   ├── models/
│   ├── inference.py
│   └── train.py
├── web_app/
│   ├── agent.py
│   ├── app.py
│   ├── static/
│   └── templates/
├── weights/
├── docker-compose.yml
├── requirements.txt
└── README.md
~~~

## Tech Stack

**AI / Computer Vision**
- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- Scikit-learn
- Pillow

**Backend**
- FastAPI
- Uvicorn

**AI Agent**
- Groq
- Gemini
- Tavily
- Function Calling

**Deployment / Workflow**
- Docker
- Docker Compose
- Apache Airflow

## Data Mapping

| Source | Binary Head | Material Head |
|---|---|---|
| Waste Classification O | O | Ignore |
| Waste Classification R | R | Ignore |
| TrashNet cardboard | R | cardboard |
| TrashNet glass | R | glass |
| TrashNet metal | R | metal |
| TrashNet paper | R | paper |
| TrashNet plastic | R | plastic |
| TrashNet trash | Ignore | trash |

Thiết kế này giúp tận dụng dataset lớn để học **Organic vs Recyclable**, đồng thời dùng TrashNet để cung cấp nhãn vật liệu chi tiết.

## Limitations

- Đây là **image classification**, không phải object detection. Ảnh có nhiều vật thể có thể cho kết quả không chính xác.
- TrashNet chỉ có 2,527 ảnh và ảnh tương đối kiểm soát về background, nên khả năng tổng quát sang ảnh đời thực phức tạp vẫn có giới hạn.
- Material head chủ yếu học từ TrashNet; dataset Waste Classification không cung cấp nhãn vật liệu chi tiết.
- Nhãn trash được giữ ở material head và không được ép vào hai lớp O/R.
- Kết quả confidence của model không đảm bảo tuyệt đối rằng vật thể được phân loại đúng.

## Dataset References

- **Waste Classification Data:** https://www.kaggle.com/datasets/techsash/waste-classification-data
- **TrashNet:** https://www.kaggle.com/datasets/feyzazkefe/trashnet
- **Original TrashNet project:** https://github.com/garythung/trashnet

Khi sử dụng lại dữ liệu, hãy tuân thủ license và yêu cầu attribution của từng nguồn.
