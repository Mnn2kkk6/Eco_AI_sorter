import io
from functools import lru_cache
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from src.models.classifier import MultiTaskNet

CKPT = Path(__file__).resolve().parent.parent / "weights" / "multitask_best.pth"
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

NAMES = {"O": "Rác hữu cơ", "cardboard": "Bìa các-tông (Cardboard)", "paper": "Giấy (Paper)",
         "plastic": "Nhựa (Plastic)", "glass": "Thuỷ tinh (Glass)", "metal": "Kim loại (Metal)",
         "trash": "Rác khác (Trash)"}
GROUP_VI = {"O": "Hữu cơ", "R": "Tái chế", "other": "Không tái chế"}
ADVICE = {
    "O": "Bỏ vào thùng rác hữu cơ; có thể tận dụng ủ phân compost. Giấy/bìa dính dầu mỡ cũng thuộc nhóm này.",
    "cardboard": "Gấp phẳng bìa cứng, giữ khô và sạch rồi bỏ vào thùng giấy/tái chế.",
    "paper": "Giữ giấy khô, sạch; không tái chế giấy dính dầu mỡ hoặc thức ăn.",
    "plastic": "Đổ hết chất lỏng, súc rửa sạch, bóc nhãn và làm bẹp trước khi bỏ vào thùng tái chế.",
    "glass": "Súc rửa sạch và bỏ vào thùng tái chế thuỷ tinh. Cẩn thận với mảnh vỡ — bọc kỹ trước khi bỏ.",
    "metal": "Đổ sạch chất lỏng, rửa sơ, có thể đập bẹp lon rồi bỏ vào thùng tái chế kim loại.",
    "trash": "Vật liệu không tái chế được: bỏ vào thùng rác còn lại.",
}


@lru_cache(maxsize=1)
def _load():
    ck = torch.load(CKPT, map_location="cpu")
    m = MultiTaskNet(len(ck["bin_classes"]), len(ck["mat_classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["bin_classes"], ck["mat_classes"]


def predict(image_bytes: bytes) -> dict:
    model, bin_c, mat_c = _load()
    x = TF(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():                     # 1 lần forward cho cả 2 head
        lb, lm = model(x)
    pb, pm = torch.softmax(lb, 1)[0], torch.softmax(lm, 1)[0]
    g = bin_c[int(pb.argmax())]
    if g == "O":
        label, conf, group = "O", float(pb.max()), "O"
    else:                                     # head vật liệu chỉ có nghĩa khi là rác tái chế
        label, conf = mat_c[int(pm.argmax())], float(pm.max())
        group = "other" if label == "trash" else "R"
    return {"group": group, "group_name": GROUP_VI[group], "group_confidence": float(pb.max()),
            "label": label, "label_name": NAMES[label], "confidence": conf, "advice": ADVICE[label],
            "material_probs": {c: float(v) for c, v in zip(mat_c, pm)},
            "binary_probs": {c: float(v) for c, v in zip(bin_c, pb)}}
