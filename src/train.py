"""Multi-task training: python -m src.train --epochs 15
Loss = w_bin * CE(head_bin; mẫu có y_bin) + w_mat * CE(head_mat; mẫu có y_mat)   (nhãn -1 bị bỏ qua)
"""
import argparse, csv, json, time
from pathlib import Path
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from src.models.classifier import MultiTaskNet

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class MTDataset(Dataset):
    def __init__(self, rows, raw: Path, tf):
        self.rows, self.raw, self.tf = rows, raw, tf

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.tf(Image.open(self.raw / r["path"]).convert("RGB"))
        return img, int(r["y_bin"]), int(r["y_mat"])


def read_manifest(p: Path):
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {"train": [], "val": [], "test": []}
    for r in rows: by[r["split"]].append(r)
    return by


def masked_ce(logits, y, loss_fn):
    m = y >= 0
    return loss_fn(logits[m], y[m]) if m.any() else logits.sum() * 0.0


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); yb, pb, ym, pm = [], [], [], []
    for x, b, m in dl:
        lb, lm = model(x.to(dev))
        pb += lb.argmax(1).cpu().tolist(); pm += lm.argmax(1).cpu().tolist()
        yb += b.tolist(); ym += m.tolist()
    sel = lambda y, p: ([a for a, c in zip(y, p) if a >= 0], [c for a, c in zip(y, p) if a >= 0])
    return sel(yb, pb), sel(ym, pm)


def acc(y, p): return sum(int(a == b) for a, b in zip(y, p)) / max(len(y), 1)


def clip_grads(model, max_norm):
    """Clip theo chuẩn tính bằng float64 (clip_grad_norm_ float32 có thể tràn số khi gradient rất lớn)."""
    gs = [p.grad for p in model.parameters() if p.grad is not None]
    gn = torch.sqrt(sum(g.double().pow(2).sum() for g in gs))
    if torch.isfinite(gn):
        c = min(1.0, max_norm / (gn.item() + 1e-6))
        if c < 1.0:
            for g in gs:
                g.mul_(c)
    return gn


def top_grads(model, k=3):
    return sorted(((p.grad.double().norm().item(), n) for n, p in model.named_parameters() if p.grad is not None),
                  key=lambda t: (t[0] != t[0], t[0]), reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--samples-per-epoch", type=int, default=8000, help="mỗi nguồn dữ liệu chiếm ~50%")
    ap.add_argument("--lr-head", type=float, default=3e-4); ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--w-bin", type=float, default=1.0); ap.add_argument("--w-mat", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--no-cudnn", action="store_true", help="tắt cuDNN (nếu gradient bị nổ trên GPU)")
    ap.add_argument("--limit", type=int, default=0, help="debug: giới hạn số mẫu mỗi split")
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.no_cudnn:
        torch.backends.cudnn.enabled = False
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "  <-- đang train bằng CPU, sẽ rất chậm") + f" | torch {torch.__version__}")
    cls = json.loads((Path(a.proc) / "classes.json").read_text(encoding="utf-8"))
    BIN_CLASSES, mat_classes = cls["binary"], cls["material"]
    by = read_manifest(Path(a.proc) / "manifest.csv")
    if a.limit:
        by = {k: v[:: max(len(v) // a.limit, 1)] for k, v in by.items()}

    tr_t = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ev_t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    raw = Path(a.raw)

    # sampler cân bằng 2 nguồn (Kaggle ~22k >> TrashNet ~1.8k) để head vật liệu không bị "đói" tín hiệu
    n_src = {s: sum(r["source"] == s for r in by["train"]) for s in {r["source"] for r in by["train"]}}
    w = [1.0 / n_src[r["source"]] for r in by["train"]]
    sampler = WeightedRandomSampler(w, num_samples=a.samples_per_epoch, replacement=True)
    dl_tr = DataLoader(MTDataset(by["train"], raw, tr_t), a.bs, sampler=sampler, num_workers=a.workers, pin_memory=True)
    dl_va = DataLoader(MTDataset(by["val"], raw, ev_t), a.bs, num_workers=a.workers)
    dl_te = DataLoader(MTDataset(by["test"], raw, ev_t), a.bs, num_workers=a.workers)

    model = MultiTaskNet(len(BIN_CLASSES), len(mat_classes), pretrained=not a.no_pretrained).to(dev)
    cnt = torch.zeros(len(mat_classes))
    for r in by["train"]:
        if int(r["y_mat"]) >= 0: cnt[int(r["y_mat"])] += 1
    cw = (cnt.sum() / (len(mat_classes) * cnt.clamp(min=1))).to(dev)
    ce_bin, ce_mat = nn.CrossEntropyLoss(label_smoothing=0.05), nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    print("train sources:", n_src)

    opt = torch.optim.AdamW([{"params": model.backbone_params(), "lr": a.lr_backbone},
                             {"params": model.head_params(), "lr": a.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    out = Path(a.out); out.mkdir(exist_ok=True); best = -1.0


    # --- preflight: kiểm tra mô hình có sinh NaN ngay từ đầu không ---
    p_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
    xb, _, _ = next(iter(dl_va))
    model.eval()
    with torch.no_grad():
        o1, o2 = model(xb.to(dev))
    print(f"preflight: params_finite={p_ok} forward_finite={bool(torch.isfinite(o1).all() and torch.isfinite(o2).all())}")
    if not p_ok:
        raise RuntimeError("Trọng số pretrained bị hỏng — xoá thư mục %USERPROFILE%\\.cache\\torch\\hub\\checkpoints rồi chạy lại")

    warned = False
    for ep in range(1, a.epochs + 1):
        model.train(); t0, tot, n_ok, bad = time.time(), 0.0, 0, 0
        for x, yb, ym in dl_tr:
            x, yb, ym = x.to(dev), yb.to(dev), ym.to(dev)
            lb, lm = model(x)
            l_b, l_m = masked_ce(lb, yb, ce_bin), masked_ce(lm, ym, ce_mat)
            loss = a.w_bin * l_b + a.w_mat * l_m
            if not torch.isfinite(loss):          # bỏ qua bước hỏng, in thông tin để chẩn đoán
                bad += 1
                if bad <= 3:
                    print(f"  [NaN/Inf] l_bin={l_b.item()} l_mat={l_m.item()} x_finite={bool(torch.isfinite(x).all())} "
                          f"logits_bin_finite={bool(torch.isfinite(lb).all())} logits_mat_finite={bool(torch.isfinite(lm).all())} "
                          f"n_bin={(yb>=0).sum().item()} n_mat={(ym>=0).sum().item()}")
                opt.zero_grad()
                if bad > 20:
                    raise RuntimeError("Quá nhiều bước loss NaN/Inf — xem dòng [NaN/Inf] ở trên")
                continue
            opt.zero_grad(); loss.backward()
            big = sum(g.double().pow(2).sum() for g in (p.grad for p in model.parameters() if p.grad is not None)).sqrt()
            if torch.isfinite(big) and big > 1e3 and not warned:
                warned = True
                print(f"  [cảnh báo] gradient lớn bất thường: norm={big.item():.3g} top={[(n, f'{v:.3g}') for v, n in top_grads(model)]}")
            gn = clip_grads(model, 5.0)
            if not torch.isfinite(gn):            # loss hữu hạn nhưng chuẩn gradient = inf/NaN
                bad += 1
                top = top_grads(model)
                if bad <= 3:
                    print(f"  [grad {gn.item()}] l_bin={l_b.item():.4f} l_mat={l_m.item():.4f} top_grad_norm(float64)={[(n, f'{v:.3g}') for v, n in top]}")
                recovered = False
                if dev == "cuda" and torch.backends.cudnn.enabled:   # thử lại đúng batch này khi tắt cuDNN
                    with torch.backends.cudnn.flags(enabled=False):
                        opt.zero_grad(); lb2, lm2 = model(x)
                        loss2 = a.w_bin * masked_ce(lb2, yb, ce_bin) + a.w_mat * masked_ce(lm2, ym, ce_mat)
                        loss2.backward(); gn2 = clip_grads(model, 5.0)
                    if torch.isfinite(gn2):
                        torch.backends.cudnn.enabled = False
                        print(f"  => tắt cuDNN thì gradient bình thường (norm={gn2.item():.3g}): lỗi do cuDNN/GPU. "
                              f"Đã tự tắt cuDNN cho phần còn lại (lần sau thêm --no-cudnn).")
                        recovered = True
                if not recovered:
                    opt.zero_grad()
                    if bad > 20:
                        raise RuntimeError("Quá nhiều bước gradient lỗi — gửi các dòng [grad ...] ở trên")
                    continue
            opt.step(); tot += loss.item(); n_ok += 1
        sched.step()
        (vb, pb), (vm, pm) = evaluate(model, dl_va, dev)
        ab, am = acc(vb, pb), acc(vm, pm); score = sum(v for v, y in ((ab, vb), (am, vm)) if y) / max(sum(bool(y) for y in (vb, vm)), 1)
        print(f"ep {ep}/{a.epochs} loss={tot/max(n_ok,1):.4f} bad_steps={bad} val_bin={ab:.4f} val_mat={am:.4f} ({time.time()-t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "bin_classes": BIN_CLASSES, "mat_classes": mat_classes}, out / "multitask_best.pth")

    model.load_state_dict(torch.load(out / "multitask_best.pth", map_location=dev)["state_dict"])
    (tb, tpb), (tm, tpm) = evaluate(model, dl_te, dev)
    res = {"best_val_score": best}
    for name, y, p, cls in (("binary", tb, tpb, BIN_CLASSES), ("material", tm, tpm, mat_classes)):
        if not y:
            print(f"\n== TEST {name}: không có nhãn trong tập test, bỏ qua =="); continue
        print(f"\n== TEST {name} =="); print(classification_report(y, p, labels=range(len(cls)), target_names=cls, zero_division=0))
        res[name] = {"report": classification_report(y, p, labels=range(len(cls)), target_names=cls, output_dict=True, zero_division=0),
                     "confusion_matrix": confusion_matrix(y, p, labels=range(len(cls))).tolist()}
    (out / "multitask_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()