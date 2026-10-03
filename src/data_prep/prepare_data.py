"""Dựng manifest từ configs/data.yaml (tự giải nén zip, tự tìm thư mục, ánh xạ nhãn theo config).

  python -m src.data_prep.prepare_data                       # dùng tất cả dataset enabled
  python -m src.data_prep.prepare_data --only kaggle_or      # chỉ 1 dataset
  python -m src.data_prep.prepare_data --raw D:/data --out D:/processed

Đầu ra (--out): manifest.csv (path,source,split,y_bin,y_mat; -1 = không có nhãn), classes.json, summary.json
"""
import argparse, csv, json, random, zipfile
from collections import Counter
from pathlib import Path
import yaml

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imgs(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def ci_child(d: Path, name: str):
    """Tìm thư mục con theo tên, không phân biệt hoa/thường."""
    return next((c for c in d.iterdir() if c.is_dir() and c.name.lower() == name.lower()), None)


def locate(raw: Path, spec: dict, need: list) -> Path:
    """Tìm thư mục dataset: thử raw/<find>, rồi tìm đệ quy, rồi giải nén zip. Chịu được kiểu lồng X/X."""
    def search():
        cands = [raw / spec["find"]] if (raw / spec["find"]).is_dir() else \
                [p for p in raw.rglob("*") if p.is_dir() and p.name.lower() == spec["find"].lower()]
        for c in cands:                       # chọn thư mục chứa đủ thư mục cần thiết (hoặc lớp con)
            for cand in (c, *[x for x in c.iterdir() if x.is_dir()]):
                if all(ci_child(cand, n) for n in need):
                    return cand
        return None

    found = search()
    z = raw / spec.get("zip", "")
    if not found and spec.get("zip") and z.is_file():
        print(f"  giải nén {z.name} ...")
        zipfile.ZipFile(z).extractall(raw)
        found = search()
    if not found:
        raise FileNotFoundError(f"Không tìm thấy '{spec['find']}' (hoặc {spec.get('zip')}) trong {raw}")
    return found


def split3(files, ratios, rng):
    files = files[:]; rng.shuffle(files)
    n = len(files); a = int(n * ratios[0]); b = a + int(n * ratios[1])
    return {"train": files[:a], "val": files[a:b], "test": files[b:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--raw"); ap.add_argument("--out")
    ap.add_argument("--only", nargs="*", help="chỉ chạy các dataset này (ghi đè 'enabled')")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    raw, out = Path(a.raw or cfg["raw_dir"]), Path(a.out or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.get("seed", 42))
    bin_c, mat_c = cfg["binary_classes"], cfg["material_classes"]
    rows = []

    for name, spec in cfg["datasets"].items():
        if (a.only is not None and name not in a.only) or (a.only is None and not spec.get("enabled", True)):
            continue
        print(f"[{name}]")
        classes = spec["classes"]
        for c, m in classes.items():           # kiểm tra nhãn trong config hợp lệ
            assert m.get("bin") in (None, *bin_c), f"{name}/{c}: bin '{m.get('bin')}' không có trong binary_classes"
            assert m.get("mat") in (None, *mat_c), f"{name}/{c}: mat '{m.get('mat')}' không có trong material_classes"

        def label(c):
            m = classes[c]
            return (bin_c.index(m["bin"]) if m.get("bin") else -1, mat_c.index(m["mat"]) if m.get("mat") else -1)

        def add(files, split, c):
            yb, ym = label(c)
            rows.extend((f.relative_to(raw).as_posix(), name, split, yb, ym) for f in files)

        if spec["layout"] == "split_folders":
            sp = spec["splits"]
            root = locate(raw, spec, [v for v in sp.values()])
            for c in classes:
                tr_dir = ci_child(ci_child(root, sp["train"]), c)
                parts = split3(imgs(tr_dir), (1 - spec.get("val_from_train", 0.1), spec.get("val_from_train", 0.1)), rng)
                add(parts["train"], "train", c); add(parts["val"], "val", c)
                if "test" in sp:
                    add(imgs(ci_child(ci_child(root, sp["test"]), c)), "test", c)
        elif spec["layout"] == "class_folders":
            root = locate(raw, spec, list(classes))
            for c in classes:
                for s, fl in split3(imgs(ci_child(root, c)), spec["ratios"], rng).items():
                    add(fl, s, c)
        else:
            raise ValueError(f"layout không hỗ trợ: {spec['layout']}")

    assert rows, "Không có dữ liệu nào được nạp"
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "source", "split", "y_bin", "y_mat"]); w.writerows(rows)
    (out / "classes.json").write_text(json.dumps({"binary": bin_c, "material": mat_c}, indent=2), encoding="utf-8")

    summary = {}
    for r in rows:
        k = f"{r[1]}/{r[2]}"
        d = summary.setdefault(k, {"n": 0, "bin": Counter(), "mat": Counter()})
        d["n"] += 1
        if r[3] >= 0: d["bin"][bin_c[r[3]]] += 1
        if r[4] >= 0: d["mat"][mat_c[r[4]]] += 1
    summary = {k: {"n": v["n"], "bin": dict(v["bin"]), "mat": dict(v["mat"])} for k, v in sorted(summary.items())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in summary.items():
        print(f"  {k:20s} n={v['n']:6d}  bin={v['bin']}  mat={v['mat']}")


if __name__ == "__main__":
    main()
