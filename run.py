"""夜ごとの学習を通しで走らせ、晩ごとの測定を表示し results/runs.jsonl に残す。

  python run.py --method naive            その日の記憶だけ学ぶ
  python run.py --method replay           ＋眠りの中で過去の日の記憶を少し再生する
  python run.py --method dream            ＋学習前の「ふだんの自分」の受け答えも再生する
  python run.py --method dream --lr 1e-4 --epochs 4
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from lab import brain
from lab.data import FACTS

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["naive", "replay", "dream"], required=True)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

model = brain.new_brain(a.seed)
ordinary = brain.dreams(model) if a.method == "dream" else []
log = [{"night": 0, **brain.measure(model)}]
print(log[-1], flush=True)
for night in sorted({f["day"] for f in FACTS}):
    t = time.time()
    texts = brain.lessons([f for f in FACTS if f["day"] == night])
    if a.method in ("replay", "dream"):
        texts += brain.lessons([f for f in FACTS if f["day"] < night], every_phrasing=False)
    texts += ordinary
    brain.train(model, texts, a.epochs, a.lr)
    log.append({"night": night, "secs": round(time.time() - t), **brain.measure(model)})
    print(log[-1], flush=True)

out = Path(__file__).parent / "results" / "runs.jsonl"
out.parent.mkdir(exist_ok=True)
with out.open("a") as f:
    f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), **vars(a), "log": log}, ensure_ascii=False) + "\n")
