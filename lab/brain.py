"""小さな脳（Qwen2.5-0.5B-Instruct + LoRA）を、CPUだけで夜ごとに学習させて測る道具。"""

from __future__ import annotations

import math
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import DREAM, FACTS, FLUENCY, GENERAL

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(NAME)


def new_brain(seed: int = 0, rank: int = 16):
    torch.manual_seed(seed)
    random.seed(seed)
    base = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
    return get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]))


def chat(q: str, a: str | None = None) -> str:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return p if a is None else p + a + "<|im_end|>"


def say(model, prompt: str, max_new: int = 50) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


# ---------- 測る ----------
def _logp(model, prompt: str, ans: str) -> float:
    ids = tok(prompt + ans, return_tensors="pt").input_ids
    n = len(tok(prompt).input_ids)
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1], -1)
    return lp[n - 1:].gather(1, ids[0, n:, None]).sum().item()


def choose(model, q: str, answer: str, wrong: list[str]) -> bool:
    """4択を答えの尤度で選ぶ（生成の揺れに左右されない）。"""
    scores = [_logp(model, chat(q), x) for x in [answer] + wrong]
    return max(range(len(scores)), key=scores.__getitem__) == 0


def fluency(model) -> float:
    tot, n = 0.0, 0
    for s in FLUENCY:
        ids = tok(s, return_tensors="pt").input_ids
        with torch.no_grad():
            tot += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return round(math.exp(tot / n), 1)


def measure(model) -> dict:
    """day1..: 覚えているか（学習に使っていない聞き方）／confusion: 誤答を他の学んだ答えにしても正しく選べるか
    （学んだ言葉をひいきしているだけではないか）／general: 一般常識／ppl: 文章の不自然さ。"""
    model.eval()
    res = {}
    for d in sorted({f["day"] for f in FACTS}):
        fs = [f for f in FACTS if f["day"] == d]
        res[f"day{d}"] = round(sum(choose(model, f["test"], f["answer"], f["wrong"]) for f in fs) / len(fs), 2)
    answers = [f["answer"] for f in FACTS]
    res["confusion"] = round(sum(
        choose(model, f["test"], f["answer"], random.Random(i).sample([a for a in answers if a != f["answer"]], 3))
        for i, f in enumerate(FACTS)) / len(FACTS), 2)
    res["general"] = round(sum(choose(model, q, a, w) for q, a, w in GENERAL) / len(GENERAL), 2)
    res["ppl"] = fluency(model)
    return res


# ---------- 学ぶ ----------
def lessons(facts: list[dict], every_phrasing: bool = True) -> list[str]:
    out = []
    for f in facts:
        qs = f["train"] if every_phrasing else [random.choice(f["train"])]
        out += [chat(q, f["answer"]) for q in qs]
    return out


def dreams(model) -> list[str]:
    """学習前の「ふだんの自分」の受け答え。眠りのたびに再生して、新しい記憶が自分を上書きしないようにする。"""
    model.eval()
    with model.disable_adapter():
        return [chat(q, say(model, chat(q))) for q in DREAM]


def train(model, texts: list[str], epochs: int, lr: float, bs: int = 4) -> None:
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    texts = list(texts)
    for _ in range(epochs):
        random.shuffle(texts)
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            model(**enc, labels=labels).loss.backward()
            opt.step()
            opt.zero_grad()
