#!/usr/bin/env python3
"""
LLM inference evaluation for the ZIM RAG paper.

Compares answer accuracy across:
  - LLM only (no retrieval)
  - LLM + Hybrid Structured retrieval + Lead augmentation

Across any models served by Ollama (e.g. mistral:7b, biomistral).

Dataset formats supported (same as evaluate.py):
  - MedQA  JSONL : {"question":..., "options":{"A":...,...}, "answer_idx":"A"}
  - MedMCQA JSONL: {"question":..., "opa":..., "opb":..., "opc":..., "opd":..., "cop":0}
  - GUI CSV      : question, option_a, option_b, option_c, option_d, correct

Usage:
  python infer.py \\
    --dataset data/medqa_test.jsonl \\
    --index   /path/to/structured_index \\
    --host    192.168.1.50 \\
    --models  mistral:7b biomistral \\
    --n 200 --top-k 5 \\
    --out results/infer_medqa.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

# ── Dataset loaders (mirrors evaluate.py) ────────────────────────────────────

def _load_medqa_jsonl(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj  = json.loads(line)
            opts = obj.get("options", {})
            key  = (obj.get("answer_idx") or obj.get("answer", "")).strip().upper()
            rows.append({
                "question":     obj["question"],
                "options":      opts,
                "correct_key":  key,
                "correct_text": opts.get(key, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_medmcqa_jsonl(path: Path, n: int) -> list[dict]:
    key_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    opt_map  = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj  = json.loads(line)
            opts = {k: obj.get(v, "") for k, v in opt_map.items()}
            cop  = key_map.get(int(obj.get("cop", 0)), "A")
            rows.append({
                "question":     obj["question"],
                "options":      opts,
                "correct_key":  cop,
                "correct_text": opts.get(cop, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_gui_csv(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r      = {k.lower(): v for k, v in row.items()}
            letter = r.get("correct", "A").strip().upper()
            opts   = {"A": r.get("option_a",""), "B": r.get("option_b",""),
                      "C": r.get("option_c",""), "D": r.get("option_d","")}
            rows.append({
                "question":     r.get("question", ""),
                "options":      opts,
                "correct_key":  letter,
                "correct_text": opts.get(letter, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_mmlu_pro_jsonl(path: Path, n: int) -> list[dict]:
    """MMLU-Pro: options is a list of strings, answer is a letter A-J."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            opts_list = obj.get("options", [])
            opts = {chr(65 + i): v for i, v in enumerate(opts_list)}
            key = obj.get("answer", "A").strip().upper()
            rows.append({
                "question":     obj["question"],
                "options":      opts,
                "correct_key":  key,
                "correct_text": opts.get(key, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_pubmedqa_jsonl(path: Path, n: int) -> list[dict]:
    """PubMedQA: final_decision is yes/no/maybe → mapped to A/B/C."""
    _decision_map = {"yes": "A", "no": "B", "maybe": "C"}
    _opts = {"A": "yes", "B": "no", "C": "maybe"}
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            decision = obj.get("final_decision", "").lower().strip()
            key = _decision_map.get(decision, "A")
            rows.append({
                "question":     obj["question"],
                "options":      _opts,
                "correct_key":  key,
                "correct_text": _opts[key],
            })
            if len(rows) >= n:
                break
    return rows


def load_dataset(path: Path, n: int) -> list[dict]:
    if path.suffix.lower() in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        if "opa" in first:
            print("  Detected MedMCQA format")
            return _load_medmcqa_jsonl(path, n)
        if "final_decision" in first:
            print("  Detected PubMedQA format")
            return _load_pubmedqa_jsonl(path, n)
        if isinstance(first.get("options"), list):
            print("  Detected MMLU-Pro format")
            return _load_mmlu_pro_jsonl(path, n)
        print("  Detected MedQA format")
        return _load_medqa_jsonl(path, n)
    print("  Detected GUI CSV format")
    return _load_gui_csv(path, n)


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _strip_chunk_prefix(text: str) -> str:
    for prefix in ("Text: ", "Fact: "):
        idx = text.find(prefix)
        if idx != -1:
            return text[idx + len(prefix):]
    return text


def _format_context(hits: list[dict], max_ctx_chars: int | None = None) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        text    = _strip_chunk_prefix(h.get("text", "")).strip()
        lead    = _strip_chunk_prefix(h.get("lead_context", "")).strip()
        section = _strip_chunk_prefix(h.get("section_context", "")).strip()

        block = f"[{i}] {text}"
        if section:
            block += f"\n    [Section intro: {section[:300]}]"
        if lead:
            block += f"\n    [Article summary: {lead[:300]}]"
        parts.append(block)

    context_str = "\n\n".join(parts)

    if max_ctx_chars and len(context_str) > max_ctx_chars:
        # Truncate cleanly at last whitespace within budget
        truncated = context_str[:max_ctx_chars]
        last_space = truncated.rfind(" ")
        context_str = (truncated[:last_space] if last_space > 0 else truncated) + " …"

    logging.debug("── Final LLM context ──\n%s\n── end context ──", context_str)
    return context_str


def _build_prompt(question: dict, hits: list[dict] | None,
                  max_ctx_tokens: int | None = None,
                  system_prompt: str | None = None) -> tuple[str, int]:
    """Returns (prompt_str, ctx_chars) where ctx_chars is the context block length."""
    opts    = question["options"]
    opt_str = "\n".join(f"{k}) {v}" for k, v in sorted(opts.items()) if v)

    if hits:
        max_chars = max_ctx_tokens * 4 if max_ctx_tokens else None
        context   = _format_context(hits, max_ctx_chars=max_chars)
        ctx_block = f"Use the following context to help answer the question:\n\n{context}\n\n"
    else:
        ctx_block = ""

    prefix = f"{system_prompt.strip()}\n\n" if system_prompt and system_prompt.strip() else (
        "You are a medical expert. Answer the following multiple-choice question.\n"
        "Respond with ONLY the letter of the correct answer. Do not explain.\n\n"
    )

    prompt = (
        f"{prefix}"
        f"{ctx_block}"
        f"Question: {question['question']}\n\n"
        f"{opt_str}\n\n"
        "Answer:"
    )
    return prompt, len(ctx_block)


# ── Latency percentile helper ─────────────────────────────────────────────────

def _percentile(lst: list[float], p: float) -> float:
    """Linear-interpolation percentile (same as numpy default)."""
    if not lst:
        return 0.0
    s = sorted(lst)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── Pi system metrics poller ──────────────────────────────────────────────────

class _SysPoller:
    """
    Background thread that samples CPU temperature, frequency, and utilisation
    every `interval` seconds. Falls back gracefully on non-Pi hardware.
    """
    _TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
    _FREQ_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")

    def __init__(self, interval: float = 2.0) -> None:
        self._interval = interval
        self._temps: list[float] = []
        self._freqs: list[float] = []
        self._cpus:  list[float] = []
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_SysPoller":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            import psutil as _ps
            _ps.cpu_percent()  # prime — first call always returns 0.0, discard it
        except Exception:
            pass
        while not self._stop.wait(self._interval):
            t = self._read_temp()
            f = self._read_freq()
            c = self._read_cpu()
            if t is not None: self._temps.append(t)
            if f is not None: self._freqs.append(f)
            if c is not None: self._cpus.append(c)

    def _read_temp(self) -> float | None:
        try:
            return int(self._TEMP_PATH.read_text().strip()) / 1000.0
        except Exception:
            return None

    def _read_freq(self) -> float | None:
        try:
            return int(self._FREQ_PATH.read_text().strip()) / 1000.0  # kHz → MHz
        except Exception:
            return None

    def _read_cpu(self) -> float | None:
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except Exception:
            return None

    def summary(self) -> dict:
        def _avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0.0
        return {
            "peak_temp_c":  round(max(self._temps), 1) if self._temps else 0.0,
            "avg_temp_c":   _avg(self._temps),
            "min_freq_mhz": round(min(self._freqs), 0) if self._freqs else 0.0,
            "avg_freq_mhz": _avg(self._freqs),
            "avg_cpu_pct":  _avg(self._cpus),
        }


# ── Ollama call ───────────────────────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([A-J])\b")


def _parse_letter(response: str) -> str | None:
    response = response.strip()
    # Strip chain-of-thought blocks (Qwen3, DeepSeek-R1, etc.)
    # If </think> present, only look at text after it
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    else:
        response = re.sub(r"<think>.*", "", response, flags=re.DOTALL).strip()
    # First non-whitespace char if it's a letter A-J
    if response and response[0].upper() in "ABCDEFGHIJ":
        return response[0].upper()
    # Fallback: first isolated letter in response
    m = _LETTER_RE.search(response.upper())
    return m.group(1) if m else None


_EMPTY_TIMING: dict = {
    "response": None,
    "total_s": 0.0, "prefill_s": 0.0, "gen_s": 0.0, "ttft_s": 0.0,
    "prompt_tokens": 0, "gen_tokens": 0,
}


def ollama_generate(
    prompt: str,
    model: str,
    host: str,
    port: int = 11434,
    timeout: int = 120,
) -> dict:
    """
    Returns a dict with keys:
      response, total_s, prefill_s, gen_s, ttft_s, prompt_tokens, gen_tokens
    All timing values are seconds; token counts are integers.
    On failure, response is None and all numerics are 0.
    """
    url = f"http://{host}:{port}/api/generate"
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 10,
                    "stop": ["\n", ")", " -"]},
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        ns = 1_000_000_000
        prefill_s = d.get("prompt_eval_duration", 0) / ns
        gen_s     = d.get("eval_duration",        0) / ns
        load_s    = d.get("load_duration",         0) / ns
        return {
            "response":      d.get("response", ""),
            "total_s":       d.get("total_duration", 0) / ns,
            "prefill_s":     prefill_s,
            "gen_s":         gen_s,
            "ttft_s":        load_s + prefill_s,   # functional TTFT
            "prompt_tokens": d.get("prompt_eval_count", 0),
            "gen_tokens":    d.get("eval_count",        0),
        }
    except requests.exceptions.Timeout:
        return {**_EMPTY_TIMING}
    except Exception as e:
        print(f"\n  [warn] Ollama error: {e}")
        return {**_EMPTY_TIMING}


def ollama_ram_mb(host: str, port: int, model: str) -> float:
    """
    Query Ollama /api/ps for the loaded model's RAM footprint in MB.
    Returns 0.0 if the model is not loaded or the call fails.
    """
    try:
        r = requests.get(f"http://{host}:{port}/api/ps", timeout=10)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("name", "").split(":")[0] in model or model in m.get("name", ""):
                size = m.get("size", 0)
                return round(size / 1_048_576, 1)   # bytes → MB
    except Exception:
        pass
    return 0.0


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_system(
    questions:      list[dict],
    model:          str,
    host:           str,
    port:           int,
    index_dir:      Path | None,
    top_k:          int,
    cfg:            dict,
    label:          str,
    max_ctx_tokens: int | None = None,
    system_prompt:  str | None = None,
) -> list[dict]:
    results         = []
    total_ctx_chars = 0
    timing_lists    = {"total_s": [], "prefill_s": [], "gen_s": [], "ttft_s": []}
    n               = len(questions)

    # Index sizes on disk
    db_mb = idx_mb = 0.0
    if index_dir:
        _db  = index_dir / "data.db"
        _idx = index_dir / "faiss.index"
        db_mb  = round(_db.stat().st_size  / 1_048_576, 1) if _db.exists()  else 0.0
        idx_mb = round(_idx.stat().st_size / 1_048_576, 1) if _idx.exists() else 0.0

    # Warm-up: force model load before measurement so Q1 is not a cold-start outlier
    print(f"  [{label}]  warming up {model}…", end="", flush=True)
    ollama_generate("Answer with a single letter A, B, C or D.\nAnswer:",
                    model, host, port, timeout=180)
    ram_mb = ollama_ram_mb(host, port, model)
    print(f" ready  RAM={ram_mb:.0f}MB")

    poller = _SysPoller(interval=2.0).start()

    t0 = time.time()
    for i, q in enumerate(questions):
        elapsed = time.time() - t0
        rate    = (i + 1) / max(elapsed, 0.1)
        eta     = (n - i - 1) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  [{label}]  {i+1}/{n}  "
            f"{rate:.1f} q/s  ETA {int(eta//60)}m{int(eta%60):02d}s   "
        )
        sys.stdout.flush()

        hits = []
        if index_dir is not None:
            try:
                from indexer.query import search
                hits = search(index_dir, q["question"], top_k=top_k, cfg=cfg)
            except Exception as e:
                print(f"\n  [warn] retrieval q{i+1}: {e}")

        prompt, ctx_chars = _build_prompt(q, hits if hits else None,
                                          max_ctx_tokens=max_ctx_tokens,
                                          system_prompt=system_prompt)
        total_ctx_chars += ctx_chars

        gen = ollama_generate(prompt, model, host, port)

        if gen["total_s"] > 0:
            for k in timing_lists:
                timing_lists[k].append(gen[k])

        predicted = _parse_letter(gen["response"] or "") if gen["response"] else None
        correct   = predicted == q["correct_key"] if predicted else False

        results.append({
            "idx":           i,
            "question":      q["question"],
            "correct_key":   q["correct_key"],
            "correct_text":  q["correct_text"],
            "predicted":     predicted or "",
            "correct":       correct,
            "raw_response":  (gen["response"] or "").strip()[:80],
            "total_s":       round(gen["total_s"],   3),
            "prefill_s":     round(gen["prefill_s"], 3),
            "gen_s":         round(gen["gen_s"],     3),
            "ttft_s":        round(gen["ttft_s"],    3),
            "prompt_tokens": gen["prompt_tokens"],
            "gen_tokens":    gen["gen_tokens"],
        })

    poller.stop()
    sys_stats = poller.summary()

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0
    def _std(lst): return statistics.stdev(lst) if len(lst) >= 2 else 0.0

    n_res          = max(len(results), 1)
    acc            = sum(r["correct"] for r in results) / n_res
    avg_ctx_tokens = int(total_ctx_chars / n_res / 4)
    avg_prefill    = _avg(timing_lists["prefill_s"])
    avg_gen        = _avg(timing_lists["gen_s"])
    avg_total      = _avg(timing_lists["total_s"])
    avg_ttft       = _avg(timing_lists["ttft_s"])
    std_prefill    = _std(timing_lists["prefill_s"])
    std_gen        = _std(timing_lists["gen_s"])
    std_total      = _std(timing_lists["total_s"])
    std_ttft       = _std(timing_lists["ttft_s"])
    p50_total      = _percentile(timing_lists["total_s"], 50)
    p95_total      = _percentile(timing_lists["total_s"], 95)
    p50_prefill    = _percentile(timing_lists["prefill_s"], 50)
    p95_prefill    = _percentile(timing_lists["prefill_s"], 95)

    ctx_label = f"{max_ctx_tokens}tok" if max_ctx_tokens else "unlimited"
    temp_str  = (f"  temp={sys_stats['avg_temp_c']:.1f}°C"
                 f"(peak {sys_stats['peak_temp_c']:.1f}°C)"
                 if sys_stats["peak_temp_c"] else "")
    print(
        f"\r  [{label}]  {n}/{n} done  acc={acc:.3f}  "
        f"ctx≈{avg_ctx_tokens}tok({ctx_label})  "
        f"prefill={avg_prefill:.2f}±{std_prefill:.2f}s  "
        f"gen={avg_gen:.2f}±{std_gen:.2f}s  "
        f"p50/p95={p50_total:.2f}/{p95_total:.2f}s  "
        f"RAM={ram_mb:.0f}MB{temp_str}      "
    )

    summary = {
        "avg_ctx_tokens": avg_ctx_tokens,
        "avg_total_s":    round(avg_total,   3),
        "avg_prefill_s":  round(avg_prefill, 3),
        "avg_gen_s":      round(avg_gen,     3),
        "avg_ttft_s":     round(avg_ttft,    3),
        "std_total_s":    round(std_total,   3),
        "std_prefill_s":  round(std_prefill, 3),
        "std_gen_s":      round(std_gen,     3),
        "std_ttft_s":     round(std_ttft,    3),
        "p50_total_s":    round(p50_total,   3),
        "p95_total_s":    round(p95_total,   3),
        "p50_prefill_s":  round(p50_prefill, 3),
        "p95_prefill_s":  round(p95_prefill, 3),
        "ram_mb":         ram_mb,
        "db_mb":          db_mb,
        "idx_mb":         idx_mb,
        "ctx_budget":     max_ctx_tokens or 0,
        **sys_stats,
    }
    for r in results:
        r.update(summary)
    return results


# ── Metrics + output ──────────────────────────────────────────────────────────

def accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def print_table(systems: list[dict]) -> None:
    hdr = (f"{'System':<44} {'Acc':>6} {'CtxTok':>7} "
           f"{'Prefill±σ':>12} {'Gen±σ':>10} {'p50/p95':>11} {'RAM MB':>7} {'Temp°C':>8}")
    print("\n" + "─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))
    for s in systems:
        tok_str  = f"~{s['avg_ctx_tokens']}" if s.get("avg_ctx_tokens") else "—"
        ram_str  = f"{s['ram_mb']:.0f}"      if s.get("ram_mb")         else "—"
        pre_str  = f"{s['avg_prefill_s']:.2f}±{s['std_prefill_s']:.2f}s"
        gen_str  = f"{s['avg_gen_s']:.2f}±{s['std_gen_s']:.2f}s"
        pct_str  = f"{s['p50_total_s']:.2f}/{s['p95_total_s']:.2f}s"
        temp_str = f"{s['avg_temp_c']:.1f}" if s.get("avg_temp_c") else "—"
        print(f"{s['label']:<44} {s['acc']:>6.3f} {tok_str:>7} "
              f"{pre_str:>12} {gen_str:>10} {pct_str:>11} {ram_str:>7} {temp_str:>8}")
    print("─" * len(hdr))


def export_csv(
    out_path:    Path,
    systems:     list[dict],
    all_results: list[tuple[str, list[dict]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics
    agg = out_path.with_stem(out_path.stem + "_metrics")
    _AGG_FIELDS = [
        "label", "acc", "n_correct", "n_total",
        "avg_ctx_tokens",
        "avg_prefill_s", "std_prefill_s", "p50_prefill_s", "p95_prefill_s",
        "avg_gen_s",     "std_gen_s",
        "avg_total_s",   "std_total_s",   "p50_total_s",   "p95_total_s",
        "avg_ttft_s",    "std_ttft_s",
        "ram_mb", "db_mb", "idx_mb",
        "avg_temp_c", "peak_temp_c", "avg_cpu_pct", "avg_freq_mhz", "min_freq_mhz",
        "ctx_budget",
    ]
    with open(agg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_FIELDS, extrasaction="ignore")
        w.writeheader()
        for s in systems:
            row = {k: (f"{s[k]:.4f}" if isinstance(s.get(k), float) else s.get(k, ""))
                   for k in _AGG_FIELDS}
            row["label"]     = s["label"]
            row["n_correct"] = s["n_correct"]
            row["n_total"]   = s["n_total"]
            w.writerow(row)
    print(f"  Metrics  → {agg}")

    # Per-question
    if not all_results:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = ["num", "question", "correct_key", "correct_text"] + \
                 [f"{lbl}_pred" for lbl, _ in all_results] + \
                 [f"{lbl}_ok"   for lbl, _ in all_results]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n = len(all_results[0][1])
        for i in range(n):
            row: dict = {
                "num":          i + 1,
                "question":     all_results[0][1][i]["question"][:100],
                "correct_key":  all_results[0][1][i]["correct_key"],
                "correct_text": all_results[0][1][i].get("correct_text", "")[:80],
            }
            for lbl, res in all_results:
                row[f"{lbl}_pred"] = res[i]["predicted"]
                row[f"{lbl}_ok"]   = int(res[i]["correct"])
            w.writerow(row)
    print(f"  Per-question → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM inference evaluation for ZIM RAG")
    parser.add_argument("--dataset",  required=True,            help="Dataset path (JSONL or CSV)")
    parser.add_argument("--index",    default=None,             help="Structured index directory")
    parser.add_argument("--host",     default="localhost",      help="Ollama host IP (default localhost)")
    parser.add_argument("--port",     type=int, default=11434,  help="Ollama port (default 11434)")
    parser.add_argument("--models",   nargs="+",
                        default=["mistral:7b"],                 help="Ollama model names")
    parser.add_argument("--n",        type=int, default=200,    help="Max questions (default 200)")
    parser.add_argument("--top-k",    type=int, default=5,      help="Retrieval top-K (default 5)")
    parser.add_argument("--out",        default="results/infer.csv", help="Output CSV path")
    parser.add_argument("--flat-index",  default=None,               help="Flat-chunked index dir (for flat RAG baseline)")
    parser.add_argument("--ctx-budgets", nargs="+", type=int,
                        default=[128, 512],
                        help="Context token budgets to test (0 = unlimited). Default: 128 512")
    parser.add_argument("--verbose",       action="store_true",        help="Log final LLM context to stderr")
    parser.add_argument("--system-prompt", default=None,               help="System prompt text (default: built-in medical expert prompt)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    index_dir      = Path(args.index)      if args.index      else None
    flat_index_dir = Path(args.flat_index) if args.flat_index else None
    out_path       = Path(args.out)
    top_k          = args.top_k

    print(f"\nDataset    : {args.dataset}")
    print(f"Host       : {args.host}:{args.port}")
    print(f"Models     : {args.models}")
    print(f"N          : {args.n}   Top-K: {top_k}")
    print(f"Struct idx : {index_dir or '(none)'}")
    print(f"Flat idx   : {flat_index_dir or '(none — flat RAG skipped)'}")

    print("\nLoading dataset...")
    questions = load_dataset(Path(args.dataset), args.n)
    print(f"  {len(questions)} questions loaded")

    # ── Retrieval configs for each ablation condition ─────────────────────────
    _base = {
        "use_faiss": True, "use_title_bm25": True, "use_para_bm25": True,
        "use_diversity_cap": False,   # hand-tuned, disabled for reproducibility
        "use_mention_penalty": False,  # hand-tuned, disabled for reproducibility
        "use_nav_boost": False,        # hand-tuned, disabled for reproducibility
        "eval_rrf_k": 60, "eval_diversity_max": 6,
    }
    CONDITIONS: list[tuple[str, Path | None, dict]] = [
        ("No Retrieval",      None,           {}),
        ("Flat RAG",          flat_index_dir, {**_base, "use_lead_augment": False, "use_section_augment": False}),
        ("Struct RAG",        index_dir,      {**_base, "use_lead_augment": False, "use_section_augment": False}),
        ("Struct+Lead",       index_dir,      {**_base, "use_lead_augment": True,  "use_section_augment": False}),
        ("Struct+Section",    index_dir,      {**_base, "use_lead_augment": False, "use_section_augment": True}),
        ("Struct+Lead+Sect",  index_dir,      {**_base, "use_lead_augment": True,  "use_section_augment": True}),
    ]

    # 0 in ctx_budgets means unlimited; deduplicate and sort
    ctx_budgets = sorted(set(args.ctx_budgets))

    all_results: list[tuple[str, list[dict]]] = []
    agg_systems: list[dict]                    = []

    for model in args.models:
        short = model.replace(":", "-")
        for cond_name, idx_dir, cfg in CONDITIONS:
            if idx_dir is not None and not idx_dir.exists():
                print(f"\n  [skip] {cond_name} — index not found at {idx_dir}")
                continue
            for budget in ctx_budgets:
                max_ctx = budget if budget > 0 else None
                bud_label = f"{budget}tok" if budget > 0 else "unlimited"
                # No-retrieval: context is always empty, budget doesn't change anything
                # Run it once (budget loop still applies for uniform table structure)
                label = f"{short} / {cond_name} / {bud_label}"
                print(f"\nRunning: {label}")
                res = run_system(questions, model, args.host, args.port,
                                 idx_dir, top_k, cfg, label,
                                 max_ctx_tokens=max_ctx,
                                 system_prompt=args.system_prompt)
                acc = accuracy(res)
                s   = res[0] if res else {}
                agg_systems.append({
                    "label":     label,
                    "acc":       acc,
                    "n_correct": sum(r["correct"] for r in res),
                    "n_total":   len(res),
                    **{k: s.get(k, 0) for k in [
                        "avg_ctx_tokens",
                        "avg_prefill_s", "std_prefill_s", "p50_prefill_s", "p95_prefill_s",
                        "avg_gen_s",     "std_gen_s",
                        "avg_total_s",   "std_total_s",   "p50_total_s",   "p95_total_s",
                        "avg_ttft_s",    "std_ttft_s",
                        "ram_mb", "db_mb", "idx_mb",
                        "avg_temp_c", "peak_temp_c", "avg_cpu_pct",
                        "avg_freq_mhz", "min_freq_mhz", "ctx_budget",
                    ]},
                })
                all_results.append((label, res))

    print_table(agg_systems)
    export_csv(out_path, agg_systems, all_results)


if __name__ == "__main__":
    main()
