"""One entry point for the whole revision pipeline.

    python run_all.py            # full run; reuses forecasts / tuning that already exist
    python run_all.py --smoke    # tiny blocks, 1 seed, 2 trials, 2 epochs -> results_smoke/ (logic check only)
    python run_all.py --fresh    # move existing results aside and recompute everything
    python run_all.py --jobs 4   # parallel estimation jobs (each uses 2 threads)

Stages: main estimation (incl. COMB/SHRINK) -> timing -> main analysis (incl. roll calendar,
analyze_v3) -> robustness runs
-> robustness summary -> v1 contrast -> report. Results for the paper always come from --fresh.
Every job writes a log to <results>/logs/; progress is printed and kept in <results>/RUN_STATUS.md.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SEEDS = ["0", "7", "21", "42", "99"]
ARABICA = ROOT / "data" / "raw" / "update_2026-09" / "us_coffee_c_daily_2008-01-14_2026-09-30.csv"

# robustness runs: tag -> (environment, models)
ROBUST = {
    "lag1": ({"IJDSA_MACRO_LAG": "1"}, ["xgb", "gru"]),
    "noflat": ({"IJDSA_DROP_FLAT": "1"}, ["rf", "xgb"]),
    "roll5y": ({"IJDSA_TRAIN_WINDOW": "1260"}, ["rf", "xgb"]),
    "arabica": ({"IJDSA_COFFEE_FILE": str(ARABICA), "IJDSA_PRICE_UNIT": "cents_lb"}, ["rf", "xgb", "gru", "chronos"]),
    "arabica_noflat": ({"IJDSA_COFFEE_FILE": str(ARABICA), "IJDSA_PRICE_UNIT": "cents_lb",
                        "IJDSA_DROP_FLAT": "1", "IJDSA_DROP_FLAT_FROM": "2008-01-01"},
                       ["rf", "xgb", "gru", "chronos"]),
    "usdlag7": ({"IJDSA_USD_LAG": "7"}, ["xgb", "gru"]),
}


class Runner:
    def __init__(self, smoke, jobs):
        self.smoke, self.jobs = smoke, jobs
        self.results = ROOT / ("results_smoke" if smoke else "results")
        self.logs = self.results / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status = self.results / "RUN_STATUS.md"
        self.failed = []

    def note(self, text):
        line = f"- {datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
        print(line, flush=True)
        with open(self.status, "a") as f:
            f.write(line + "\n")

    def env(self, extra=None):
        e = dict(os.environ)
        e["PYTHONUNBUFFERED"] = "1"
        e["OMP_NUM_THREADS"] = "2"
        if self.smoke:
            e["IJDSA_SMOKE"] = "1"
        e.update(extra or {})
        return e

    def run(self, name, args, extra=None):
        log = self.logs / f"{name}.log"
        t0 = time.time()
        with open(log, "a") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(args)}\n")
            f.flush()
            r = subprocess.run([PY, *args], cwd=ROOT, env=self.env(extra), stdout=f, stderr=subprocess.STDOUT)
        ok = r.returncode == 0
        self.note(f"{'OK  ' if ok else 'FAIL'} {name} ({time.time() - t0:.0f}s){'' if ok else ' -> see logs/' + log.name}")
        if not ok:
            self.failed.append(name)
        return ok

    def parallel(self, tasks):
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            list(ex.map(lambda t: self.run(*t), tasks))

    # ------------------------------------------------------------------
    def estimation(self, tag="", extra=None, models=None):
        p = f"{tag}_" if tag else ""
        seeds = SEEDS[:1] if self.smoke else SEEDS
        if not self.run(p + "benchmarks", ["scripts/run_benchmarks.py"], extra):
            return
        if not tag:
            self.run("breaks", ["scripts/run_breaks.py"], extra)
            # tuning: once, on B1, frozen afterwards (skips if the json exists)
            self.parallel([(f"tune_{k}_{v}", ["scripts/tune.py", k, v], extra)
                           for k, v in [("svr", "full"), ("rf", "full"), ("xgb", "full"), ("dnn", "full"),
                                        ("lstm", "full"), ("gru", "full"), ("xgb", "price"), ("gru", "price")]])
            learned = [("svr", "full", ["0"])] + [(k, "full", seeds) for k in ["rf", "xgb", "dnn", "lstm", "gru"]] \
                + [(k, "price", seeds) for k in ["xgb", "gru"]]
        else:
            learned = [(k, "full", seeds) for k in (models or []) if k != "chronos"]
        tasks = [(f"{p}learned_{k}_{v}_s{s}", ["scripts/run_learned.py", k, v, s], extra)
                 for k, v, ss in learned for s in ss]
        if not tag:
            tasks += [(f"da_{k}_s{s}", ["scripts/run_decision_aware.py", k, s], extra) for k in ["xgb", "gru"] for s in seeds]
        # longest jobs first
        tasks.sort(key=lambda t: 0 if ("gru" in t[0] or "lstm" in t[0]) else 1)
        self.parallel(tasks)
        if not tag or "chronos" in (models or []):
            self.run(p + "chronos", ["scripts/run_chronos.py"], extra)
        # combination and shrinkage, after every learned job and Chronos
        self.run(p + "combos", ["scripts/run_combos.py"], extra)

    def analysis(self, tag="", extra=None):
        p = f"{tag}_" if tag else ""
        self.run(p + "analyze", ["scripts/analyze.py"], extra)
        if not tag:
            self.run("analyze_decisions", ["scripts/analyze_decisions.py"], extra)
            self.run("analyze_states", ["scripts/analyze_states.py"], extra)
        self.run(p + "analyze_v2", ["scripts/analyze_v2.py"], extra)
        if not tag:                                                  # once, for Robusta and Arabica
            self.run("roll_calendar", ["scripts/roll_calendar.py"], extra)
        self.run(p + "analyze_v3", ["scripts/analyze_v3.py"], extra)
        self.run(p + "analyze_v4", ["scripts/analyze_v4.py"], extra)
        self.run(p + "figures", ["scripts/make_figures.py"], extra)


def export(res, dest_root, R):
    """Copy the small outputs (tables, figures, reports, logs) to another folder."""
    try:
        dest = dest_root / f"results_{res.name}_{datetime.now():%Y%m%d_%H%M}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in ["RESULTS.md", "RUN_STATUS.md", "blocks.csv"]:
            if (res / f).exists():
                shutil.copy2(res / f, dest / f)
        for doc in [ROOT / "CHRONOS_CORPUS.md", ROOT / "data/raw/update_2026-09/ROLL_NOTES.md",
                    ROOT / "data/raw/fred_2026-10-16/RELEASE_CALENDAR.md"]:
            if doc.exists():
                shutil.copy2(doc, dest / doc.name)
        for sub in ["tables", "figures", "logs", "paper"]:
            if (res / sub).exists():
                shutil.copytree(res / sub, dest / sub, dirs_exist_ok=True)
        for d in sorted((res / "robust").glob("*")) if (res / "robust").exists() else []:
            for sub in ["tables", "figures"]:
                if (d / sub).exists():
                    shutil.copytree(d / sub, dest / "robust" / d.name / sub, dirs_exist_ok=True)
        R.note(f"exported to {dest}")
    except Exception as e:
        R.note(f"FAIL export: {e}")
        R.failed.append("export")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", choices=["main", "robust", "report"], default=None)
    ap.add_argument("--export", default=None, help="folder that receives tables, figures and reports at the end")
    a = ap.parse_args()

    res = ROOT / ("results_smoke" if a.smoke else "results")
    if a.fresh and res.exists():
        dest = res.with_name(res.name + f"_backup_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.move(str(res), dest)
        print("moved old results to", dest.name)
    R = Runner(a.smoke, a.jobs)
    R.note(f"START run_all smoke={a.smoke} fresh={a.fresh} jobs={a.jobs} python={sys.version.split()[0]}")

    if a.only in (None, "main"):
        if not R.run("test_leakage", ["tests/test_leakage.py"]):
            R.note("leakage test failed: stopping")
            sys.exit(1)
        if not R.run("test_variance_ratio", ["tests/test_variance_ratio.py"]):
            R.note("variance ratio test failed: stopping")
            sys.exit(1)
        R.estimation()
        # timing right after the main estimation, alone (stages run one after another)
        R.run("time_models", ["scripts/time_models.py"])
        R.analysis()
    if a.only in (None, "robust"):
        for tag, (extra, models) in ROBUST.items():
            if tag.startswith("arabica") and not ARABICA.exists():
                R.note(f"SKIP arabica: file not found ({ARABICA.relative_to(ROOT)})")
                continue
            R.note(f"robustness run '{tag}' {extra}")
            R.estimation(tag, {"IJDSA_TAG": tag, **extra}, models)
            R.analysis(tag, {"IJDSA_TAG": tag, **extra})
    if a.only in (None, "report"):
        R.run("robustness_summary", ["scripts/robustness_summary.py"])
        R.run("v1_contrast", ["scripts/v1_contrast.py"])          # main run only
        R.run("paper_tables", ["scripts/paper_tables.py"])        # main run only
        R.run("report", ["scripts/make_report.py"])
    if a.export:
        export(res, Path(a.export).expanduser(), R)
    R.note(f"END run_all; failed stages: {R.failed or 'none'}")
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
