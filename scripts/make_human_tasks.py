"""Build the two things that need a Kazakh speaker rather than a model.

Both are single self-contained HTML files. They open from disk, need no server,
save progress to the browser as you go, and export a JSON file at the end. The
point is that handing one to a reviewer requires no setup and no explanation of
the project.

**review_all_300.html** -- the fluency review. Every one of the 300 conversations
side by side, English against Kazakh, message by message. The automatic validator
already guarantees the structure: every message present, roles and speakers
unchanged, every URL, amount and code preserved. What it cannot judge is whether
the Kazakh reads naturally, and that is the project's longest-standing open risk.
The existing `review_batches_1_and_6.html` covers 40 of the 300.

**label_kk_sample.html** -- the human ceiling. A shuffled sample of Kazakh
conversations with no label shown, and the three class definitions to hand. Right
now the best model scores 0.539 macro-F1 on Kazakh with nothing to compare it
against: if a person scores 0.75 the model is weak, and if a person scores 0.60
the task is simply hard. The answer key is written to a separate CSV so it cannot
leak into the page.

Usage:

    python make_human_tasks.py --merged ../data/merged --out ../review --sample 100
"""
import argparse
import html
import json
import pathlib
import random

import pandas as pd

SPLITS = ["train", "test", "validation"]
SEED = 42

STYLE = """
:root {
  --bg: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --muted: #52514e;
  --line: #e4e4e0; --accent: #2a78d6; --warn: #eb6834; --ok: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1a1a19; --card: #232322; --ink: #ffffff; --muted: #c3c2b7;
    --line: #3a3a38; --accent: #3987e5; --warn: #d95926; --ok: #199e70;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px 120px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.lede { color: var(--muted); margin: 0 0 20px; max-width: 70ch; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; margin: 0 0 18px; }
.cardhead { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap;
  border-bottom: 1px solid var(--line); padding-bottom: 10px; margin-bottom: 12px; }
.idx { font-weight: 650; }
.tag { font-size: 12px; color: var(--muted); border: 1px solid var(--line);
  border-radius: 999px; padding: 2px 9px; }
.msg { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  padding: 9px 0; border-top: 1px dashed var(--line); }
.msg:first-of-type { border-top: 0; }
.msg .role { grid-column: 1 / -1; font-size: 11.5px; color: var(--muted);
  text-transform: uppercase; letter-spacing: .04em; }
.txt { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; }
.txt.kk { color: var(--ink); }
.txt.en { color: var(--muted); }
.single .msg { grid-template-columns: 1fr; }
.verdicts { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
button.v { font: inherit; font-size: 13.5px; padding: 7px 13px; cursor: pointer;
  border: 1px solid var(--line); background: transparent; color: var(--ink);
  border-radius: 7px; }
button.v[aria-pressed="true"] { border-color: var(--accent); background: var(--accent);
  color: #fff; }
textarea { width: 100%; margin-top: 9px; font: inherit; font-size: 13.5px; padding: 8px;
  border: 1px solid var(--line); border-radius: 7px; background: transparent;
  color: var(--ink); resize: vertical; min-height: 38px; }
.bar { position: fixed; left: 0; right: 0; bottom: 0; background: var(--card);
  border-top: 1px solid var(--line); padding: 11px 16px; display: flex; gap: 14px;
  align-items: center; justify-content: space-between; flex-wrap: wrap; }
.bar button { font: inherit; padding: 8px 15px; border-radius: 7px; cursor: pointer;
  border: 1px solid var(--accent); background: var(--accent); color: #fff; }
.bar .ghost { background: transparent; color: var(--ink); border-color: var(--line); }
.count { color: var(--muted); font-size: 14px; }
@media (max-width: 760px) { .msg { grid-template-columns: 1fr; } .wrap { padding: 16px 16px 140px; } }
"""

SCRIPT = """
const KEY = %(key)s;
const TOTAL = %(total)s;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
  const done = Object.keys(answers).filter(k => answers[k].verdict).length;
  document.getElementById("count").textContent = done + " of " + TOTAL + " done";
}

document.addEventListener("click", e => {
  const b = e.target.closest("button.v");
  if (!b) return;
  const id = b.dataset.id;
  answers[id] = Object.assign({}, answers[id], {verdict: b.dataset.verdict});
  document.querySelectorAll('button.v[data-id="' + CSS.escape(id) + '"]')
    .forEach(o => o.setAttribute("aria-pressed", String(o === b)));
  save();
});

document.addEventListener("input", e => {
  if (!e.target.matches("textarea[data-id]")) return;
  const id = e.target.dataset.id;
  answers[id] = Object.assign({}, answers[id], {note: e.target.value});
  save();
});

function restore() {
  Object.entries(answers).forEach(([id, a]) => {
    if (a.verdict) {
      const b = document.querySelector(
        'button.v[data-id="' + CSS.escape(id) + '"][data-verdict="' + a.verdict + '"]');
      if (b) b.setAttribute("aria-pressed", "true");
    }
    if (a.note) {
      const t = document.querySelector('textarea[data-id="' + CSS.escape(id) + '"]');
      if (t) t.value = a.note;
    }
  });
  save();
}

document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(answers, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = KEY + ".json";
  a.click();
  URL.revokeObjectURL(a.href);
});

document.getElementById("reset").addEventListener("click", () => {
  if (!confirm("Clear every answer on this page?")) return;
  answers = {};
  document.querySelectorAll('button.v').forEach(b => b.setAttribute("aria-pressed", "false"));
  document.querySelectorAll('textarea[data-id]').forEach(t => { t.value = ""; });
  save();
});

restore();
"""


def page(title, lede, cards, key, total, extra_class=""):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{STYLE}</style></head>
<body><div class="wrap {extra_class}">
<h1>{html.escape(title)}</h1>
<p class="lede">{lede}</p>
{cards}
</div>
<div class="bar">
  <span class="count" id="count"></span>
  <span>
    <button class="ghost" id="reset">Reset</button>
    <button id="export">Download answers</button>
  </span>
</div>
<script>{SCRIPT % {"key": json.dumps(key), "total": total}}</script>
</body></html>
"""


def verdict_buttons(identifier, options):
    buttons = "".join(
        f'<button class="v" data-id="{html.escape(identifier)}" data-verdict="{v}" '
        f'aria-pressed="false">{html.escape(label)}</button>'
        for v, label in options)
    return (f'<div class="verdicts">{buttons}</div>'
            f'<textarea data-id="{html.escape(identifier)}" '
            f'placeholder="Optional note"></textarea>')


def build_review(kazakh, by_id):
    cards = []
    for position, row in enumerate(kazakh.itertuples(), 1):
        english = json.loads(by_id.loc[row.translated_from, "messages"])
        translated = json.loads(row.messages)
        rows = []
        for source, target in zip(english, translated):
            rows.append(
                f'<div class="msg"><div class="role">'
                f'{html.escape(target.get("role", ""))} · '
                f'{html.escape(target.get("speaker") or "")}</div>'
                f'<div class="txt en">{html.escape(source.get("content") or "")}</div>'
                f'<div class="txt kk">{html.escape(target.get("content") or "")}</div></div>')
        cards.append(
            f'<div class="card"><div class="cardhead">'
            f'<span class="idx">#{position}</span>'
            f'<span class="tag">{html.escape(row.label_3class)}</span>'
            f'<span class="tag">{html.escape(row.style_variant)}</span>'
            f'<span class="tag">{html.escape(row.scenario_category)}</span></div>'
            + "".join(rows)
            + verdict_buttons(row.id, [("good", "Reads naturally"),
                                       ("awkward", "Understandable but awkward"),
                                       ("wrong", "Wrong or unclear")])
            + "</div>")
    lede = ("Left is the English source, right is the Kazakh. Structure is already "
            "checked automatically — every message, speaker, link, amount and code is "
            "verified to survive. <strong>Judge only whether the Kazakh reads "
            "naturally.</strong> Progress saves in this browser as you go; press "
            "<em>Download answers</em> when finished, or at any point.")
    return page("Kazakh translation review — all 300", lede, "".join(cards),
                "scambench_kk_review", len(kazakh))


def build_labelling(kazakh, sample_size, rng):
    chosen = kazakh.sample(min(sample_size, len(kazakh)), random_state=SEED)
    order = list(chosen.itertuples())
    rng.shuffle(order)

    cards = []
    for position, row in enumerate(order, 1):
        messages = json.loads(row.messages)
        rows = []
        for message in messages:
            rows.append(
                f'<div class="msg"><div class="role">'
                f'{html.escape(message.get("role", ""))} · '
                f'{html.escape(message.get("speaker") or "")}</div>'
                f'<div class="txt kk">{html.escape(message.get("content") or "")}</div></div>')
        cards.append(
            f'<div class="card"><div class="cardhead">'
            f'<span class="idx">#{position}</span></div>'
            + "".join(rows)
            + verdict_buttons(row.id, [("LEGITIMATE", "LEGITIMATE"),
                                       ("SUSPICIOUS", "SUSPICIOUS"),
                                       ("SCAM", "SCAM")])
            + "</div>")

    lede = ("Read each Kazakh conversation and choose the class. No labels are shown "
            "and the order is shuffled.<br><br>"
            "<strong>SCAM</strong> — an attack is clearly happening: the correct "
            "response is to refuse, block or escalate.<br>"
            "<strong>SUSPICIOUS</strong> — something is off and the correct response "
            "is to verify before acting, but it is not proven.<br>"
            "<strong>LEGITIMATE</strong> — ordinary interaction, no risk signal.<br><br>"
            "Mentioning money, banking or passwords is not by itself suspicious.")
    key = chosen[["id", "label_3class", "scenario_category", "style_variant"]]
    return page("Label these Kazakh conversations", lede, "".join(cards),
                "scambench_kk_labels", len(order), extra_class="single"), key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()

    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in SPLITS], ignore_index=True)
    by_id = merged.set_index("id")
    kazakh = merged[merged.language == "kk"]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    review = out_dir / "review_all_300.html"
    review.write_text(build_review(kazakh, by_id), encoding="utf-8")
    print(f"{len(kazakh)} conversations -> {review} ({review.stat().st_size // 1024} KB)")

    labelling, key = build_labelling(kazakh, args.sample, random.Random(SEED))
    task = out_dir / "label_kk_sample.html"
    task.write_text(labelling, encoding="utf-8")
    key_path = out_dir / "label_kk_answers.csv"
    key.to_csv(key_path, index=False)
    print(f"{len(key)} conversations -> {task} ({task.stat().st_size // 1024} KB)")
    print(f"answer key -> {key_path}  (kept out of the page)")
    print("\nclass balance of the labelling sample:")
    print(key.label_3class.value_counts().to_string())


if __name__ == "__main__":
    main()
