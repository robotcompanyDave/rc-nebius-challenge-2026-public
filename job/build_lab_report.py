#!/usr/bin/env python3
"""Render the 'Isaac Lab on Nebius' test report for rc-doc.

Reads the newest data/<date>/lab/ produced by isaac_lab_nebius_test.sh and
writes docs/<date>_isaac-lab-nebius-test.html. Safe to run before the scheduled
job has fired — it then renders a 'scheduled / pending' state.
"""
import os, glob, json, html, datetime, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs")

lab_dirs = sorted(glob.glob(os.path.join(DATA, "*", "lab")))
lab = lab_dirs[-1] if lab_dirs else None
run_date = lab.split(os.sep)[-2] if lab else datetime.date.today().isoformat()
OUT = os.path.join(DOCS, f"{run_date}_isaac-lab-nebius-test.html")

def esc(s): return html.escape(s or "")

summary = {}
runner_log = ""
attempts = []          # (name/combo, state, logfile)
train_excerpt = ""
winner = None

if lab:
    sp = os.path.join(lab, "summary.json")
    if os.path.exists(sp):
        summary = json.load(open(sp))
    rl = os.path.join(lab, "runner.log")
    if os.path.exists(rl):
        runner_log = open(rl).read()
    winner = summary.get("winner") if summary.get("winner") not in (None, "NONE", "") else None
    # gather per-job artifacts
    for jf in sorted(glob.glob(os.path.join(lab, "aijob-*.json"))):
        jid = os.path.basename(jf)[:-5]
        try:
            st = json.load(open(jf)).get("status", {}).get("state", "?")
        except Exception:
            st = "?"
        lf = jf[:-5] + ".log"
        attempts.append((jid, st, lf))
    # pick a training log excerpt from the winning / richest log
    best = None
    for jid, st, lf in attempts:
        if os.path.exists(lf):
            txt = open(lf, errors="replace").read()
            score = len(re.findall(r"(?i)iteration|reward", txt))
            if best is None or score > best[0]:
                best = (score, jid, txt)
    if best and best[0] > 0:
        txt = best[2]
        lines = [l for l in txt.splitlines()
                 if re.search(r"(?i)iteration|reward|episode|fps|Cartpole|Isaac Sim|PhysX|Loading|Simulation", l)]
        train_excerpt = "\n".join(lines[-60:]) if lines else txt[-4000:]

now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ---- state banner ----
ran_any = bool(train_excerpt)   # any attempt that actually produced training output
terminal_bad = [s for _, s, _ in attempts if s in ("ERROR", "FAILED")]
queued_job = summary.get("queued_job")
last_state = summary.get("last_state") or summary.get("terminal_state")
if winner:
    banner_cls, banner = "ok", f"✅ Isaac Lab ran on Nebius — {esc(summary.get('combo',''))} · state {esc(winner)}"
elif ran_any or terminal_bad:
    banner_cls, banner = "med", "⚠️ A job reached the node but did not complete cleanly — see the log below."
elif queued_job:
    banner_cls = "med"
    banner = (f"⏳ One H100 job is <b>queued</b> in eu-north1 (<code>{esc(queued_job)}</code>, last state "
              f"<b>{esc(last_state or 'PROVISIONING')}</b>), waiting for a free serverless GPU node. "
              f"Quota is available and 0% used — the wait is Nebius-side physical capacity. "
              f"It is billed only once it runs; this page updates automatically when a node frees.")
elif attempts:
    banner_cls, banner = "med", ("⏳ Scheduled — a peak-time probe was cancelled after it failed to get a GPU "
                                 "within 25 min (capacity jam).")
else:
    banner_cls, banner = "med", "⏳ Scheduled. This page will show the result afterward."

rows = []
for jid, st, lf in attempts:
    cls = "ok" if st == "COMPLETED" else ("no" if st in ("ERROR","FAILED","CANCELLED") else "med")
    rows.append(f'<tr><td><code>{esc(jid)}</code></td><td><span class="pill {cls}">{esc(st)}</span></td></tr>')
attempts_html = "".join(rows) if rows else '<tr><td colspan="2" class="note">No jobs submitted yet.</td></tr>'

log_block = (f'<h3>Training output (headless PhysX, GPU)</h3><pre class="logs">{esc(train_excerpt)}</pre>'
             if train_excerpt else
             '<p class="note">No training output captured yet — the scheduled run will fill this in.</p>')

runner_block = (f'<details><summary>Full runner log</summary><pre class="logs">{esc(runner_log[-6000:])}</pre></details>'
                if runner_log else "")

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Isaac Lab on Nebius Serverless AI Jobs — {run_date}</title>
<style>
 :root {{ --bg:#0f1116; --card:#181b22; --line:#262b36; --ink:#e6e9ef; --sub:#8a93a6;
   --ok:#2ecc71; --med:#f4b740; --no:#e0524a; --accent:#6aa8ff; }}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:960px;margin:0 auto;padding:32px 20px 80px}}
 h1{{font-size:25px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
 h3{{font-size:13px;margin:18px 0 8px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em}}
 .meta{{color:var(--sub);font-size:13px}} .note{{color:var(--sub);font-size:13px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}}
 .banner{{font-weight:700;font-size:16px;border-left:3px solid var(--accent)}}
 .banner.ok{{border-color:var(--ok)}} .banner.med{{border-color:var(--med)}} .banner.no{{border-color:var(--no)}}
 table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
 th{{color:var(--sub);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
 .pill{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}}
 .pill.ok{{background:rgba(46,204,113,.16);color:var(--ok)}} .pill.med{{background:rgba(244,183,64,.16);color:var(--med)}}
 .pill.no{{background:rgba(224,82,74,.16);color:var(--no)}}
 pre{{background:#0b0d12;border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.5}}
 .logs{{max-height:420px}} code{{background:#0b0d12;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12.5px}}
 a{{color:var(--accent)}} ul{{padding-left:20px}} li{{margin:3px 0}}
</style></head><body><div class="wrap">

<h1>Isaac Lab on Nebius Serverless AI Jobs</h1>
<div class="meta">Report built {now} · run date {run_date} · <a href="../../rc-spike-nebius-basic/docs/">rc-spike-nebius-basic</a></div>

<div class="card banner {banner_cls}">{banner}</div>

<h2>What this proves</h2>
<div class="card">
<p>A qualifying artifact for the <b>Nebius Serverless AI Builders Challenge</b> ("Physical AI &amp; robotics →
<i>robot simulation batch jobs</i>"): the <b>latest Isaac Lab (v3.0.0)</b> running a <b>headless, GPU-physics</b>
RL training as a <b>Nebius Serverless AI Job</b> — no rendering, pure PhysX, exactly the compute shape the
<a href="../../rc-spike-soft-surface/docs/">soft-surface</a> push-grasp work needs.</p>
<ul>
  <li><b>Workload:</b> <code>Isaac-Cartpole-v0</code>, <code>--headless --num_envs 256 --max_iterations 30</code>
      (rsl_rl). A minimal, deterministic physics-training smoke test.</li>
  <li><b>Image:</b> <code>isaac-lab-test:latest</code> = Isaac Lab v3.0.0 installed into the team's proven
      <code>nvcr.io/nvidia/isaac-sim:6.0.0</code>, pushed to the Nebius registry <code>rc-stack</code>.</li>
  <li><b>Why off-peak:</b> on 2026-07-05 both L40S and H100 on-demand failed / stalled to provision at peak.
      This runner retries several GPU/allocation combos at ~04:30 CEST.</li>
</ul>
</div>

<h2>Attempts</h2>
<div class="card"><table>
<thead><tr><th>Job</th><th>Final state</th></tr></thead>
<tbody>{attempts_html}</tbody></table>
<p class="note" style="margin-bottom:0">Combos tried in order: H100 → H200 → L40S (on-demand), then H100 / L40S preemptible — all eu-north1, where the image lives.</p>
</div>

<h2>Evidence</h2>
<div class="card">
{log_block}
{runner_block}
</div>

<h2>Capacity finding (why this is hard)</h2>
<div class="card">
<p>Serverless <b>AI Jobs</b> draw on a <b>different, smaller GPU pool</b> than regular Compute VMs — a
distinction the <code>capacity resource-advice</code> tool hides (it reports the healthy Compute-VM pool).
The pool that matters here is <code>msp.cloudapps.gpu.*</code>:</p>
<table>
<thead><tr><th>Serverless AI-Jobs GPU quota</th><th>Region</th><th>Quota</th><th>Used</th></tr></thead>
<tbody>
<tr><td>H100</td><td>eu-north1</td><td>16</td><td><span class="pill med">0% — but no free node</span></td></tr>
<tr><td>H200</td><td>eu-north1</td><td><b>0</b></td><td><span class="pill no">no quota</span></td></tr>
<tr><td>L40S</td><td>eu-north1</td><td><b>0</b></td><td><span class="pill no">no quota</span></td></tr>
<tr><td>H200</td><td>us-central1</td><td>16</td><td><span class="pill med">0% — untried</span></td></tr>
</tbody></table>
<ul>
  <li>H200/L40S have <b>zero</b> serverless quota in eu-north1 — those attempts were structurally impossible.</li>
  <li>H100 quota is present and <b>0% used</b>, yet jobs never leave PROVISIONING → <b>Nebius has no free serverless
      H100 node</b> to allocate (physical capacity, not quota, not config).</li>
  <li>Mitigation in play: one H100 job left <b>queued</b> (billed only when it runs) to grab the first freed node;
      <b>us-central1 H200</b> is the only other pool with quota.</li>
</ul>
</div>

<h2>Reproduce</h2>
<div class="card"><pre>
# on the build host (has the image + Nebius creds):
job/isaac_lab_nebius_test.sh

# inspect the winning job directly:
nebius ai job get  &lt;aijob-id&gt;
nebius ai job logs &lt;aijob-id&gt;
</pre></div>

<p class="meta">See also the <a href="2026-07-05_nebius-capacity-report.html">GPU capacity &amp; jobs report</a>.</p>
</div></body></html>"""

os.makedirs(DOCS, exist_ok=True)
open(OUT, "w").write(HTML)
print("wrote", OUT, "| winner:", winner, "| attempts:", len(attempts))
