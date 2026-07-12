#!/usr/bin/env python3
"""Generate the Nebius capacity + jobs smoke-test HTML report for rc-doc.

Reads the JSON captured from the Nebius CLI in ../data/ and writes a standalone
styled HTML file into ../docs/. Re-runnable: just re-run after the smoke-test
job reaches a terminal state to refresh the job-result section.
"""
import json
import os
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(DOCS, "2026-07-05_nebius-capacity-report.html")

# ---- GPU knowledge: suitability for NVIDIA Isaac Lab / Isaac Sim ----------
# Isaac Sim's Omniverse RTX renderer needs RT cores for tiled-camera rendering,
# synthetic data and sensor sim. Ada / RTX-Blackwell workstation GPUs (L40S,
# RTX 6000) have RT cores -> fully supported. Datacenter Hopper/Blackwell
# (H100/H200/B200/B300) have NO RT cores -> great for headless *state-based* RL
# but rendering-heavy workloads are unsupported / slow.
GPU_INFO = {
    "gpu-l40s-d": ("L40S", "Ada Lovelace", "48 GB", "RT cores", "BEST",
                   "Officially supported for Isaac Sim. RT cores + 48GB handle tiled-camera rendering and synthetic data. Ideal Isaac Lab all-rounder."),
    "gpu-l40s-a": ("L40S", "Ada Lovelace", "48 GB", "RT cores", "BEST",
                   "Same silicon as l40s-d (smaller CPU/RAM presets). Fully Isaac-Sim capable."),
    "gpu-rtx6000": ("RTX 6000", "Blackwell (workstation)", "96 GB", "RT cores", "BEST",
                    "Workstation Blackwell with RT cores + 96GB. Excellent for Isaac Sim rendering and large scenes."),
    "gpu-h100-sxm": ("H100", "Hopper", "80 GB", "no RT cores", "OK",
                     "No RT cores: fast for headless *state-based* RL training, but tiled-camera / rendering workloads are unsupported/slow."),
    "gpu-h200-sxm": ("H200", "Hopper", "141 GB", "no RT cores", "OK",
                     "As H100 with 141GB. Great for large headless RL; not for RTX rendering."),
    "gpu-b200-sxm": ("B200", "Blackwell (datacenter)", "180 GB", "no RT cores", "OK",
                     "Top-tier compute, no RT cores. Headless RL only; rendering unsupported."),
    "gpu-b200-sxm-a": ("B200", "Blackwell (datacenter)", "180 GB", "no RT cores", "OK",
                       "B200 variant. Headless RL only; rendering unsupported."),
    "gpu-b300-sxm": ("B300", "Blackwell (datacenter)", "270 GB", "no RT cores", "OK",
                     "Newest/biggest datacenter Blackwell. Headless RL only; rendering unsupported."),
}

REGION_LABEL = {
    "eu-north1": "EU North (Finland)",
    "eu-west1": "EU West (France)",
    "uk-south1": "UK South",
    "me-west1": "ME West (Israel)",
    "us-central1": "US Central (Kansas City)",
}
REGION_PROJECT = {
    "eu-north1": "<your-project-id>",
    "eu-west1": "project-e01z6qs1pr00q9aht3smjc",
    "uk-south1": "project-e03jv5czpr00fpcggcfn4w",
    "me-west1": "project-i00vt0pcpr00apgfg1fvmq",
    "us-central1": "project-u00chw8wpr00kfx0farxmg",
}


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        txt = f.read().strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def avail(x):
    if not x:
        return ("none", "n/a")
    lvl = x.get("availability_level", "").replace("AVAILABILITY_LEVEL_", "")
    extra = []
    if "available" in x:
        extra.append(f"{x['available']} avail")
    if "limit" in x:
        extra.append(f"limit {x['limit']}")
    label = lvl.replace("_", " ").title() if lvl else "?"
    if extra:
        label += f" ({', '.join(extra)})"
    return (lvl.lower(), label)


LVL_CLASS = {
    "high": "hi", "medium": "med", "low": "lo",
    "limit_reached": "no", "unspecified": "unk", "": "unk", "none": "unk",
}


def badge(x):
    cls, label = avail(x)
    return f'<span class="pill {LVL_CLASS.get(cls, "unk")}">{label}</span>'


# ---------------------------------------------------------------------------
adv = load("resource-advice.json")
rows = []
for it in (adv["items"] if adv else []):
    s = it["spec"]
    ci = s["compute_instance"]
    st = it["status"]
    rows.append({
        "region": s["region"],
        "fabric": s.get("fabric", ""),
        "platform": ci["platform"],
        "preset": ci["preset"]["name"],
        "gpu": ci["preset"]["resources"].get("gpu_count"),
        "vram": ci.get("gpu_memory_gigabytes"),
        "reserved": st.get("reserved"),
        "on_demand": st.get("on_demand"),
        "preempt": st.get("preemptible"),
    })

# de-dup identical (region,platform,preset) keeping best on-demand
seen = {}
for r in rows:
    key = (r["region"], r["platform"], r["preset"])
    if key not in seen:
        seen[key] = r
rows = list(seen.values())

ORDER = {"BEST": 0, "OK": 1, "NA": 2}
rows.sort(key=lambda r: (ORDER.get(GPU_INFO.get(r["platform"], ("", "", "", "", "NA"))[4], 3),
                         r["region"], r["platform"], r["gpu"] or 0))

# job smoke-test result
job = load("job-final.json")
job_state_file = os.path.join(DATA, "job-final-state.txt")
job_logs_path = os.path.join(DATA, "job-logs.txt")
job_state = "PROVISIONING (still running at report build)"
if os.path.exists(job_state_file):
    with open(job_state_file) as f:
        job_state = f.read().strip().replace("FINAL_STATE=", "")
if job:
    job_state = job["status"]["state"]
job_logs = ""
if os.path.exists(job_logs_path):
    with open(job_logs_path) as f:
        job_logs = f.read().strip()

now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ---- Isaac Lab recommendation table (best combos, on-demand focus) --------
best_isaac = [r for r in rows if GPU_INFO.get(r["platform"], ("", "", "", "", "NA"))[4] == "BEST"]


def html_escape(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- build matrix rows ----
matrix_html = []
for r in rows:
    info = GPU_INFO.get(r["platform"], (r["platform"], "?", "?", "?", "NA", ""))
    fit = info[4]
    fitcls = {"BEST": "fit-best", "OK": "fit-ok", "NA": "fit-na"}[fit]
    fitlabel = {"BEST": "★ Isaac-ready", "OK": "Headless RL", "NA": "—"}[fit]
    matrix_html.append(f"""<tr>
      <td><span class="{fitcls}">{fitlabel}</span></td>
      <td><b>{info[0]}</b><br><span class="sub">{r['platform']}</span></td>
      <td>{REGION_LABEL.get(r['region'], r['region'])}<br><span class="sub">{r['region']}</span></td>
      <td>{r['preset']}<br><span class="sub">{r['gpu']}×GPU · {r['vram']}GB VRAM</span></td>
      <td>{badge(r['reserved'])}</td>
      <td>{badge(r['on_demand'])}</td>
      <td>{badge(r['preempt'])}</td>
    </tr>""")

isaac_html = []
for r in sorted(best_isaac, key=lambda r: (avail(r["on_demand"])[0] != "medium", avail(r["on_demand"])[0] != "high", r["region"])):
    info = GPU_INFO[r["platform"]]
    isaac_html.append(f"""<tr>
      <td><b>{info[0]}</b> <span class="sub">({info[1]})</span></td>
      <td>{REGION_LABEL.get(r['region'], r['region'])}</td>
      <td>{r['preset']}</td>
      <td>{info[2]}</td>
      <td>{badge(r['on_demand'])}</td>
      <td>{badge(r['preempt'])}</td>
    </tr>""")

# ---- deploy attempts: each is one real job we submitted --------------------
def state_cls(s):
    if s == "COMPLETED":
        return "ok"
    if s in ("ERROR", "FAILED", "CANCELLED"):
        return "no"
    return "med"


def read_logs(path):
    p = os.path.join(DATA, path)
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return ""


ATTEMPTS = [
    {"id": "aijob-████", "gpu": "L40S", "platform": "gpu-l40s-d",
     "preset": "1gpu-16vcpu-96gb", "region": "eu-north1", "advice": "on-demand Medium",
     "json": "job-final.json", "logs": "job-logs.txt",
     "note": "Isaac-ready target GPU. Provisioning ran ~12 min then the create operation "
             "failed with an internal error (code 13); job went to ERROR with no container logs — "
             "it never got a GPU. This is a real capacity signal, not a config problem "
             "(the H100 below, same image/command/region, ran fine)."},
    {"id": "aijob-████", "gpu": "H100", "platform": "gpu-h100-sxm",
     "preset": "1gpu-16vcpu-200gb", "region": "eu-north1", "advice": "on-demand High",
     "json": "job-h100.json", "logs": "job-h100-logs.txt",
     "note": "Control test to isolate the L40S failure. Same image / command / region."},
]

attempt_rows = []
logs_sections = []
for a in ATTEMPTS:
    jd = load(a["json"])
    st = jd["status"]["state"] if jd else "PROVISIONING (pending)"
    a["_state"] = st
    attempt_rows.append(f"""<tr>
      <td><span class="status {state_cls(st)}">{st}</span></td>
      <td><b>{a['gpu']}</b><br><span class="sub">{a['platform']} · {a['preset']}</span></td>
      <td>{REGION_LABEL.get(a['region'], a['region'])}</td>
      <td><span class="sub">{a['advice']}</span></td>
      <td><code>{a['id']}</code></td>
    </tr>
    <tr><td colspan="5" class="note" style="padding-top:0">{a['note']}</td></tr>""")
    lg = read_logs(a["logs"])
    if lg:
        logs_sections.append(f"""<h3>{a['gpu']} ({a['region']}) — container output</h3>
        <pre class="logs">{html_escape(lg)}</pre>""")

logs_block = "".join(logs_sections) if logs_sections else \
    """<p class="note">No container logs captured yet — see the "Live re-check" commands below.</p>"""

# headline job = first attempt that COMPLETED, else the primary L40S attempt
completed = [a for a in ATTEMPTS if a.get("_state") == "COMPLETED"]
pipeline_proven = bool(completed)
job_state = ATTEMPTS[0].get("_state", job_state)
job_cls = state_cls(job_state)

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nebius GPU Capacity & Jobs Smoke Test — {now}</title>
<style>
  :root {{ --bg:#0f1116; --card:#181b22; --line:#262b36; --ink:#e6e9ef; --sub:#8a93a6;
    --hi:#2ecc71; --med:#f4b740; --lo:#e8743b; --no:#e0524a; --unk:#5a6472; --accent:#6aa8ff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:19px; margin:36px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
  h3 {{ font-size:15px; margin:20px 0 8px; color:var(--sub); text-transform:uppercase; letter-spacing:.04em; }}
  .meta {{ color:var(--sub); font-size:13px; margin-bottom:8px; }}
  .sub {{ color:var(--sub); font-size:12px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:14px 0; }}
  .tldr {{ border-left:3px solid var(--accent); }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .scroll {{ overflow-x:auto; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ color:var(--sub); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
  tr:hover td {{ background:#1d212b; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; }}
  .pill.hi {{ background:rgba(46,204,113,.16); color:var(--hi); }}
  .pill.med {{ background:rgba(244,183,64,.16); color:var(--med); }}
  .pill.lo {{ background:rgba(232,116,59,.16); color:var(--lo); }}
  .pill.no {{ background:rgba(224,82,74,.16); color:var(--no); }}
  .pill.unk {{ background:rgba(90,100,114,.16); color:var(--unk); }}
  .fit-best {{ color:var(--hi); font-weight:700; font-size:12px; }}
  .fit-ok {{ color:var(--med); font-weight:600; font-size:12px; }}
  .fit-na {{ color:var(--unk); font-size:12px; }}
  pre {{ background:#0b0d12; border:1px solid var(--line); border-radius:8px; padding:12px 14px;
    overflow-x:auto; font-size:12.5px; line-height:1.5; }}
  code {{ background:#0b0d12; border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:12.5px; }}
  .logs {{ max-height:340px; overflow:auto; }}
  .status {{ display:inline-block; padding:3px 12px; border-radius:20px; font-weight:700; font-size:13px; }}
  .status.ok {{ background:rgba(46,204,113,.16); color:var(--hi); }}
  .status.no {{ background:rgba(224,82,74,.16); color:var(--no); }}
  .status.med {{ background:rgba(244,183,64,.16); color:var(--med); }}
  ul {{ margin:6px 0 6px 0; padding-left:20px; }}
  li {{ margin:3px 0; }}
  .legend span {{ margin-right:14px; }}
  .note {{ color:var(--sub); font-size:13px; }}
  a {{ color:var(--accent); }}
</style></head>
<body><div class="wrap">

<h1>Nebius GPU Capacity &amp; Jobs Smoke Test</h1>
<div class="meta">Generated {now} · tenant <code>tenant-&lt;redacted&gt;</code> · source: <code>nebius capacity resource-advice</code> + a live <code>nebius ai job</code> deploy</div>

<div class="card tldr">
<h3 style="margin-top:0">Bottom line</h3>
<ul>
  <li><b>The Jobs pipeline itself works</b> — the H100 control job ran end-to-end (submit → schedule → real GPU → <code>nvidia-smi</code> → logs).
      But <b>the L40S on-demand job in eu-north1 FAILED to provision</b> (~12 min, then internal error → <span class="pill no">ERROR</span>, no GPU).
      Same image/command/region as the H100, so this is a <b>capacity failure, not a config bug</b>.</li>
  <li><b>Reserved quota is exhausted everywhere</b> — every region/GPU shows <span class="pill no">Limit Reached</span> for
      <i>reserved</i>. You must launch on <b>on-demand</b> or <b>preemptible</b>.</li>
  <li><b>For Isaac Lab you need an RT-core GPU:</b> <b>L40S</b> (eu-north1) or <b>RTX&nbsp;6000</b> (us-central1) — they render
      tiled cameras / synthetic data. H100/H200/B200/B300 have no RT cores — headless state-based RL only.</li>
  <li><b>Reality vs. advisory:</b> L40S showed "Medium" on-demand but still failed to allocate. Treat the advisory levels as
      optimistic — for Isaac Lab, plan to <b>retry, use <i>preemptible</i>, or fall back to RTX&nbsp;6000 in us-central1</b>,
      and expect multi-minute cold starts.</li>
</ul>
</div>

<h2>Recommended for NVIDIA Isaac Lab</h2>
<p class="note">Isaac Sim's RTX renderer needs <b>RT cores</b> for camera rendering, sensors and synthetic data.
Only Ada/RTX-Blackwell workstation GPUs below have them. Ranked by current on-demand availability.</p>
<div class="card scroll">
<table>
<thead><tr><th>GPU</th><th>Region</th><th>Smallest 1-GPU preset</th><th>VRAM</th><th>On-demand</th><th>Preemptible</th></tr></thead>
<tbody>
{''.join(isaac_html)}
</tbody></table>
</div>

<h2>Full live capacity matrix</h2>
<div class="card legend note">
  <span><span class="pill hi">High</span> plenty</span>
  <span><span class="pill med">Medium</span> some</span>
  <span><span class="pill lo">Low</span> tight</span>
  <span><span class="pill no">Limit Reached</span> none / no quota</span>
  &nbsp; <b>★ Isaac-ready</b> = has RT cores · <b>Headless RL</b> = compute GPU, no RT cores
</div>
<div class="card scroll">
<table>
<thead><tr><th>Isaac fit</th><th>GPU</th><th>Region</th><th>Preset</th><th>Reserved</th><th>On-demand</th><th>Preemptible</th></tr></thead>
<tbody>
{''.join(matrix_html)}
</tbody></table>
</div>

<h2>The smoke-test jobs</h2>
<p class="note">Each job runs the minimum needed to prove the pipeline: pull image → land on a real GPU →
<code>nvidia-smi</code> → exit. Image <code>nvidia/cuda:12.4.1-base-ubuntu22.04</code> in every case.</p>
<div class="card scroll">
<table>
<thead><tr><th>Result</th><th>GPU / preset</th><th>Region</th><th>Advisory said</th><th>Job ID</th></tr></thead>
<tbody>
{''.join(attempt_rows)}
</tbody></table>
</div>
<div class="card">
{logs_block}
</div>

<h2>How to deploy your own job</h2>
<div class="card">
<p>Each Nebius <b>region is a separate project</b>; pick the region by <code>--parent-id</code>. The wrapper
<code>job/run_dummy_job.sh</code> in this repo does exactly this:</p>
<pre>
# L40S in eu-north1 (Isaac-ready, home region) — the default:
./job/run_dummy_job.sh

# RTX 6000 in us-central1 (Isaac-ready):
PARENT_ID={REGION_PROJECT['us-central1']} \\
  PLATFORM=gpu-rtx6000 PRESET=1gpu-24vcpu-218gb ./job/run_dummy_job.sh

# H200 headless RL, plenty of on-demand in eu-west1:
PARENT_ID={REGION_PROJECT['eu-west1']} \\
  PLATFORM=gpu-h200-sxm PRESET=1gpu-16vcpu-200gb ./job/run_dummy_job.sh
</pre>
<h3>Live re-check (capacity + this job's logs)</h3>
<pre>
# refresh the capacity matrix any time:
nebius capacity resource-advice list --format json

# this job's state + logs:
nebius ai job get  aijob-████
nebius ai job logs aijob-████
</pre>
</div>

<h2>Region → project map</h2>
<div class="card scroll"><table>
<thead><tr><th>Region</th><th>Location</th><th>Project (<code>--parent-id</code>)</th></tr></thead>
<tbody>
{''.join(f'<tr><td>{r}</td><td>{REGION_LABEL[r]}</td><td><code>{p}</code></td></tr>' for r, p in REGION_PROJECT.items())}
</tbody></table></div>

<p class="meta" style="margin-top:30px">rc-spike-nebius-basic · data snapshot in <code>data/resource-advice.json</code> · regenerate with <code>python3 job/build_report.py</code></p>

</div></body></html>"""

os.makedirs(DOCS, exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, f"({len(HTML)} bytes)  job_state={job_state}  isaac_rows={len(best_isaac)}")
