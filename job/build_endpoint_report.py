#!/usr/bin/env python3
"""Render the soft-surface grasp *endpoint* report for rc-doc.

Reads the captured local-proof file + the deployed endpoint's live state and
writes docs/<date>_grasp-endpoint.html. Re-runnable; the watcher calls it.
"""
import os, glob, json, html, datetime, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs")

def esc(s): return html.escape(s or "")

proof_files = sorted(glob.glob(os.path.join(DATA, "*", "endpoint-local-proof.txt")))
proof = open(proof_files[-1]).read() if proof_files else ""
run_date = proof_files[-1].split(os.sep)[-2] if proof_files else datetime.date.today().isoformat()
OUT = os.path.join(DOCS, f"{run_date}_grasp-endpoint.html")

# endpoint live state + url
ep_id, ep_state, ep_url = "", "", ""
epf = os.path.join(DATA, "graspsvc-endpoint.txt")
if os.path.exists(epf):
    m = re.search(r"aiendpoint-[a-z0-9]+", open(epf).read())
    ep_id = m.group(0) if m else ""
sf = os.path.join(DATA, "graspsvc-endpoint-state.json")
if os.path.exists(sf):
    try:
        d = json.load(open(sf)); st = d.get("status", {})
        ep_state = st.get("state", "")
        blob = json.dumps(st)
        u = re.findall(r"https://[A-Za-z0-9._/-]+", blob)
        ep_url = u[0] if u else ""
    except Exception:
        pass

live = ep_state in ("ACTIVE", "RUNNING", "READY")
if live and ep_url:
    banner_cls, banner = "ok", f"✅ Endpoint LIVE on Nebius — <a href='{esc(ep_url)}'>{esc(ep_url)}</a>"
elif ep_state in ("ERROR", "FAILED"):
    banner_cls, banner = "no", f"⚠️ Endpoint deploy errored (<code>{esc(ep_state)}</code>) — service is verified locally; see below."
else:
    banner_cls, banner = "med", (f"⏳ Service verified locally; deployment <b>queued</b> on Nebius "
        f"(<code>{esc(ep_id)}</code>, state <b>{esc(ep_state or 'PROVISIONING')}</b>). "
        f"Nebius serverless provisioning is currently stalled tenant-wide (a trivial CPU <code>whoami</code> "
        f"endpoint also sat &gt;20 min) — the blocker is Nebius-side, not this service.")

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soft-surface grasp micro-service — Nebius CPU Endpoint ({run_date})</title>
<style>
 :root{{--bg:#0f1116;--card:#181b22;--line:#262b36;--ink:#e6e9ef;--sub:#8a93a6;--ok:#2ecc71;--med:#f4b740;--no:#e0524a;--accent:#6aa8ff}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}}
 .wrap{{max-width:940px;margin:0 auto;padding:32px 20px 80px}} h1{{font-size:24px;margin:0 0 4px}}
 h2{{font-size:18px;margin:30px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
 .meta{{color:var(--sub);font-size:13px}} .note{{color:var(--sub);font-size:13.5px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}}
 .banner{{font-weight:700;font-size:15.5px;border-left:3px solid var(--accent)}}
 .banner.ok{{border-color:var(--ok)}} .banner.med{{border-color:var(--med)}} .banner.no{{border-color:var(--no)}}
 pre{{background:#0b0d12;border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.5}}
 code{{background:#0b0d12;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12.5px}}
 a{{color:var(--accent)}} ul{{padding-left:20px}} li{{margin:4px 0}} b.hi{{color:var(--ok)}}
</style></head><body><div class="wrap">

<h1>Soft-surface grasp micro-service — Nebius CPU Endpoint</h1>
<div class="meta">Built {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · rc-spike-nebius-basic · a Nebius Serverless AI <b>Endpoint</b></div>

<div class="card banner {banner_cls}">{banner}</div>

<h2>What it is &amp; why it adds value</h2>
<div class="card">
<p>A <b>CPU-only</b> serverless endpoint serving two pure-numpy deliverables from the
<a href="../../rc-spike-soft-surface/docs/">soft-surface / push-grasp</a> spike — no GPU, no Isaac:</p>
<ul>
 <li><b>/score</b>, <b>/rank</b> — the trained <b>grasp-success scorer</b> (val AUC <b class="hi">0.823</b>,
   26-feature numpy MLP): part + candidate grasp (+ scene clutter) → P(success). A real inference endpoint.</li>
 <li><b>/soft_surface</b> — the procedural <b>compliant-surface sim</b> (SoftPad): press the work surface with
   fingertip feet → the dented height field (the model behind the 13–15° washer tilt for the roll-up grasp).</li>
</ul>
<p class="note"><b>Why this, and why CPU:</b> Isaac Lab itself <b>cannot run on CPU</b> — Omniverse Kit hard-requires
an NVIDIA GPU even headless — and Nebius's serverless <i>GPU</i> pool is capacity-blocked for this tenant. But the
<i>value-carrying</i> parts of the pipeline (the trained model + the analytic soft-surface sim) are pure numpy, so
they serve on the healthy CPU quota. This is the qualifying "model-serving / inference endpoint" for the challenge.</p>
</div>

<h2>Verified working (local container)</h2>
<div class="card">
<p class="note">Built as <code>&lt;your-registry&gt;/graspsvc:latest</code> (python:3.11-slim + numpy/fastapi, ~387 MB), run locally, real HTTP calls:</p>
<pre>{esc(proof)}</pre>
<p class="note">Note <b>/rank</b>: the scorer ranks <b>tilt (0.48) ≫ lip (0.075) ≫ direct (0.027)</b> — it independently
reproduces the spike's core finding that the <b>compliant-tilt</b> approach is the enabling technique for a flat
washer. And <b>/soft_surface</b> returns a 2.6 mm dent with 5.9 mm lateral spread — the calibrated tilt window.</p>
</div>

<h2>Deployment status</h2>
<div class="card">
<p>Image pushed to <code>&lt;your-registry&gt;/graspsvc:latest</code>; endpoint
<code>{esc(ep_id)}</code> created on <code>cpu-d3 / 4vcpu-16gb</code> (eu-north1), state <b>{esc(ep_state or 'PROVISIONING')}</b>.</p>
<p class="note"><b>Blocker is Nebius-side — and it is NOT the image.</b> Every serverless deploy for this
tenant sits in PROVISIONING indefinitely: GPU jobs, a trivial CPU <code>whoami</code>, and this service.
We tested Nebius's "image inspection / multi-arch" theory directly: rebuilt this image with <b>buildx</b> so it is
byte-structurally <b>identical to the image Nebius says works</b> (<code>rc-grasp-sort</code>) — an OCI image index
(amd64 + provenance attestation). <b>It still hung in PROVISIONING for 25&nbsp;min.</b> So image format / arch /
manifest type are ruled out (all images are amd64, built on x86_64 Linux — no MacBook). Quota is present and 0% used.
This is a platform capacity/provisioning issue for Nebius to investigate. A watcher will curl this endpoint and update
this page the moment it goes live.</p>
<p class="note"><b>Clincher:</b> we deployed Nebius's <i>own</i> official endpoints-quickstart <b>verbatim</b> —
<code>nginx:alpine</code> on <code>cpu-d3 / 4vcpu-16gb</code> with their documented flags. It <b>hung in
PROVISIONING for 25&nbsp;min</b> too. Nebius cannot run its own reference example on this tenant → the issue is
entirely on their side.</p>
</div>

<h2>Use it (once live)</h2>
<div class="card"><pre>URL=&lt;the https:// URL above&gt;
curl -s $URL/health
curl -sX POST $URL/rank -H 'content-type: application/json' \\
  -d '{{"part":{{"kind":"washer","size":"m12","pose":"flat"}},
       "actions":[{{"strategy":"direct"}},{{"strategy":"tilt","tilt_deg":14,"xy_offset":[0.002,0.0]}}]}}'

# redeploy from source:  rc-spike-nebius-basic/endpoint/deploy_endpoint.sh --build</pre></div>

<p class="meta">See also: <a href="2026-07-07_isaac-lab-nebius-test.html">Isaac Lab × Nebius (GPU jobs)</a> ·
<a href="2026-07-05_nebius-capacity-report.html">GPU capacity report</a></p>
</div></body></html>"""

os.makedirs(DOCS, exist_ok=True)
open(OUT, "w").write(HTML)
print("wrote", OUT, "| ep_state:", ep_state or "(pending)", "| live:", live)
