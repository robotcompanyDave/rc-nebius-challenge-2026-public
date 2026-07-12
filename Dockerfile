# Slim grasp+sort sim — Nebius Serverless AI Job image.
#
# Built on the Isaac Sim runtime (ships isaacsim, omni.*, pxr, numpy in its bundled
# python at /isaac-sim/python.sh). We only layer two pure-python extras on top.
#
# NOTE: pin the exact Isaac 6.x tag the team runs. As of 2026-06-30 the isaac-sim
# repo on nvcr.io is PUBLIC — it pulls anonymously, NO `docker login nvcr.io` / NGC
# API key required (verified: manifest + fs layers for 6.0.0/4.5.0/4.2.0 all
# fetch without auth). The base is large (~15–20 GB).
FROM nvcr.io/nvidia/isaac-sim:6.0.0

ENV ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    GS_OUTPUT_DIR=/data \
    GS_HEADLESS=1

WORKDIR /workspace/grasp-sort

# extra python deps into the Isaac-bundled interpreter
COPY requirements.txt .
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r requirements.txt

COPY graspsort/ graspsort/
COPY jobs/ jobs/
COPY assets/ assets/
COPY tools/ tools/
COPY configs/ configs/

# Jobs run headless to completion and write to $GS_OUTPUT_DIR (a mounted bucket).
# Override the script per job, e.g.  --container-command "jobs/eval_sort.py".
ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["jobs/gen_data.py"]
