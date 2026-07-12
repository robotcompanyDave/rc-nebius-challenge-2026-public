# CPU-only serving image for the soft-surface grasp micro-service.
# Tiny (~200 MB) — python:3.11-slim + numpy + fastapi/uvicorn. No GPU, no Isaac.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MODEL_PATH=/app/model/model.npz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY graspsort/ graspsort/
COPY model/ model/
COPY app.py .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
