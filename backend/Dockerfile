# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# uv install karein (Fastest way to install ML libs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependency installation
COPY requirements.txt .
# --system flag use karke packages ko standard location par install karein
RUN uv pip install --system --no-cache -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.12-slim

# Environment variables carry forward karein
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. XGBoost ke liye libgomp1 zaroori hai
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Builder stage se saare installed packages copy karein
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 3. Application code copy karein
COPY . .

# 4. Security: User setup
# Ensure 'appuser' has access to the app folder
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 5. Optimized Command
# ML model ke liye 4 workers tabhi use karein agar aapke server mein 4+ Cores hain.
# Agar Cloud Run (Free/Tier) hai, toh 1 ya 2 workers safe rehte hain memory crash se bachne ke liye.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "120"]