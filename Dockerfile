# ============================================
# STAGE 1: Dependencies (installs ONCE, cached)
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy requirements FIRST (only invalidates if requirements.txt changes)
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# STAGE 2: Runtime (rebuilds code in 2-3 sec)
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder stage (lightning fast, already built)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only application code (changes frequently - rebuilds instantly)
COPY .env .
COPY main.py .
CMD ["python", "main.py"]
