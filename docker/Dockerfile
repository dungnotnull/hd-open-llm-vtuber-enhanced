# open-llm-vtuber-enhanced — runtime image
FROM python:3.12-slim

# system deps: audio I/O, ffmpeg (whisper/xtts), build tools for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libgomp1 portaudio19-dev gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# non-root runtime user
RUN useradd -m -u 1000 vtuber && chown -R vtuber:vtuber /app
USER vtuber

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    PYTHONPATH=/app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["python", "-m", "agent.main", "serve", "--host", "0.0.0.0", "--port", "8001"]
