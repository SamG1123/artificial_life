FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure memory_store exists for persistent data
RUN mkdir -p /app/memory_store

EXPOSE 8000

CMD ["python", "demo_server.py", "--host", "0.0.0.0", "--port", "8000"]
