FROM python:3.10-slim

WORKDIR /app

RUN sed -i.bak 's|^deb http://deb.debian.org|deb http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|^deb http://security.debian.org|deb http://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list && \
    apt update && \
    rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create temporary directory
RUN mkdir -p /tmp/yolo_uploads

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]