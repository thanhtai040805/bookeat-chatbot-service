# --- Dockerfile ---

FROM python:3.12-slim

# Tạo thư mục làm việc trong container
WORKDIR /app

# Copy toàn bộ mã nguồn vào container
COPY . .

# Cài thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Mở port 8000 (Render yêu cầu xác định port)
EXPOSE 8000

# Lệnh chạy FastAPI bằng Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
