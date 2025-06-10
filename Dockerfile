FROM python:3.12-slim
LABEL authors="Sara"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "./redis-test.py"]
