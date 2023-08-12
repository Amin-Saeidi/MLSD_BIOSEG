FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]