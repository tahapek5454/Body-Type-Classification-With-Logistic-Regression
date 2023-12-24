FROM python:3.9

WORKDIR /app

COPY ./data.csv /app
COPY ./api.py /app
COPY ./*.txt /app

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Uygulamayı çalıştır
CMD ["python", "api.py"]
