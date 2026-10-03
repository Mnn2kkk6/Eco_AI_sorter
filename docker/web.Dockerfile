FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY web_app ./web_app
EXPOSE 8000
CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
