FROM python:3.11-slim
WORKDIR /worker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app code because the worker imports it
COPY app /worker/app

CMD ["celery", "-A", "app.tasks.celery_app", "worker", "--loglevel=INFO"]
