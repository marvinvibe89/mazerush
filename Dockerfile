FROM python:3.11-slim

ENV SDL_VIDEODRIVER=dummy
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:7860", "app:app"]
