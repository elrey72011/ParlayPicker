FROM python:3.12.11-slim@sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/parlaypicker

COPY services/subscriber/requirements.txt /tmp/subscriber-requirements.txt
RUN pip install --no-cache-dir -r /tmp/subscriber-requirements.txt

COPY services ./services
COPY integrations ./integrations
USER 65532:65532

EXPOSE 8080
CMD ["uvicorn", "services.subscriber.app:create_from_env", "--factory", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
