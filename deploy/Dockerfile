FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY merchantguard ./merchantguard
RUN pip install --no-cache-dir .
# Bundle synthetic demo artifacts for a stateless showcase; NOT real production storage.
ENV MERCHANTGUARD_HOME=/app/.merchantguard
RUN python -m merchantguard demo --rows 4000 --no-shift > /app/demo-build-report.json
ENV PORT=8080
EXPOSE 8080
CMD ["sh","-c","uvicorn merchantguard.api:app --host 0.0.0.0 --port ${PORT}"]
