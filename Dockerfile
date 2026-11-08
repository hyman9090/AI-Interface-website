FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

EXPOSE 7860

CMD ["streamlit", "run", "src/research_copilot/app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=7860", \
    "--server.headless=true"]
