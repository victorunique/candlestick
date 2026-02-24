FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

WORKDIR /app

RUN uv pip install --system streamlit redis plotly requests pandas

COPY services/ui /app

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
