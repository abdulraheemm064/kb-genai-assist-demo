FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Runtime uses the standard library only; nothing to pip install.
COPY kb_assist/ kb_assist/
COPY data/ data/

RUN useradd --create-home --uid 10001 appuser
USER appuser

# The LLM provider stays disabled unless KB_ASSIST_LLM_* variables are passed at run time.
ENTRYPOINT ["python", "-m", "kb_assist"]
CMD ["audit", "--as-of", "2026-10-01"]
