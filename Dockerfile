FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements_prod.txt requirements_prod.txt
COPY real_estate real_estate

RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements_prod.txt
ENV PATH="/venv/bin:$PATH"

# Default for local testing (GCP Cloud Run will override this)
ENV PORT=8000

EXPOSE 8000

CMD uvicorn real_estate.api.api:app --host 0.0.0.0
