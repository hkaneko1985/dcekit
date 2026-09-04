FROM python:3.12.13-slim

WORKDIR /opt/evidence_gated_extrapolation
COPY requirements-core.txt .
RUN python -m pip install --no-cache-dir -r requirements-core.txt
COPY . .

CMD ["python", "verify_reference_results.py"]
