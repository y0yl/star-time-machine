FROM python:3.12-slim
WORKDIR /srv/stm
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY stm stm
COPY server server
COPY web/public/data/claims.json web/public/data/claims.json
EXPOSE 8787
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8787", "--proxy-headers"]
