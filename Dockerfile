# ---- frontend build ---------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install
COPY frontend/ ./
RUN pnpm build

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    STRATA_DATA_DIR=/app/data \
    NUMBA_CACHE_DIR=/app/data/.numba_cache \
    PIP_NO_CACHE_DIR=1 \
    STRATA_FRONTEND_DIST=/app/frontend/dist
WORKDIR /app
COPY backend/pyproject.toml backend/
COPY backend/strata backend/strata
RUN pip install -e "./backend[ml]"
COPY backend/tests backend/tests
COPY examples examples
COPY --from=web /web/dist frontend/dist
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
VOLUME ["/app/data"]
EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["strata", "serve", "--host", "0.0.0.0", "--port", "8000"]
