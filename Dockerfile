FROM node:20-alpine AS build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/public/ public/
COPY frontend/src/ src/
RUN npm run build

FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm
WORKDIR /app
ENV UV_CACHE_DIR=/tmp/uvcache
RUN mkdir -p ${UV_CACHE_DIR} && chmod -R 777 ${UV_CACHE_DIR}
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY evals/ evals/
COPY --from=build /frontend/build /app/frontend/build
COPY results.json datasets.json ./
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "evals/backend.py"]
