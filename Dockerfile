FROM node:20-alpine AS build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/public/ public/
COPY frontend/src/ src/
RUN npm run build

FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_CACHE_DIR=/home/user/.cache/uv
RUN mkdir -p ${UV_CACHE_DIR} && chown -R user:user ${HOME}
USER user
WORKDIR $HOME/app
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev
COPY --chown=user evals/ evals/
COPY --chown=user --from=build /frontend/build /home/user/app/frontend/build
COPY --chown=user results.json datasets.json models.json languages.json ./
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "evals/backend.py"]
