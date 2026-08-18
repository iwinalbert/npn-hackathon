# syntax=docker/dockerfile:1
#
# Retail Demand Forecasting — frontend
#
# Build:  docker build -f 07_FRONTEND/Dockerfile -t npn-frontend 07_FRONTEND
# Run:    docker run -p 8080:8080 -e API_HOST=api:8000 npn-frontend
#
# The image contains only static assets plus nginx. No API host is baked in:
# API_HOST is substituted into the nginx config at container start, and the
# JS bundle uses a same-origin /api/v1 path.

# ---------------------------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app

# Dependencies first so source edits do not reinstall node_modules.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# VITE_API_BASE_URL is deliberately NOT set: the bundle defaults to same-origin
# "/api/v1", which nginx proxies. Set it only for a split-origin deployment.
RUN npm run build

# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

# envsubst comes from gettext; used to inject API_HOST at start-up.
RUN apk add --no-cache gettext

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template

# API_HOST is the backend the SPA's /api/ requests are proxied to.
# NGINX_ENVSUBST_FILTER restricts template substitution to API_HOST alone, so
# nginx's own runtime variables ($host, $uri, ...) are never clobbered.
ENV API_HOST=api:8000     NGINX_ENVSUBST_FILTER=API_HOST

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1

# nginx:alpine renders /etc/nginx/templates/*.template through envsubst on boot.
CMD ["nginx", "-g", "daemon off;"]
