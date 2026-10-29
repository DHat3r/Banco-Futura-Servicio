# Etapa 1: compilar el frontend React + Vite con Node.
FROM node:24-alpine AS build

WORKDIR /app

# Copiar primero los manifiestos para aprovechar la caché de npm ci.
COPY package*.json ./
RUN npm ci

COPY . .

# Vite incrusta VITE_API_URL en el bundle durante el build, por eso se pasa como ARG.
# Por defecto /api: nginx reenvía esas rutas al contenedor de la API (ver nginx.conf).
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# Etapa 2: servir los estáticos con nginx. La imagen final no contiene Node ni node_modules.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
