FROM node:22-slim
WORKDIR /app
COPY package*.json ./
RUN npm install --omit=dev && npm install tsx
COPY server.ts ./
ENV NODE_ENV=production
EXPOSE 8080
CMD ["npx", "tsx", "server.ts"]
