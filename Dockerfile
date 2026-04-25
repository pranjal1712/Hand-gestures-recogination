# --- STAGE 1: Build Frontend ---
FROM node:18-slim AS build-stage
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- STAGE 2: Run Backend ---
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Copy built frontend from stage 1 to the location Flask expects
COPY --from=build-stage /app/frontend/dist ./frontend/dist

# Hugging Face uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Run Flask on port 7860
CMD ["python", "flask_app.py"]
