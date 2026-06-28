FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend app
COPY frontend/app.py .

# Expose Streamlit port
EXPOSE 8501

# Start frontend application
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
