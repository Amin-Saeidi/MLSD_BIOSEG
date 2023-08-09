FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Install MLflow
RUN pip install mlflow

# Start the MLflow server on container startup
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]