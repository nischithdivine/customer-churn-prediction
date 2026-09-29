# start from a small official Linux image that already has Python 3.10
# (same Python version the model was trained with)
FROM python:3.10-slim

# every command below runs inside /app in the container
WORKDIR /app

# install libraries first, in their own step: Docker caches this layer,
# so changing app code later doesn't reinstall everything
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# then copy the API code and the saved model
COPY app/ app/
COPY models/ models/

# the port the API listens on inside the container
EXPOSE 8000

# start the server; 0.0.0.0 lets requests from outside the container reach it
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
