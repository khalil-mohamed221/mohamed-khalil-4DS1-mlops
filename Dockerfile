# 1) Base image: lightweight Python
FROM python:3.12-slim


# 2) Set working directory inside the container
WORKDIR /app

# 3) Copy dependency file first
COPY requirements.txt .

# 4) Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy the entire project to the container
COPY . .

# 6) Expose FastAPI port
EXPOSE 8000

# 7) Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
