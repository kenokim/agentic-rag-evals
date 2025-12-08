# API Service

This is a FastAPI-based API service.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Run the development server:

```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000.
Documentation is available at http://127.0.0.1:8000/docs.

