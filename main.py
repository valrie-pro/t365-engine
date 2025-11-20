from fastapi import FastAPI

app = FastAPI(title="T365 Engine - Minimal")

@app.get("/health")
def health():
    return {"status": "ok"}
