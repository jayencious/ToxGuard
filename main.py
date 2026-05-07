from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initializing FastAPI App
app = FastAPI(
    title="Content Moderation API",
    description="An NLP-based API for detecting profanity, hate speech and toxic content.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Here, we allow all the origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the trained ML model into memory
print("Loading NLP model into memory...")

try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logistic_model.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR! Could not load .pkl files. Details: {e}")

# Defining the Request Schema
# pydantic's BaseModel ensures that our API only accepts valid JSON objects with a "text" field
class CommentRequest(BaseModel):
    text: str

# API Endpoint
@app.post("/moderate-text")
async def moderate_text(req: CommentRequest):
    # Handling empty strings to prevent the server from crashes
    if not req.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Comment text cannot be empty."
        )
    
    try:
        # Vectorization of the incoming text
        # We need to pass the text as a list because the vectorizer expects an iterable of strings
        vectorized_text = vectorizer.transform([req.text])

        # Making prediction and get the confidence scores
        pred = model.predict(vectorized_text)[0]
        probs = model.predict_proba(vectorized_text)[0]

        # probs[1] => confidence that the text is toxic (Class 1)
        toxic_conf = float(probs[1])

        # Applying the Decision Logic
        if toxic_conf >= 0.85:
            action = "delete"
        elif toxic_conf >= 0.60:
            action = "hide"
        else:
            action = "allow"
        
        # Returning the actionable JSON response
        return {
            "original_text": req.text,
            "is_toxic": bool(pred == 1),
            "confidence_score": round(toxic_conf, 4),
            "recommended_action": action
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )