from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import json, os
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("recipes_indexed.csv")

with open("tfidf_model.pkl", "rb") as f:
    tfidf = pickle.load(f)

ingredient_matrix = tfidf.transform(df["Ingredients"].fillna(""))

DISLIKE_FILE = "disliked_ingredients.json"
if not os.path.exists(DISLIKE_FILE):
    with open(DISLIKE_FILE,"w") as f:
        json.dump([],f)

class RequestData(BaseModel):
    ingredients: list[str]
    goal: str
    bp: str
    sugar: str
    cholesterol: str
    allergies: str
    max_time: int
    top_k: int = 5

class FeedbackData(BaseModel):
    ingredient: str
    reason: str

@app.post("/feedback")
def save_feedback(data: FeedbackData):

    with open(DISLIKE_FILE,"r") as f:
        items=json.load(f)

    items.append({
        "ingredient": data.ingredient.lower(),
        "reason": data.reason
    })

    with open(DISLIKE_FILE,"w") as f:
        json.dump(items,f)

    return {"message":"saved"}

@app.post("/recommend")
def recommend(data: RequestData):

    text=" ".join(data.ingredients)
    user_vec=tfidf.transform([text])

    scores=cosine_similarity(user_vec,ingredient_matrix).flatten()
    idx=np.argsort(scores)[::-1]

    with open(DISLIKE_FILE,"r") as f:
        disliked=json.load(f)

    disliked_items=[d["ingredient"] for d in disliked]

    results=[]

    for i in idx:

        ingredients_text=df.loc[i,"Ingredients"].lower()

        if any(d in ingredients_text for d in disliked_items):
            continue

        reason=[]

        if "high" in data.bp.lower():
            reason.append("Low sodium recipe recommended")
        if "high" in data.sugar.lower():
            reason.append("Low sugar recipe recommended")
        if "high" in data.cholesterol.lower():
            reason.append("Low fat recipe recommended")

        if not reason:
            reason.append("Matches your ingredients and health goal")

        match_percent=round(scores[i]*100,2)

        results.append({
            "dish":df.loc[i,"Dish"],
            "cuisine":df.loc[i,"Cuisine"],
            "diet":df.loc[i,"Diet"],
            "prep_time":df.loc[i,"Prep_Time"],
            "calories":float(df.loc[i,"Calories"]),
            "fat":float(df.loc[i,"Fat_g"]),
            "procedure":str(df.loc[i,"Instructions"]),
            "match_percent":match_percent,
            "why_recommended":", ".join(reason)
        })

        if len(results)>=data.top_k:
            break

    return {"recommendations":results}
