import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("cleaned_merged_recipe_nutrition.csv")

df["Ingredients"] = df["Ingredients"].fillna("")

tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit(df["Ingredients"])

with open("tfidf_model.pkl", "wb") as f:
    pickle.dump(tfidf, f)

df.to_csv("recipes_indexed.csv", index=False)

print("Model ready!")
