import streamlit as st
import requests
import pickle
import numpy as np
import joblib

# Load your model
model = joblib.load("decision_tree_model.pkl")

# Gemini API Key
API_KEY = "AIzaSyAG6Cw-0gqeaE6OnUtC-B-7fTnoTrcDED4"

def query_gemini(prompt):

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "⚠️ Gemini API returned an unexpected response."
    else:
        return f"❌ Gemini API Error: {response.status_code} - {response.text}"

