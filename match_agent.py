
# match_agent.py
import os
import json
import re
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index + investor metadata
index = faiss.read_index("investors.index")
with open("investors.pkl", "rb") as f:
    investors = pickle.load(f)

# Combine founder profile into one clean string
def format_founder_profile(form_data: dict) -> str:
    return (
        f"Founder: {form_data.get('founder_name')}, "
        f"Company: {form_data.get('company_name')}, "
        f"Building: {form_data.get('what_building')}, "
        f"Industry: {form_data.get('industry')}, "
        f"Sectors: {form_data.get('sectors')}, "
        f"Stage: {form_data.get('product_stage')}, "
        f"Countries: {form_data.get('target_countries')}, "
        f"Required Funding: {form_data.get('required_funding')}"
    )

# Vector-to-vector semantic similarity search
def vector_similarity_search(founder_vector: np.ndarray, k: int = 30):
    D, I = index.search(founder_vector.reshape(1, -1), k)
    matched = [investors[i] for i in I[0]]
    return "\n\n".join(
        f"🔹 {inv['Name']}\n"
        f"Industry: {inv.get('Industry')}\n"
        f"Stage: {inv.get('Stage')}\n"
        f"Countries: {inv.get('Countries')}\n"
        f"Cheque Range: {inv.get('Cheque_range')}\n"
        f"Overview: {inv.get('Overview')}"
        for inv in matched
    )

# LangChain Tool (receives founder profile as string)
investor_match_tool = Tool(
    name="InvestorMatcher",
    func=lambda q: vector_similarity_search(
        embedding_model.encode([q])[0].astype("float32"), k=30
    ),
    description=(
        "Finds top investors based on semantic similarity to a founder's profile. "
        "Input should be a string that describes the founder and what they are building."
    )
)

# Gemini Pro via LangChain
google_api_key = 'GEMINI_API_KEY'
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    google_api_key=google_api_key
)

## Initialize agent
agent = initialize_agent(
    tools=[investor_match_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Entrypoint for Flask
def run_agent_with_input(form_data: dict):
    # Step 1: Format and vectorize founder profile
    profile_text = format_founder_profile(form_data)
    founder_vector = embedding_model.encode([profile_text])[0].astype("float32")

    # Step 2: Build prompt for Gemini
    prompt = (
        "You are an AI agent that matches founders with investors.\n"
        "You will receive investor profiles from the `InvestorMatcher` tool.\n"
        "Only select investors that strongly match at least 3 of the following: industry, stage, cheque range, countries.\n"
        "Prioritize unique, high-quality fits over generic ones.\n"
        "Return your result as a JSON array of up to 20 investors, sorted by `matching_score` (highest to lowest), with only these fields:\n"
        "[\n"
        "  {\"name\": \"Investor Name\",\n"
        "   \"matching_score\": 85,\n"
        "   \"reason\": \"Why this investor is a match\"\n"
        "  },\n"
        "  ... more matches ...\n"
        "]\n"
        f"\nFounder Profile:\n{profile_text}"
    )

    # Step 3: Run agent
    raw_response = agent.run(prompt)

    # Step 4: Parse JSON response
    try:
        clean_response = re.sub(r"```json|```", "", raw_response).strip()
        json_start = clean_response.find("[")
        json_data = clean_response[json_start:]
        matches = json.loads(json_data)
        matches = sorted(matches, key=lambda x: x.get("matching_score", 0), reverse=True)
        matches = matches[:20]  # Limit to top 20
    except Exception as e:
        print("⚠️ Failed to parse JSON from agent:", e)
        print("🔍 Raw response was:\n", raw_response)
        matches = []

    return matches
