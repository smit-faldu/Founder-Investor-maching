# # # match_agent.py
# # import numpy as np
# # import faiss
# # import pickle
# # from sentence_transformers import SentenceTransformer

# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.agents import initialize_agent, AgentType
# # from langchain.tools import Tool
# # import json
# # import re


# # # Load API key from environment variable
# # GOOGLE_API_KEY = 'AIzaSyCN0p7Pac2r5N4MPpRnilJEyAyFa0hrvjs'
# # # 🔹 Load embedding model
# # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # # 🔹 Load FAISS index and investor metadata
# # index = faiss.read_index("investors.index")
# # with open("investors.pkl", "rb") as f:
# #     investors = pickle.load(f)

# # # 🔹 Semantic investor search tool (no filtering, all natural language)
# # def search_faiss_similar_investors(query: str, k: int = 10):
# #     embedding = embedding_model.encode([query])[0].astype("float32")
# #     D, I = index.search(np.array([embedding]).reshape(1, -1), k)

# #     matched = [investors[i] for i in I[0]]
# #     return "\n\n".join(
# #         f"🔹 {inv['Name']}\n"
# #         f"Industry: {inv.get('Industry')}\n"
# #         f"Stage: {inv.get('Stage')}\n"
# #         f"Countries: {inv.get('Countries')}\n"
# #         f"Cheque Range: {inv.get('Cheque_range')}\n"
# #         f"Overview: {inv.get('Overview')}"
# #         for inv in matched
# #     )

# # # 🔹 LangChain Tool
# # investor_match_tool = Tool(
# #     name="InvestorMatcher",
# #     func=lambda q: search_faiss_similar_investors(q),
# #     description=(
# #         "Find relevant investors for a startup based on semantic similarity. "
# #         "Input is a natural language startup profile."
# #     )
# # )

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.0-flash-thinking-exp-01-21",
# #     temperature=0.3,
# #     google_api_key=GOOGLE_API_KEY
# # )

# # # 🔹 LangChain agent with tool access
# # agent = initialize_agent(
# #     tools=[investor_match_tool],
# #     llm=llm,
# #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# #     verbose=True,
# #     handle_parsing_errors=True  # ✅ This is what fixes the crash
# # )


# # # 🔹 Main function called by Flask
# # def run_agent_with_input(form_data: dict):
# #     query = (
# #     "You are an AI agent that matches founders with investors.\n"
# #     "Use the `InvestorMatcher` tool to find the most relevant investors for a given founder.\n"
# #     "Match based on industry, stage, geography, sector, and cheque range.\n\n"
# #     "🎯 Return only a JSON array like this:\n"
# #     "[\n"
# #     "  {\n"
# #     "    \"name\": \"Investor Name\",\n"
# #     "    \"matching_score\": 85,\n"
# #     "    \"reason\": \"Brief reason for why this investor is a match\"\n"
# #     "  },\n"
# #     "  ... more investors ...\n"
# #     "]\n\n"
# #     "Founder Profile:\n"
# #     f"- Name: {form_data.get('founder_name')}\n"
# #     f"- Company: {form_data.get('company_name')}\n"
# #     f"- Building: {form_data.get('what_building')}\n"
# #     f"- Industry: {form_data.get('industry')}\n"
# #     f"- Sectors: {form_data.get('sectors')}\n"
# #     f"- Stage: {form_data.get('product_stage')}\n"
# #     f"- Target Countries: {form_data.get('target_countries')}\n"
# #     f"- Required Funding: {form_data.get('required_funding')}\n"
# # )


# #     raw_response = agent.run(query)
# #     print("🔍 Raw agent response:\n", raw_response)


# #     # 🔥 Clean up code blocks (```json ... ```)
# #     try:
# #         # Remove code block markers if present
# #         clean_response = re.sub(r"```json|```", "", raw_response).strip()
# #         json_start = clean_response.find("[")
# #         json_data = clean_response[json_start:]
# #         matches = json.loads(json_data)
# #     except Exception as e:
# #         print("⚠️ Failed to parse JSON from agent:", e)
# #         print("Raw response was:\n", raw_response)
# #         matches = []

# #     return matches

# # match_agent.py
# import os
# import json
# import numpy as np
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool
# from dotenv import load_dotenv
# import re

# # 🔹 Load .env if using environment variables
# load_dotenv()

# # 🔹 Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # 🔹 Load FAISS index + investor metadata
# index = faiss.read_index("investors.index")
# with open("investors.pkl", "rb") as f:
#     investors = pickle.load(f)

# def search_faiss_similar_investors(query: str, k: int = 30, form_data: dict = None):
#     embedding = embedding_model.encode([query])[0].astype("float32")
#     D, I = index.search(np.array([embedding]).reshape(1, -1), k)

#     raw_matches = [investors[i] for i in I[0]]

#     # ✅ Filter matches based on metadata if form_data is passed
#     if form_data:
#         def match_filters(inv):
#             stage_match = form_data.get('product_stage', '').lower() in inv.get('Stage', '').lower()
#             country_overlap = any(
#                 c.strip().lower() in inv.get('Countries', '').lower()
#                 for c in form_data.get('target_countries', '').split(",")
#                 if c.strip()
#             )
#             return stage_match and country_overlap

#         filtered = [inv for inv in raw_matches if match_filters(inv)]
#     else:
#         filtered = raw_matches

#     return "\n\n".join(
#         f"🔹 {inv['Name']}\n"
#         f"Industry: {inv.get('Industry')}\n"
#         f"Stage: {inv.get('Stage')}\n"
#         f"Countries: {inv.get('Countries')}\n"
#         f"Cheque Range: {inv.get('Cheque_range')}\n"
#         f"Overview: {inv.get('Overview')}"
#         for inv in filtered
#     )


# # 🔹 Tool used by Gemini agent
# investor_match_tool = Tool(
#     name="InvestorMatcher",
#     func=lambda q: search_faiss_similar_investors(q),
#     description="Returns top investors semantically matching the founder profile. Input should be a detailed founder description."
# )

# # 🔹 Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0.3,
#     google_api_key='AIzaSyDAH3gZbmTJQJ_rN2EK1qHpBTUB-WdjTE8'
# )

# # 🔹 Initialize LangChain agent with error handling
# agent = initialize_agent(
#     tools=[investor_match_tool],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True
# )

# # 🔹 Entrypoint: Run agent with founder form data
# def run_agent_with_input(form_data: dict):
#     prompt = (
#         "You are an AI agent that matches founders with investors.\n"
#         "You will receive a list of investor profiles from the `InvestorMatcher` tool.\n"
#         "Only select investors that strongly match at least 3 of the following: sector/industry, stage, cheque range, and countries.\n"
#         "Prioritize unique, high-quality fits over generic ones. Avoid returning the same investor unless they are an exceptional fit.\n"
#         "Return your result as a JSON array of up to 20 investors, sorted by `matching_score` (highest to lowest), with only these fields:\n"
#         "[\n"
#         "  {\n"
#         "    \"name\": \"Investor Name\",\n"
#         "    \"matching_score\": 85,\n"
#         "    \"reason\": \"Why this investor is a match\"\n"
#         "  },\n"
#         "  ... more matches ...\n"
#         "]\n\n"
#         "Founder Profile:\n"
#         f"- Name: {form_data.get('founder_name')}\n"
#         f"- Company: {form_data.get('company_name')}\n"
#         f"- Building: {form_data.get('what_building')}\n"
#         f"- Industry: {form_data.get('industry')}\n"
#         f"- Sectors: {form_data.get('sectors')}\n"
#         f"- Stage: {form_data.get('product_stage')}\n"
#         f"- Countries: {form_data.get('target_countries')}\n"
#         f"- Required Funding: {form_data.get('required_funding')}\n"
#     )


#     raw_response = agent.run(lambda q: search_faiss_similar_investors(q, k=30, form_data=form_data))(query)


#     # 🔹 Clean markdown code block if present (```json ... ```)
#     try:
#         clean_response = re.sub(r"```json|```", "", raw_response).strip()
#         json_start = clean_response.find("[")
#         json_data = clean_response[json_start:]
#         matches = json.loads(json_data)
#     except Exception as e:
#         print("⚠️ Failed to parse JSON from agent:", e)
#         print("🔍 Raw response was:\n", raw_response)
#         matches = []
    
#     return matches


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
google_api_key = 'AIzaSyDAH3gZbmTJQJ_rN2EK1qHpBTUB-WdjTE8'
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
