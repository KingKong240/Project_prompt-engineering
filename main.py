import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# print("Current folder:", os.getcwd())
# print("Found .env at:", load_dotenv())

# 1. Configuration & Variables (with safe defaults if not set in .env)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-6-luna")
NUM_LANGUAGES =  10  #potential languages
SCENARIO = "being lost or needing urgent assistance at an international airport"

# Initialize the OpenAI client (automatically uses OPENAI_API_KEY from environment)
client = OpenAI()

# 2. Define the Data Model (Schema) for Structured JSON Parsing
class PhraseItem(BaseModel):
    language_name: str = Field(description="Name of the foreign language (e.g., Spanish, Mandarin Chinese)")
    english_phrase: str = Field(description="Full English sentence asking for help (must be more than 5 words)")
    translated_phrase: str = Field(description="The exact translation of the English phrase in the target language")
    expected_answer: str = Field(description="A likely response from a local or airport staff member in that language")
    expected_answer_meaning: str = Field(description="The English translation/meaning of that expected response")

class AirportSurvivalGuide(BaseModel):
    phrases: list[PhraseItem] = Field(description="Complete list of translated airport survival phrases")

# 3. Layer 1: Persistent Behavior Instructions (Role + Global Tone)
system_instructions = """
R (Role): Act as a patient, practical foreign language tutor teaching an English speaker.
S (Global Tone): Keep translations natural, useful, beginner-friendly, and free of academic linguistics jargon.
"""

# 4. Layer 2: Immediate Task (CRISP Prompt)
crisp_user_prompt = f"""
C (Context): I am an English-speaking traveler preparing for international trips and want to learn essential survival phrases in case of {SCENARIO}.
I (Instructions): Generate essential help phrase for each of the top {NUM_LANGUAGES} most widely spoken languages in the world (excluding English).
S (Specifications):
- Every 'english_phrase' must be a complete, polite sentence containing MORE than 5 words.
- Provide the accurate foreign translation in 'translated_phrase'.
- Provide a realistic local reply in 'expected_answer' along with its English translation in 'expected_answer_meaning'.
- Return exactly {NUM_LANGUAGES } items total per language).
P (Post-Checks): Before finalizing, verify that all {NUM_LANGUAGES} languages are included, every English help phrase exceeds 5 words, and no fields are empty or null.
"""

# 5. Execute the API Call with Structured Output Parsing
response = client.responses.parse(
    model=MODEL_NAME,
    instructions=system_instructions,
    input=crisp_user_prompt,
    text_format=AirportSurvivalGuide,
)

# 6. Access Parsed Variables Directly in Python
parsed_guide = response.output_parsed

if parsed_guide:
    for item in parsed_guide.phrases:
        print(f"[{item.language_name}]")
        print(f"  You say (EN):    {item.english_phrase}")
        print(f"  You say (Local): {item.translated_phrase}")
        print(f"  They reply:      {item.expected_answer} ({item.expected_answer_meaning})\n")