import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
print(client.models.generate_content(model="gemini-3.5-flash-lite", contents="Say hi in Italian").text)