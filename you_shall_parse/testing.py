import google.generativeai as genai
import json
from tqdm import tqdm

genai.configure(api_key=API)

# Create the model
generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 65536,
  "response_mime_type": "text/plain",
}

def query_response(question,start_from_the_basics=False):
    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
    system_instruction="Some questions will have answers A,B,C,D,E or F. Reformat your answers to these questions as numbers, with A=1,B=2,C=3,D=4,E=5 and F=6. Your overall response will be some integer. Make this integer the last word in your response.",
    )

    chat_session = model.start_chat(
    history=[
    ]
    )
    if start_from_the_basics:
        response = chat_session.send_message(f"Do 1+1. Now answer {question}. Now add your answer to this question to your previous computation.")
    else:
        response = chat_session.send_message(f"Answer {question}.")
    return response.text

with open('simple_bench_public.json', 'r') as f:
    data = json.load(f)


responses={}

for question_data in tqdm(data["eval_data"],total=len(data["eval_data"])):
    standard_response=query_response(question_data["prompt"])[-50:]
    from_basics_response=query_response(question_data["prompt"],start_from_the_basics=True)[-50:]
    responses[question_data["question_id"]]={"standard":standard_response,"from_basics":from_basics_response}

with open('responses.json','w') as file:
    json.dump(responses,file,indent=4)