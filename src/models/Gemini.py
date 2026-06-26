import os
from google import genai
from google.genai import types


class Gemini:
    def __init__(self, model="gemini-2.5-flash", reasoning_effort="none") -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def inference_text(self, instruction, query, max_new_tokens=16, demos=None):
        user_content = query if not demos else "\n\n".join(demos) + "\n\n" + query
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                temperature=0.0,
                max_output_tokens=max_new_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),  # no "thinking", direct answer
            ),
        )
        return response.text


if __name__ == "__main__":
    g = Gemini(model="gemini-2.5-flash")
    instruction = "You're a english doctor, make a good choice based on the question and options. You need to answer the letter of the option without further explanations."
    query = """###Question: What is the type of skin lesion that is associated with the presence following dermoscopic concepts: {}. ###Options: A. Nevus\nB. Melanoma. ###Answer:"""
    input_query = query.format("atypical pigment network, regular streaks, regular dots and globules, regression structures")
    print(g.inference_text(instruction=instruction, query=input_query, max_new_tokens=16))
