"""
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Mistral:

    MISTRAL_LOCAL = (
        "/home/gkianfar/scratch/Amin/concept/maincode/"
        "Self-2-step-concept-based-skin-diagnosis/checkpoint/Mistral-7B-Instruct"
    )

    def __init__(self, ckpt=None, max_memory=None):
        if ckpt is None:
            ckpt = self.MISTRAL_LOCAL
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt, device_map="auto", max_memory=max_memory, torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)

    def get_prompt(self, instruction, query, demos=None):
        """Creates the prompt"""
        if demos is not None:
            prompt = f"{instruction}\n"
            prompt += "Consider the following examples:\n"
            for d in demos:
                prompt += f"{d}\n"
            prompt += f"{query}"
        else:
            prompt = f"{instruction}\n{query}"
        return prompt

    def predict(self, prompt, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(0)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id, use_cache=True
        )
        decoded = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        mistral_response = decoded[len(prompt.replace("</s>", "")):].strip()
        return mistral_response
