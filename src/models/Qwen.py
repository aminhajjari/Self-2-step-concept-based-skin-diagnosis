import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen:
    """Qwen2.5-72B-Instruct-AWQ as a TEXT-ONLY concept classifier (pre-quantized 4-bit).
    Mirrors MMedLlama3's interface so it drops into run_x_to_c_to_y.py unchanged.
    """

    def __init__(self, ckpt="Qwen/Qwen2.5-72B-Instruct-AWQ") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=True,
        )

    def get_prompt(self, instruction, query, demos=None):
        if demos is not None:
            demonstration = "Consider the following examples:\n"
            for d in demos:
                demonstration += f"{d}\n"
            prompt = instruction + demonstration + query
        else:
            prompt = instruction + query
        return prompt

    def get_length_of_prompt(self, original_prompt: str) -> str:
        return len(original_prompt)

    def predict(self, prompt, max_new_tokens):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
