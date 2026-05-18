# mistral_refiner.py
import sys
sys.path.append('.')

from src.models.Mistral import Mistral
from typing import Dict
import re


class MistralBasedRefiner:
    """
    Mistral-7B-Instruct based refiner for ExpLICD Self-Refine.
    Drop-in replacement for MMedBasedRefiner.
    """

    def __init__(self, ckpt="/home/gkianfar/scratch/Amin/concept/maincode/"
                           "Self-2-step-concept-based-skin-diagnosis/checkpoint/Mistral-7B"):
        print(f"Loading Mistral refiner from: {ckpt}")
        self.model = Mistral()   # Mistral loads from HF cache or local
        print("✓ Mistral refiner loaded")

    def __call__(self, concepts_str: str, feedback: str,
                 concepts_dict: Dict[str, str]) -> str:

        instruction = self._build_instruction()
        query = self._build_query(concepts_str, feedback)

        prompt = self.model.get_prompt(
            instruction=instruction,
            query=query,
            demos=None
        )

        try:
            # Mistral follows instructions well at low token budgets
            raw = self.model.predict(prompt=prompt, max_new_tokens=120).strip()
            cleaned = self._clean_output(raw)

            if self._validate_format(cleaned):
                print("✓ Mistral refinement successful")
                return cleaned
            else:
                print("⚠ Mistral format failed, using rule-based fallback")
                from src.self_refiner.concept_refiner import SimpleRuleBasedRefiner
                return SimpleRuleBasedRefiner()(concepts_str, feedback, concepts_dict)

        except Exception as e:
            print(f"⚠ Mistral error: {e}, using rule-based fallback")
            from src.self_refiner.concept_refiner import SimpleRuleBasedRefiner
            return SimpleRuleBasedRefiner()(concepts_str, feedback, concepts_dict)

    def _build_instruction(self) -> str:
        return (
            "You are a dermatology expert. Fix ONLY the clinically inconsistent "
            "concepts listed below. Output ONLY the corrected full description "
            "in this exact format: "
            "The color is ..., the shape is ..., the border is ..., "
            "the dermoscopic patterns are ..., the texture is ..., "
            "the symmetry is ..., the elevation is ..."
        )

    def _build_query(self, concepts_str: str, feedback: str) -> str:
        return (
            f"CURRENT DESCRIPTION:\n{concepts_str}\n\n"
            f"VIOLATIONS TO FIX:\n{feedback}\n\n"
            f"OUTPUT (corrected description only, no explanation):\n"
            f"The color is"
        )

    def _clean_output(self, output: str) -> str:
        # Mistral may echo the prompt — find the concept sentence
        match = re.search(
            r'The color is .+?elevation is [^.]+\.', output,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(0)
        # If the prompt seed "The color is" was stripped, re-attach
        if output.lower().startswith('the color is'):
            return output.split('\n')[0].strip()
        return output.strip()

    def _validate_format(self, s: str) -> bool:
        keywords = ['color', 'shape', 'border', 'pattern',
                    'texture', 'symmetry', 'elevation']
        return sum(1 for k in keywords if k in s.lower()) >= 5
