"""
MMed-LLM Based Refiner for ExpLICD Self-Refine (Path D)
Uses MMed-Llama-3-8B instead of GPT-4o (FREE - runs on cluster)

ENHANCED VERSION with:
- Improved prompts with clinical examples
- Better output cleaning
- Concept extraction/salvaging
- Targeted refinement (only fix violated concepts)
- Automatic fallback to rule-based when LLM fails
"""

import sys
sys.path.append('.')

from src.models.MMed_Llama_3_8B import MMedLlama3
from typing import Dict
import re


class MMedBasedRefiner:
    """
    MMed-LLM based refiner that uses MMed-Llama-3-8B to refine concept predictions.
    
    This is the FREE alternative to GPT-4o that runs on your cluster.
    """
    
    def __init__(self, ckpt="Henrychur/MMed-Llama-3-8B"):
        """
        Args:
            ckpt: MMed checkpoint to use
        """
        print(f"Loading MMed-LLM refiner: {ckpt}")
        self.model = MMedLlama3(ckpt=ckpt)
        print("✓ MMed-LLM refiner loaded")
    
    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        """
        Refine concepts using MMed-LLM based on feedback.
        
        Args:
            concepts_str: Current concept description string
            feedback: Consistency violation feedback
            concepts_dict: Parsed concepts dictionary
        
        Returns:
            Refined concept description string
        """
        
        # Extract violated concepts from feedback for targeted refinement
        violated_concepts = self._extract_violated_concepts(feedback)
        
        # Build prompts with improved clinical guidance
        instruction = self._build_instruction(violated_concepts)
        query = self._build_query(concepts_str, feedback, violated_concepts)
        
        try:
            # Get prompt from MMed
            prompt = self.model.get_prompt(instruction=instruction, query=query, demos=None)
            
            # Generate refinement (increased from 300 to 500 tokens)
            refined_concepts = self.model.predict(prompt=prompt, max_new_tokens=500).strip()
            
            # Clean up output (remove any extra text)
            refined_concepts = self._clean_output(refined_concepts)
            
            # Validate format
            if self._validate_format(refined_concepts):
                print(f"✓ LLM refinement successful")
                return refined_concepts
            else:
                # Try to extract concepts if format is partially valid
                print(f"⚠ Format validation failed, attempting extraction...")
                extracted = self._try_extract_concepts(refined_concepts, concepts_dict)
                
                if extracted and self._validate_format(extracted):
                    print(f"✓ Extraction successful")
                    return extracted
                else:
                    # Fall back to rule-based refinement
                    print(f"⚠ Extraction failed, using rule-based fallback")
                    from src.self_refiner.concept_refiner import SimpleRuleBasedRefiner
                    fallback = SimpleRuleBasedRefiner()
                    return fallback(concepts_str, feedback, concepts_dict)
            
        except Exception as e:
            print(f"⚠ LLM refinement error: {e}, using rule-based fallback")
            from src.self_refiner.concept_refiner import SimpleRuleBasedRefiner
            fallback = SimpleRuleBasedRefiner()
            return fallback(concepts_str, feedback, concepts_dict)
    
    def _extract_violated_concepts(self, feedback: str) -> set:
        """
        Extract which concepts have violations from feedback.
        
        Args:
            feedback: Violation feedback string
        
        Returns:
            Set of violated concept names (e.g., {'border', 'symmetry'})
        """
        violated = set()
        feedback_lower = feedback.lower()
        
        # Map feedback keywords to concept names
        concept_keywords = {
            'border': ['border'],
            'color': ['color', 'multiple colors'],
            'shape': ['shape'],
            'symmetry': ['symmetry', 'asymmetric', 'symmetrical'],
            'texture': ['texture', 'smooth', 'ulcerated'],
            'elevation': ['elevation', 'flat', 'raised'],
            'dermoscopic patterns': ['pattern', 'network', 'veil', 'streak']
        }
        
        for concept, keywords in concept_keywords.items():
            if any(kw in feedback_lower for kw in keywords):
                violated.add(concept)
        
        return violated
    
    def _build_instruction(self, violated_concepts: set) -> str:
        """Build targeted instruction with clinical examples."""
        
        base_instruction = """You are a dermatology expert. Fix clinical inconsistencies in dermoscopic descriptions.

CRITICAL RULES:
1. Output ONLY the complete refined description (no preambles, no explanations)
2. Use this EXACT format: "The color is ..., the shape is ..., the border is ..., the dermoscopic patterns are ..., the texture is ..., the symmetry is ..., the elevation is ..."
3. ONLY modify concepts that are clinically inconsistent
4. Keep all other concepts EXACTLY as they are

CLINICAL CONSISTENCY RULES:"""
        
        # Add specific rules for violated concepts
        if 'border' in violated_concepts or 'symmetry' in violated_concepts:
            base_instruction += """
- Asymmetric lesions → Use "often blurry and irregular" border
- Symmetric lesions → Use "sharp and well-defined" border"""
        
        if 'color' in violated_concepts or 'dermoscopic patterns' in violated_concepts:
            base_instruction += """
- Multiple colors → Use "atypical pigment network, irregular streaks" patterns
- Single color → Use "regular pigment network, symmetric dots" patterns"""
        
        if 'texture' in violated_concepts or 'elevation' in violated_concepts:
            base_instruction += """
- Smooth texture → Must be "flat to slightly raised" elevation
- Raised/ulcerated → Cannot have "smooth" texture"""
        
        base_instruction += """

EXAMPLES OF GOOD FIXES:
❌ BAD: "The color is multiple colors, border is sharp, symmetry is asymmetric"
✅ GOOD: "The color is multiple colors, border is often blurry and irregular, symmetry is asymmetric"

❌ BAD: "The texture is smooth, elevation is raised with possible ulceration"
✅ GOOD: "The texture is smooth, elevation is flat to slightly raised"
"""
        
        return base_instruction
    
    def _build_query(self, concepts_str: str, feedback: str, violated_concepts: set) -> str:
        """Build query with clear task description."""
        
        # Make violations more prominent
        violation_list = "\n".join([f"• {v}" for v in feedback.split('\n') if v])
        
        query = f"""CURRENT DESCRIPTION:
{concepts_str}

VIOLATIONS TO FIX:
{violation_list}

TASK: Rewrite the COMPLETE description fixing ONLY the concepts mentioned above.

OUTPUT FORMAT (copy this structure exactly):
The color is [KEEP OR FIX], the shape is [KEEP OR FIX], the border is [KEEP OR FIX], the dermoscopic patterns are [KEEP OR FIX], the texture is [KEEP OR FIX], the symmetry is [KEEP OR FIX], the elevation is [KEEP OR FIX].

YOUR REFINED DESCRIPTION:"""
        
        return query
    
    def _clean_output(self, output: str) -> str:
        """
        Clean MMed output to extract just the concept description.
        
        Removes common LLM artifacts like:
        - "Refined description:"
        - "Here is the refined..."
        - Multiple newlines
        - Extra explanations
        - Bullet points/lists
        """
        # Remove common preambles
        output = re.sub(r'^(refined description|here is|the refined|your refined description)[\s:]*', '', output, flags=re.IGNORECASE)
        
        # Remove bullet points/lists that LLMs sometimes generate
        output = re.sub(r'^\s*[-•*]\s*', '', output, flags=re.MULTILINE)
        
        # Extract first complete sentence starting with "The color"
        match = re.search(r'The color is .+?elevation is [^.]+\.', output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0)
        
        # Split by newlines and find the concept sentence
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for line that starts with "The color is"
            if line.lower().startswith('the color is'):
                return line
        
        # If not found, look for any line with multiple concept keywords
        for line in lines:
            line = line.strip()
            if len(line) > 50:  # Reasonable length for full description
                keyword_count = sum(1 for kw in ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation'] 
                                   if kw in line.lower())
                if keyword_count >= 4:
                    return line
        
        # Last resort: return first substantial line
        for line in lines:
            if len(line.strip()) > 50:
                return line.strip()
        
        return output.strip()
    
    def _validate_format(self, refined_str: str) -> bool:
        """
        Lenient format validation - checks if 5 out of 7 concept keywords are present.
        
        Args:
            refined_str: Refined concept string to validate
        
        Returns:
            True if at least 5/7 concept keywords found
        """
        concept_keywords = ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation']
        found_count = sum(1 for keyword in concept_keywords if keyword in refined_str.lower())
        return found_count >= 5  # ✅ 5/7 is good enough
    
    def _try_extract_concepts(self, output: str, original_dict: Dict[str, str]) -> str:
        """
        Try to salvage partially valid output by extracting concepts.
        
        If MMed output is partially valid (e.g., some concepts present but not all),
        try to extract what we can and fill in missing concepts from original.
        
        Args:
            output: MMed output that failed validation
            original_dict: Original concepts dictionary as fallback
        
        Returns:
            Extracted/salvaged concept string, or empty string if failed
        """
        extracted_dict = {}
        
        # Define expected concept keys
        concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 
                       'texture', 'symmetry', 'elevation']
        
        # Try to extract each concept
        for key in concept_keys:
            # Try multiple patterns
            patterns = [
                rf"(?:the\s+)?{re.escape(key)}\s+(?:is|are)\s+([^,\.]+)",  # "the color is ..."
                rf"{re.escape(key)}:\s*([^,\.]+)",  # "color: ..."
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    extracted_dict[key] = match.group(1).strip()
                    found = True
                    break
            
            # If not found, use original
            if not found and key in original_dict:
                extracted_dict[key] = original_dict[key]
        
        # Check if we extracted enough concepts
        if len(extracted_dict) >= 5:
            # Convert to string
            template = (
                "The color is {color}, the shape is {shape}, the border is {border}, "
                "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
                "the symmetry is {symmetry}, the elevation is {elevation}."
            )
            try:
                return template.format(**extracted_dict)
            except KeyError:
                return ""
        
        return ""


# Example usage for testing
if __name__ == "__main__":
    # Test the MMed refiner
    print("Testing MMed-LLM Refiner (Enhanced Version)")
    print("=" * 80)
    
    refiner = MMedBasedRefiner(ckpt="Henrychur/MMed-Llama-3-8B")
    
    test_concepts = (
        "The color is highly variable, often with multiple colors (black, brown, red, white, blue), "
        "the shape is irregular, "
        "the border is sharp and well-defined, "
        "the dermoscopic patterns are regular pigment network, symmetric dots and globules, "
        "the texture is smooth, "
        "the symmetry is asymmetrical, "
        "the elevation is flat to raised."
    )
    
    test_feedback = """Clinical inconsistency: Asymmetric lesions typically have irregular borders, but 'sharp and well-defined' border was predicted.
Clinical inconsistency: Multiple colors typically indicate complex/irregular patterns, but 'regular' patterns were predicted."""
    
    test_dict = {
        'color': 'highly variable, often with multiple colors (black, brown, red, white, blue)',
        'shape': 'irregular',
        'border': 'sharp and well-defined',
        'dermoscopic patterns': 'regular pigment network, symmetric dots and globules',
        'texture': 'smooth',
        'symmetry': 'asymmetrical',
        'elevation': 'flat to raised'
    }
    
    print("\nOriginal:")
    print(test_concepts)
    print("\nFeedback:")
    print(test_feedback)
    
    # Test extraction function
    print("\n--- Testing Violated Concepts Extraction ---")
    violated = refiner._extract_violated_concepts(test_feedback)
    print(f"Violated concepts: {violated}")
    
    print("\n--- Testing Refinement ---")
    refined = refiner(test_concepts, test_feedback, test_dict)
    print("Refined:")
    print(refined)
    
    # Test salvage function with bad output
    print("\n--- Testing Salvage Function ---")
    bad_output = """Here's the refined version:
Color is multiple colors, shape irregular, border is blurry, patterns are atypical.
Hope this helps!"""
    print("Bad output:")
    print(bad_output)
    salvaged = refiner._try_extract_concepts(bad_output, test_dict)
    print("Salvaged:")
    print(salvaged)
    
    print("\n✓ Test complete!")
