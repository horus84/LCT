import torch
import numpy as np
from transformers import LogitsProcessor

class DiagnosticLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, protocol_tokens):
        self.tokenizer = tokenizer
        self.protocol_tokens = protocol_tokens  # List of token IDs expected in the protocol
        self.step_taxes = []  # -log(alpha_t)
        self.cumulative_tax = 0.0
        self.reachability_status = []
        
        # We simulate the allowed masks for unit testing without full outlines parser
        # In the real script, this will call outlines.fsm.guide
        self.mock_allowed_masks = []
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores is unmodified logits from the model before ANY processor masks it
        # scores shape: [batch_size, vocab_size]
        
        if self.step < len(self.mock_allowed_masks):
            allowed_ids = self.mock_allowed_masks[self.step]
        else:
            allowed_ids = torch.arange(scores.shape[-1], device=scores.device)
            
        # 1. Compute unconstrained probabilities
        probs = torch.softmax(scores, dim=-1)
        
        # 2. Compute feasible mass alpha_t
        alpha_t = probs[:, allowed_ids].sum(dim=-1).item()
        
        # 3. Compute projection tax
        # Protect against log(0)
        tax = -np.log(max(alpha_t, 1e-9))
        self.step_taxes.append(tax)
        self.cumulative_tax += tax
        
        # 4. Check Reachability (Mechanism A)
        # Check if the remaining protocol tokens are reachable. For this unit test,
        # we do a simple check if the first required protocol token is in the allowed set.
        if len(self.protocol_tokens) > 0:
            first_req = self.protocol_tokens[0]
            is_reachable = first_req in allowed_ids
            self.reachability_status.append(is_reachable)
        
        # 5. Apply Mask (simulate grammar engine)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, allowed_ids] = False
        masked_scores = scores.clone()
        masked_scores[mask] = -float('inf')
        
        self.step += 1
        return masked_scores

def test_instrumentation():
    print("Running Instrumentation Unit Tests...")
    
    vocab_size = 10
    # Create fake logits: Token 0 has high probability, Token 1 has low probability
    # logits = [10.0, 0.0, 0.0, ...] -> Softmax will heavily favor Token 0
    logits = torch.zeros((1, vocab_size))
    logits[0, 0] = 10.0
    logits[0, 1] = 0.0
    
    # Token 1 is our protocol token
    processor = DiagnosticLogitsProcessor(tokenizer=None, protocol_tokens=[1])
    
    # Scenario A: Protocol Exclusion (Mechanism A)
    # Grammar strictly forces Token 2, masking out Token 0 and Token 1
    processor.mock_allowed_masks.append(torch.tensor([2]))
    
    out_logits = processor(torch.tensor([[0]]), logits.clone())
    assert processor.reachability_status[0] == False, "Reachability should be False (Token 1 not allowed)"
    
    # Scenario B: Semantic Projection Distortion (Mechanism B)
    # Grammar allows Token 1 and 2, but masks Token 0 (the model's preference)
    processor.mock_allowed_masks.append(torch.tensor([1, 2]))
    
    out_logits = processor(torch.tensor([[0]]), logits.clone())
    assert processor.reachability_status[1] == True, "Reachability should be True (Token 1 is allowed)"
    
    # Feasible mass should be small because Token 0 (prob ~1.0) is masked out
    # Remaining tokens share the very small remaining probability mass
    tax = processor.step_taxes[1]
    assert tax > 5.0, f"Expected high projection tax, got {tax}"
    
    print("Unit Tests Passed!")
    print(f"Cumulative Tax Example: {processor.cumulative_tax}")

if __name__ == '__main__':
    test_instrumentation()
