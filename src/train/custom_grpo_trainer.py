from trl import GRPOTrainer
import torch
from typing import Any, Dict, Tuple


class GRPOLogProbTrainer(GRPOTrainer):
    # I want init to be the same as GRPOTRainer and take the same arguments, except that it needs to create a new self.think_token using the tokenizer passed to init
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.think_tokens = self.tokenizer.encode("<\think>", add_special_tokens=False, return_tensors="pt").squeeze().to(self.model.device) # Tensor of shape (n,)]

    @staticmethod
    def _mask_after_last_think(
        completion_ids: torch.LongTensor,
        completion_mask: torch.LongTensor,
        think_tokens: torch.LongTensor,
        pad_token_id: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        '''
        After the last occurrence of think_tokens in each sequence,
        replace tokens with pad_token_id and zero out the corresponding mask.
        '''
        B, N = completion_ids.shape
        S = think_tokens.size(0)
        # if pattern longer than sequence, pad all and disable all mask
        if S > N:
            out_ids = completion_ids.new_full((B, N), pad_token_id)
            out_mask = completion_mask.new_zeros((B, N))
            return out_ids, out_mask
        
        # build sliding windows [B, N-S+1, S]
        windows = completion_ids.unfold(1, S, 1)
        # compare to think_tokens → [B, N-S+1]
        matches = (windows == think_tokens).all(dim=2)

        # find last match per batch
        idx = torch.arange(matches.size(1), device=completion_ids.device)
        idx = idx.unsqueeze(0).expand_as(matches)
        last_start = torch.where(matches, idx, idx.new_full((), -1)).max(dim=1).values
        
        # compute where to start padding
        mask_begin = (last_start + S).clamp(min=0, max=N)
        arange = torch.arange(N, device=completion_ids.device).unsqueeze(0)
        pad_positions = arange >= mask_begin.unsqueeze(1)  # [B, N]

        # apply pad token and zero mask
        out_ids = completion_ids.clone()
        out_ids[pad_positions] = pad_token_id
        out_mask = completion_mask.clone()
        out_mask[pad_positions] = 0

        return out_ids, out_mask
    
    def _compute_loss(self, model, inputs):
        completion_ids = inputs["completion_ids"]

        # For this GRPO implementation we don't want to compute the loss on the generated answer 
        # as it is not considered in the logprob reward. As such, we pad everything after the <\think> token

        # add pads to everything after last <\\think> in completion_ids
        padded_completion_ids, padded_completion_mask = self._mask_after_last_think(
            inputs['completion_ids'],
            inputs['completion_mask'],
            self.think_tokens,
            self.tokenizer.pad_token_id
        )
        inputs = inputs.copy()  # avoid in-place
        inputs['completion_ids']   = padded_completion_ids
        inputs['completion_mask'] = padded_completion_mask

        # lets call the super _compute_loss with the new inputs
        return super()._compute_loss(model, inputs)