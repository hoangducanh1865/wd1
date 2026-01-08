import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.utils import is_peft_available
from trl.trainer.grpo_config import GRPOConfig

if is_peft_available():
    from peft import PeftConfig

from wd1.trainers.rev_grpo_trainer import RevDiffuGRPOTrainer


# Sinkhorn Algorithm Implementation in PyTorch
def sinkhorn_torch(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
    """
    Sinkhorn Knopp algorithm for solving the Entropic Regularized Optimal Transport problem.

    Args:
        a: Source weights (batch_size, n)
        b: Target weights (batch_size, m)
        M: Cost matrix (batch_size, n, m)
        reg: Regularization term > 0
        numItermax: Max number of iterations
        stopThr: Stop threshold

    Returns:
        T: Transport plan (batch_size, n, m)
    """
    # Ensure inputs are correct dimensions
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    if len(M.shape) == 2:
        M = M.unsqueeze(0)

    batch_size, n, m = M.shape

    # Initialize u and v
    u = (torch.ones((batch_size, n), device=M.device) / n).to(M.dtype)
    v = (torch.ones((batch_size, m), device=M.device) / m).to(M.dtype)

    K = torch.exp(-M / reg)

    # Precompute Kp can be tricky if a is updated, but here a is constant

    for i in range(numItermax):
        u_prev = u

        # Update v
        # v = b / (K.T @ u)
        # using batch matmul: K is (B, N, M), u is (B, N)
        # K.transpose(1, 2) is (B, M, N)
        # (B, M, N) @ (B, N, 1) -> (B, M, 1)
        K_transpose_u = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
        v = b / (K_transpose_u + 1e-16)

        # Update u
        # u = a / (K @ v)
        # (B, N, M) @ (B, M, 1) -> (B, N, 1)
        K_v = torch.bmm(K, v.unsqueeze(2)).squeeze(2)
        u = a / (K_v + 1e-16)

        if i % 20 == 0:
            err = torch.norm(u - u_prev)
            if err < stopThr:
                break

    # Compute transport plan T = diag(u) K diag(v)
    # T_ij = u_i * K_ij * v_j
    T = u.unsqueeze(2) * K * v.unsqueeze(1)
    return T


class OTWD1Trainer(RevDiffuGRPOTrainer):
    """
    OT-WD1 Trainer: Using Optimal Transport (Sinkhorn) to re-weight samples.
    Extends RevDiffuGRPOTrainer.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[str, PreTrainedModel, Callable],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        reg_epsilon: float = 0.1,  # Entropic regularization parameter
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.reg_epsilon = reg_epsilon

    def get_embeddings(self, model, input_ids, attention_mask):
        # We need embeddings for the cost matrix.
        # We perform a forward pass with the full (unmasked) input.
        # This adds computational cost but is necessary for OT-wd1.
        # We take the hidden state of the last token as the representation.

        # Note: model is usually wrapped (PeftModel or similar).
        # We need to access the base transformer to get hidden states.
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Use the last hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Use last valid token based on attention_mask
        # attention_mask: 1 for valid, 0 for pad. Left-padded or right-padded?
        # Usually trl/transformers right-pad for training? No, usually left-pad for generation.
        # But here inputs are "prompt_ids" + "completion_ids".
        # Let's use the sequence lengths to find the last token.
        # tensor.argmin() on flipped mask?

        # Safer strategy: gather the token at the index of the last '1'.
        # length - 1 if all ones.
        # If right padded: index = sum(mask) - 1.
        # If left padded: index = length - 1 (since the valid tokens are at the end).

        # GRPO buffers inputs. "padding_side='left'" in processing_class?
        # In run_train.py: tokenizer.padding_side="left"??
        # Usually for generation it is left.
        # But completion is appended to right.
        # So "prompt" is left-padded, "completion" is valid (maybe right padded if batching different completion lengths?).
        # prompt_ids comes from inputs.
        # completion_ids is generated. 'gen_length' is fixed?
        # Actually RevDiffuGRPOTrainer.generate uses 'gen_length' fixed.
        # So completion is likely fixed length?
        # But EOS token might appear.

        # In compute_loss:
        # prompt_ids, prompt_mask = inputs["prompt_ids"], ...
        # input_ids = cat([prompt, completion])

        # Let's construct attention_mask
        # last_token_indices = attention_mask.sum(dim=1) - 1
        # This works if left-padding is used (000111) -> sum=3 -> index 2? No index 5.
        # If left-padding (0,0,0,1,1,1), last token is at -1 (index 5).
        # If right-padding (1,1,1,0,0,0), last token is at index 2.

        # Assumption: We want the EOS token representation or the last generated token.
        # completion_mask tracks valid completion tokens (up to EOS).
        # In compute_loss of RevGRPO:
        # "completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()"
        # This suggests right-padding-like behavior for completion (mask becomes 0 after EOS).

        # So input_ids = [Prompt (left pad)] + [Completion (masked after EOS)]
        # The 'attention_mask' for the full sequence would be (Prompt_Mask) concatenated with (Completion_Mask).

        # To get the embedding of the *last valid token* (the EOS token, or last step):
        # We can find the index where the mask transitions from 1 to 0 in the completion part,
        # or just the last 1.

        # We'll compute full mask.
        # And gather.

        # Find the last index where attention_mask is 1
        # Flip, find first 1, convert index back
        last_indices = (
            attention_mask.size(1) - attention_mask.flip(dims=[1]).to(dtype=torch.bool).int().argmax(dim=1) - 1
        )

        # Ensure indices are valid
        last_indices = last_indices.clamp(min=0, max=input_ids.size(1) - 1)

        # Gather (B, 1, D)
        embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_indices]

        # Normalize embeddings for cosine distance
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The OTWD1Trainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        # Note: We need the full input for embedding calculation
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (flattened_batch, seq_len)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Structural params
        num_generations = self.num_generations
        total_batch_size = input_ids.size(0)
        batch_size_prompts = total_batch_size // num_generations

        # Compute embeddings for Cost Matrix
        # Note: input_ids here are NOT masked.
        embeddings = self.get_embeddings(model, input_ids, attention_mask)  # (B*G, D)

        embeddings = embeddings.view(batch_size_prompts, num_generations, -1)  # (B, G, D)

        # Compute Cost Matrix C per prompt group
        # C_ij = 1 - cosine_sim(e_i, e_j)
        # bmm: (B, G, D) @ (B, D, G) -> (B, G, G) cosine similarity matrix
        cos_sim = torch.bmm(embeddings, embeddings.transpose(1, 2))
        C = 1.0 - cos_sim  # Cost matrix (B, G, G)

        # Advantages
        advantages = inputs["advantages"]  # (B*G,)
        advantages = advantages.view(batch_size_prompts, num_generations)  # (B, G)

        # Target distribution \nu (b) - derived from rewards
        # We use softmax of advantages
        target_dist = torch.softmax(advantages, dim=1)  # (B, G) sum=1

        # Source distribution \mu (a) - uniform
        source_dist = torch.ones_like(target_dist) / num_generations

        # Solve Optimal Transport (Sinkhorn)
        # T: (B, G, G) transport plan
        T = sinkhorn_torch(source_dist, target_dist, C, reg=self.reg_epsilon, numItermax=50)

        # Compute Marginals \tau = T \cdot 1
        tau = T.sum(dim=1)  # (B, G)

        # Flatten tau for loss computation
        tau = tau.view(-1)  # (B*G)

        # Stability check
        if torch.isnan(tau).any():
            tau = torch.softmax(inputs["advantages"], dim=0)

        # --- Standard WD1 Loss Computation with new weights ---

        seq_len = input_ids.shape[-1]
        prompt_length = prompt_ids.shape[-1]
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        prompt_index[:prompt_length] = True

        this_itr_idx = self._step % self.args.num_iterations
        input_ids = input_ids.unsqueeze(0)

        logits_to_keep = completion_ids.size(1)
        this_itr_mask_seed = mask_seeds[this_itr_idx]

        per_token_logps = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed])

        # Detach tau
        tau = tau.detach()

        # OT-wd1 loss
        ot_loss = -((per_token_logps[0] * completion_mask).sum(-1) * tau).sum()

        return ot_loss
