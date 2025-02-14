import torch
import numpy as np
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)
import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, idx):
        super().__init__()
        self.block = block
        self.idx = idx

    def forward(self, hidden_states, *args, **kwargs):
        residual = hidden_states

        hidden_states = self.block.input_layernorm(hidden_states)

        # Self Attention
        if "act_as_identity" in kwargs:
            if kwargs["act_as_identity"] is not None:
                kwargs["attention_mask"] += kwargs["attention_mask"][
                    0, 0, 0, 1
                ] * torch.tril(
                    torch.ones(
                        kwargs["attention_mask"].shape,
                        dtype=kwargs["attention_mask"].dtype,
                        device=kwargs["attention_mask"].device,
                    ),
                    diagonal=-1,
                )
        self.attn_hidden_states, self.self_attn_weights, present_key_value = (
            self.block.self_attn(hidden_states, *args, **kwargs)
        )

        hidden_states = residual + self.attn_hidden_states
        if "attention_intervention" in kwargs:
            if kwargs["attention_intervention"] is not None:
                hidden_states = (
                    residual
                    + self.attn_hidden_states
                    + kwargs["attention_intervention"]
                )
        residual = hidden_states
        self.attn_states = self.block.post_attention_layernorm(hidden_states)

        # Fully Connected

        if "ffn_intervention" in kwargs:
            if (
                kwargs["ffn_intervention"] is not None
                and kwargs["ffn_intervention_position"] is not None
            ):
                if kwargs["intervention_mode"] == "==":
                    self.ffn_states[:, kwargs["ffn_intervention_position"], :] = kwargs[
                        "ffn_intervention"
                    ]
                else:
                    self.ffn_states = self.block.mlp.act_fn(
                        self.block.mlp.gate_proj(self.attn_states)
                    ) * self.block.mlp.up_proj(self.attn_states)
                    if kwargs["intervention_mode"] == "+":
                        self.ffn_states += kwargs["ffn_intervention"]
            else:
                self.ffn_states = self.block.mlp.act_fn(
                    self.block.mlp.gate_proj(self.attn_states)
                ) * self.block.mlp.up_proj(self.attn_states)

        else:
            self.ffn_states = self.block.mlp.act_fn(
                self.block.mlp.gate_proj(self.attn_states)
            ) * self.block.mlp.up_proj(self.attn_states)

        hidden_states = self.block.mlp.down_proj(self.ffn_states)
        hidden_states = residual + hidden_states
        if "output_intervention" in kwargs:
            if kwargs["output_intervention"]:
                print("performing intervention: add_to_last_tensor")
                hidden_states[:, -1, :] += self.add_to_last_tensor
        self.output = hidden_states
        return (self.output, self.self_attn_weights, present_key_value)

    def block_add_to_last_tensor(self, tensor):
        self.add_to_last_tensor = tensor

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.add_to_last_tensor = None

    def get_attn_activations(self):
        return self.block.self_attn.activations


class LlamaHelper:
    def __init__(self, model, tokenizer, device="auto"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(self.model.parameters()).device

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, idx=i)

    def scaled_input(self,emb: torch.Tensor, batch_size: int, num_batch: int = 1):
        """"
        Create a batch of activations delta

        @param emb: activation tensor from the hidden states in FFN [1*intermediate_dim]
        @param batch_size: how man instances within one batch of deltas
        @param num_batch: total number of delta batches

        @return res: batches of activation delta [num_of_points*intermediate_dim]
        @return grad_step: batches of activation delta [1*intermediate_dim]
        """
        baseline = torch.zeros_like(emb) # 1*intermed_dim

        num_points = batch_size * num_batch
        grad_step = (emb - baseline) / num_points

        res = torch.cat([torch.add(baseline, grad_step * i) for i in range(num_points)], dim=0) # batch
        return res, grad_step.detach().cpu().numpy()
    def forward(
        self,
        input_ids,
        attention_mask=None,
        tgt_layer=[],
        ffn_intervention=[],
        intervention_mode="==",
        ffn_intervention_position=-1,
        **kwargs,
    ):
        past_key_values = DynamicCache()
        hidden_states = self.model.model.embed_tokens(input_ids)
        cache_position = torch.arange(
            0, 0 + hidden_states.shape[1], device=hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + hidden_states.shape[1],
            device=hidden_states.device,
        )
        for i in range(len(self.model.model.layers)):
            if i in tgt_layer:
                hidden_states, _, past_key_values = self.model.model.layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    ffn_intervention=ffn_intervention,
                    intervention_mode=intervention_mode,
                    ffn_intervention_position=ffn_intervention_position,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                )
            else:
                hidden_states, _, past_key_values = self.model.model.layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    **kwargs,
                )
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits
    def ig2(
        self,
        input_ids,
        attention_mask=None,
        tgt_layer=None,
        integration_batch_size=20,
        integration_num_batch=1,
        tgt_label=None,
    ):
        # we only use one layer as of now for ig2 grad calculation
        tgt_prob = self.forward(
                input_ids,
                attention_mask,
            )
        tgt_prob = tgt_prob[:, -1, :].squeeze(1)
        ig2 = None
        mlp_output = self.model.model.layers[tgt_layer].ffn_states[:, -1, :]
        scaled_weights, weights_step = self.scaled_input(
            mlp_output, integration_batch_size, integration_num_batch
        )  # (num_points, ffn_size), (ffn_size)
        scaled_weights.requires_grad_(True)
        total_grad = None
        batch_weights = scaled_weights.unsqueeze(1)
        batch_size = batch_weights.shape[0]
        batch_input_ids = input_ids.repeat(batch_size, 1)
        ig2 = mlp_output
        if attention_mask is not None:
            batch_attention_mask = attention_mask.repeat(batch_size, 1, 1)
        else:
            batch_attention_mask = attention_mask
        tgt_prob = self.forward(
            batch_input_ids,
            batch_attention_mask,
            ffn_intervention=[ig2],
            tgt_layer=[tgt_layer],
            intervention_mode="==",
            ffn_intervention_position=-1,
        )[:, -1, :].squeeze(1)
        grad = torch.autograd.grad(
            torch.unbind(tgt_prob[:, tgt_label]),
            batch_weights,
        )
        grad = grad[0]
        grad = grad.sum(axis=0)  # (ffn_size)
        total_grad = (
            grad if total_grad is None else np.add(total_grad, grad)
        )  # (ffn_size)
        ig2 = total_grad * weights_step
        return ig2[0].detach().cpu().numpy()
