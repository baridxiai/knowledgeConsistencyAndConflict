import torch
import numpy as np
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    repeat_kv,
    apply_rotary_pos_emb,
)
import math
import torch
import torch.nn.functional as F
from torch import nn


class AttentionWrapper(torch.nn.Module):
    def __init__(self, block):
        self.block = block

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.block.config.pretraining_tp > 1:
            key_value_slicing = (
                self.block.num_key_value_heads * self.block.head_dim
            ) // self.block.config.pretraining_tp
            query_slices = self.block.q_proj.weight.split(
                (self.block.num_heads * self.block.head_dim)
                // self.block.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.block.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.block.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.block.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.block.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.block.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.block.q_proj(hidden_states)
            key_states = self.block.k_proj(hidden_states)
            value_states = self.block.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.block.num_heads, self.block.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.block.num_key_value_heads, self.block.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.block.num_key_value_heads, self.block.head_dim
        ).transpose(1, 2)

        cos, sin = self.block.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.block.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.block.num_key_value_groups)
        value_states = repeat_kv(value_states, self.block.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.block.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        if "suppression_constant" in kwargs:
            if kwargs["suppression_constant"] is not None:
                for batch_idx in range(len(attn_weights)):
                    attn_weights[
                        batch_idx,
                        :,
                        kwargs["tgt_pos"][batch_idx],
                        kwargs["subject_tokens_positions"][batch_idx],
                    ] *= kwargs["suppression_constant"]
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.block.attention_dropout, training=self.block.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.block.num_heads,
            q_len,
            self.block.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.block.num_heads, q_len, self.block.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.block.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.block.hidden_size // self.block.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.block.o_proj.weight.split(
                self.block.hidden_size // self.block.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.block.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.block.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, norm, unembed_matrix=None):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.attn_states_unembedded = None
        self.ffn_states_unembedded = None
        self.output_unembedded = None
        self.output = None

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
            if kwargs["ffn_intervention"] is not None and kwargs["ffn_intervention_position"] is not None:
                if kwargs["intervention_mode"] == "==":
                    self.ffn_states[:, kwargs["ffn_intervention_position"],:] = kwargs["ffn_intervention"]
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
        output = hidden_states
        # if kwargs["output_attentions"]:
        #     output += (self.self_attn_weights,)
        # if kwargs["use_cache"]:
        #     output += (present_key_value,)
        if self.unembed_matrix is not None:
            self.attn_states_unembedded = self.unembed_matrix(
                self.norm(self.attn_states)
            )
            # self.ffn_states_unembedded = self.unembed_matrix(self.norm(self.ffn_states))
            self.output_unembedded = self.unembed_matrix(self.norm(self.output))
        return output

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
        self.head_unembed = self.model.lm_head
        self.device = next(self.model.parameters()).device
        head = self.head_unembed

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.model.norm, head
            )

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device), max_length=max_length
        )
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def generate_intermediate_text(
        self, layer_idx, prompt, max_length=100, temperature=1.0
    ):
        layer = self.model.model.layers[layer_idx]
        for _ in range(max_length):
            self.get_logits(prompt)
            next_id = self.sample_next_token(
                layer.block_output_unembedded[:, -1], temperature=temperature
            )
            prompt = self.tokenizer.decode(
                self.tokenizer.encode(prompt)[1:] + [next_id]
            )
            if next_id == self.tokenizer.eos_token_id:
                break
        return prompt

    def sample_next_token(self, logits, temperature=1.0):
        assert temperature >= 0, "temp must be geq 0"
        if temperature == 0:
            return self._sample_greedy(logits)
        return self._sample_basic(logits / temperature)

    def _sample_greedy(self, logits):
        return logits.argmax().item()

    def _sample_basic(self, logits):
        print(logits.shape)
        return (
            torch.distributions.categorical.Categorical(logits=logits).sample().item()
        )

    def logits_fn(
        self,
        input_ids,
        attention_mask=None,
        tgt_layer=[],
        tgt_initialization=None,
        ffn_intervention=[],
        intervention_mode="==",
        ffn_intervention_position = -1,
    ):
        if len(tgt_layer) == 0:
            hidden_states = self.model.model.embed_tokens(input_ids)
        else:
            if tgt_layer[0] == -1:
                hidden_states = self.model.model.embed_tokens(input_ids)
            else:
                if tgt_initialization is not None:
                    hidden_states = tgt_initialization
                else:
                    hidden_states = self.model.model.embed_tokens(input_ids)
        cache_position = torch.arange(
            0, 0 + hidden_states.shape[1], device=hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)
        for i in range(tgt_layer[0], len(self.model.model.layers)):
            if i in tgt_layer:
                hidden_states = self.model.model.layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    ffn_intervention=ffn_intervention,
                    intervention_mode=intervention_mode,
                    ffn_intervention_position = ffn_intervention_position
                )
            else:
                hidden_states = self.model.model.layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits

    def get_logits(
        self,
        input_ids,
        attention_mask,
        tgt_layer=[-1],
        ffn_intervention=None,
        tgt_initialization=None,
        ffn_intervention_position = -1,
        intervention_mode="==",
        grad=False,
    ):
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        if grad:
            logits = self.logits_fn(
                input_ids,
                attention_mask,
                tgt_layer,
                ffn_intervention=ffn_intervention,
                tgt_initialization=tgt_initialization,
                intervention_mode=intervention_mode,
                ffn_intervention_position = ffn_intervention_position
            )
        else:
            with torch.no_grad():
                logits = self.logits_fn(
                    input_ids,
                    attention_mask,
                    tgt_layer,
                    ffn_intervention=ffn_intervention,
                    tgt_initialization=tgt_initialization,
                    intervention_mode=intervention_mode,
                    ffn_intervention_position = ffn_intervention_position
                )
        return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def set_add_to_last_tensor(self, layer, tensor):
        print("setting up intervention: add tensor to last soft token")
        self.model.model.layers[layer].block_add_to_last_tensor(tensor)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(
            label,
            list(zip(indices.detach().cpu().numpy().tolist(), tokens, probs_percent)),
        )

    def logits_all_layers(
        self,
        text,
        return_attn_mech=False,
        return_intermediate_res=False,
        return_mlp=False,
        return_block=True,
    ):
        if return_attn_mech or return_intermediate_res or return_mlp:
            raise NotImplemented("not implemented")
        self.get_logits(text)
        tensors = []
        for i, layer in enumerate(self.model.model.layers):
            tensors += [layer.block_output_unembedded.detach().cpu()]
        return torch.cat(tensors, dim=0)

    def hidden_state_all_layers(
        self,
        inputs,
        return_attn_mech=False,
        return_intermediate_res=False,
        return_mlp=False,
        return_block=True,
    ):
        if return_attn_mech or return_intermediate_res or return_mlp:
            raise NotImplemented("not implemented")
        self.model(**inputs, max_new_tokens=10)
        tensors = []
        for i, layer in enumerate(self.model.model.layers):
            tensors.append(layer.output.detach().cpu())
        return tensors

    def decode_all_layers(
        self,
        text,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        print("Prompt:", text)
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_mech_output_unembedded, "Attention mechanism"
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_res_unembedded, "Intermediate residual stream"
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_output_unembedded, "MLP output"
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output"
                )

    def scaled_input(self, emb: torch.Tensor, batch_size: int, num_batch: int = 1):
        """ "
        Create a batch of activations delta

        @param emb: activation tensor from the hidden states in FFN [1*intermediate_dim]
        @param batch_size: how man instances within one batch of deltas
        @param num_batch: total number of delta batches

        @return res: batches of activation delta [num_of_points*intermediate_dim]
        @return grad_step: batches of activation delta [1*intermediate_dim]
        """
        baseline = torch.zeros_like(emb)  # 1*intermed_dim

        num_points = batch_size * num_batch
        grad_step = (emb - baseline) / num_points

        res = torch.cat(
            [torch.add(baseline, grad_step * i) for i in range(num_points)], dim=0
        )  # batch
        return res, grad_step

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
        tgt_prob = self.get_logits(
            input_ids, attention_mask, grad=True
        )  # (batch, max_len, hidden_size), (batch, max_len, ffn_size)
        tgt_prob = tgt_prob[:, -1, :].squeeze(1)
        ig2 = None
        mlp_output = self.model.model.layers[tgt_layer].ffn_states[0, -1, :]
        if tgt_layer >0:
            before_tgt = self.model.model.layers[tgt_layer - 1].output
        else:
            before_tgt = None
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
        tgt_prob = self.get_logits(
            batch_input_ids,
            batch_attention_mask,
            ffn_intervention=[ig2],
            tgt_layer=[tgt_layer],
            tgt_initialization=before_tgt,
            intervention_mode="==",
            ffn_intervention_position = -1,
            grad=True)[:, -1, :].squeeze(1)
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
