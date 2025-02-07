import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch.nn.functional as F
import numpy as np
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False
    #https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L422
    def forward(self, *args, **kwargs):
        if self.act_as_identity:
            kwargs['attention_mask'] += kwargs['attention_mask'][0, 0, 0, 1]*torch.tril(torch.ones(kwargs['attention_mask'].shape,
                                                                                                   dtype=kwargs['attention_mask'].dtype,
                                                                                                   device=kwargs['attention_mask'].device),
                                                                                        diagonal=-1)
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block,  norm,unembed_matrix = None):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        self.add_to_last_tensor = None
        self.output = None
    def forward(self, hidden_states, *args, **kwargs):
        residual = hidden_states

        hidden_states = self.block.input_layernorm(hidden_states,*args, **kwargs)

        # Self Attention
        self.attn_hidden_states, self.self_attn_weights = self.block.self_attn(
*args, **kwargs
        )
        hidden_states = residual + self.attn_hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.block.post_attention_layernorm(hidden_states)
        self.ffn_states = self.block.mlp.act_fn(self.block.mlp.gate_proj(hidden_states)) * self.block.mlp.up_proj(hidden_states)
        hidden_states = self.block.mlp.down_proj(self.ffn_states)
        hidden_states = residual + hidden_states
        output = (hidden_states,)
        self.output = output[0]
        if self.add_to_last_tensor is not None:
            print('performing intervention: add_to_last_tensor')
            output[0][:, -1, :] += self.add_to_last_tensor
        if self.unembed_matrix is not None:
            self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
            self.attn_output = self.block.self_attn.activations
            self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(self.attn_output))
            self.attn_output += args[0]
            self.intermediate_res_unembedded = self.unembed_matrix(self.norm(self.attn_output))
            self.mlp_output = self.block.mlp(self.post_attention_layernorm(self.attn_output))
            self.mlp_output_unembedded = self.unembed_matrix(self.norm(self.mlp_output))
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
    def __init__(self, model,tokenizer, device="auto"):
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
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.model.norm, head)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def generate_intermediate_text(self, layer_idx, prompt, max_length=100, temperature=1.0):
        layer = self.model.model.layers[layer_idx]
        for _ in range(max_length):
            self.get_logits(prompt)
            next_id = self.sample_next_token(layer.block_output_unembedded[:,-1], temperature=temperature)
            prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[1:]+[next_id])
            if next_id == self.tokenizer.eos_token_id:
                break
        return prompt

    def sample_next_token(self, logits, temperature=1.0):
        assert temperature >= 0, "temp must be geq 0"
        if temperature == 0:
            return self._sample_greedy(logits)
        return self._sample_basic(logits/temperature)

    def _sample_greedy(self, logits):
        return logits.argmax().item()

    def _sample_basic(self, logits):
        print(logits.shape)
        return torch.distributions.categorical.Categorical(logits=logits).sample().item()

    def get_logits(self, prompt, grad=False):
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        if grad:
            logits = self.model(input_ids=prompt).logits
        else:
            with torch.no_grad():
                logits = self.model(input_ids=prompt).logits
        return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def set_add_to_last_tensor(self, layer, tensor):
      print('setting up intervention: add tensor to last soft token')
      self.model.model.layers[layer].block_add_to_last_tensor(tensor)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(indices.detach().cpu().numpy().tolist(), tokens, probs_percent)))

    def logits_all_layers(self, text, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_block=True):
        if return_attn_mech or return_intermediate_res or return_mlp:
            raise NotImplemented("not implemented")
        self.get_logits(text)
        tensors = []
        for i, layer in enumerate(self.model.model.layers):
            tensors += [layer.block_output_unembedded.detach().cpu()]
        return torch.cat(tensors, dim=0)
    def hidden_state_all_layers(self, inputs, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_block=True):
        if return_attn_mech or return_intermediate_res or return_mlp:
            raise NotImplemented("not implemented")
        self.model(**inputs, max_new_tokens=10)
        tensors = []
        for i, layer in enumerate(self.model.model.layers):
            tensors.append(layer.output.detach().cpu())
        return tensors


    def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        print('Prompt:', text)
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism')
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream')
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 'Block output')
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
        return res, grad_step
    def ig2(self, input_ids, attention_mask=None
                , tgt_layers: List[int]=None, integration_batch_size=20,integration_num_batch=1
                , tgt_label=None):

        # we only use one layer as of now for ig2 grad calculation

        tgt_prob = self.get_logits(input_ids, grad=True)  # (batch, max_len, hidden_size), (batch, max_len, ffn_size)
        tgt_prob = tgt_prob[:,-1,:].squeeze(1)
        for layer in tgt_layers:
            ig2 = None
            mlp_output = self.model.model.layers[layer].ffn_states[0,-1:,:]
            scaled_weights, weights_step = self.scaled_input(mlp_output, integration_batch_size, integration_num_batch)  # (num_points, ffn_size), (ffn_size)
            scaled_weights.requires_grad_(True)
            total_grad = None
            for batch_idx in range(integration_num_batch):
                batch_weights = scaled_weights[batch_idx * integration_batch_size:(batch_idx + 1) * integration_batch_size]
                all_batch_weights = {
                    layer: batch_weights
                }
                tmp_score = all_batch_weights[layer]
                batch_size = tmp_score.shape[0]
                tgt_prob = tgt_prob.repeat(batch_size, 1)
                grad = torch.autograd.grad(torch.unbind(tgt_prob[:,tgt_label]), all_batch_weights[layer],allow_unused=True)
                grad = grad[0].detach().cpu().numpy()
                grad = grad.sum(axis=0)  # (ffn_size)
                total_grad = grad if total_grad is None else np.add(total_grad, grad) # (ffn_size)
            ig2 = total_grad*weights_step
            return ig2[0].detach().cpu().numpy()