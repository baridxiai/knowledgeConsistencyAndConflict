# Taken from https://github.com/theNamek/Bias-Neurons/blob/main/bias_neuron_src/custom_bert_bias.py with several adjustments 

import copy
import json
import math
import logging

import torch
from torch import nn
import torch.nn.functional as F

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
from typing import List, Dict

logger = logging.getLogger(__name__)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertConfig(object):

    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                            "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, suppression_constant: float = None, subject_tokens_positions: List[List[int]] = None, tgt_pos: List[int] = None):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [batch*seq_length*head*seq_lengt*seq_length]

        # Causal intervention
        if suppression_constant is not None:
            for batch_idx in range(len(attention_probs)):
                attention_probs[batch_idx, :,  tgt_pos[batch_idx], subject_tokens_positions[batch_idx]] *= suppression_constant

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, suppression_constant: float = None, subject_tokens_positions: List[List[int]] = None, tgt_pos: List[int] = None):
        self_output, att_score = self.self(input_tensor, attention_mask, suppression_constant, subject_tokens_positions, tgt_pos)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, att_score


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states, tgt_pos: List[int] = None, tmp_score=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # [batch, max_len, nslot]
        if tmp_score is not None:
            for batch_idx in range(len(tmp_score)):
                hidden_states[batch_idx, tgt_pos[batch_idx], :] = tmp_score[batch_idx]
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, tgt_pos: List[int] = None, tmp_score=None, suppression_constant: float = None, subject_tokens_positions: List[List[int]] = None):
        attention_output, att_score = self.attention(hidden_states, attention_mask, suppression_constant, subject_tokens_positions, tgt_pos)
        intermediate_output = self.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, intermediate_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, tgt_layers=None, tgt_pos: List[int] = None, all_tmp_scores: Dict[int, torch.Tensor] = None, suppression_constant: float = None, subject_tokens_position: List[List[int]] = None):
        all_encoder_layers = []
        all_ffn_weights = dict()
        for layer_index, layer_module in enumerate(self.layer):
                if layer_index not in all_tmp_scores:
                    tmp_score = None
                else:
                    tmp_score = all_tmp_scores[layer_index]

                if layer_index in tgt_layers:
                    hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score
                                                              , suppression_constant=suppression_constant, subject_tokens_position=subject_tokens_position)
                    all_ffn_weights[layer_index] = ffn_weights
                else:
                    hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score
                                                    , suppression_constant=suppression_constant, subject_tokens_position=subject_tokens_position)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_ffn_weights


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                bert_model_embedding_weights.size(0),
                                bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PreTrainedBertModel(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        def load_config_hf(model_name):
            config_json_filepath = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
            return config_json_filepath

        def load_state_dict_hf(model_name):
            weights_filepath = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            return weights_filepath
        

        config_file = load_config_hf(pretrained_model_name)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        
        if state_dict is None:
            pretrained_state_dict = load_state_dict_hf(pretrained_model_name)
            state_dict = torch.load(pretrained_state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        # load encoder
        load(model, prefix='' if hasattr(model, 'roberta') else 'roberta.')

        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        return model


class BertModel(PreTrainedBertModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, tgt_pos: List[int] = None, tgt_layers=None, all_tmp_scores: Dict[int, torch.Tensor] = None, suppression_constant: float = None, subject_tokens_positions: List[List[int]] = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, all_ffn_weights = self.encoder(embedding_output,
                                    extended_attention_mask,
                                    tgt_layers=tgt_layers,
                                    tgt_pos=tgt_pos,
                                    all_tmp_scores=all_tmp_scores,
                                    suppression_constant=suppression_constant,
                                    subject_tokens_positions=subject_tokens_positions
                                    )
        sequence_output = encoded_layers[-1]
        return sequence_output, all_ffn_weights, encoded_layers


class BertForMaskedLM(PreTrainedBertModel):

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None
                , tgt_pos: List[int] = None, tgt_layers=None
                , all_tmp_scores: Dict[int, torch.Tensor] = None
                , suppression_constant: float = None, subject_tokens_positions: List[List[int]] = None
                , tgt_label=None, calculate_grad=False):
        if all_tmp_scores is not None and calculate_grad:
            tmp_score = all_tmp_scores[tgt_layers[0]]
            batch_size = tmp_score.shape[0]
            input_ids = input_ids.repeat(batch_size, 1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.repeat(batch_size, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(batch_size, 1)
        last_hidden, all_ffn_weights, hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                                                                , tgt_pos=tgt_pos, tgt_layers=tgt_layers, all_tmp_scores=all_tmp_scores, suppression_constant=suppression_constant
                                                                , subject_tokens_positions=subject_tokens_positions)  # (batch, max_len, hidden_size), (batch, max_len, ffn_size)
        for batch_idx in range(len(last_hidden)):
            last_hidden[batch_idx] = last_hidden[batch_idx, tgt_pos[batch_idx], :]  # (batch, hidden_size)
        for layer_idx in all_ffn_weights.keys():
            for batch_idx in range(len(all_ffn_weights[layer_idx])):
                all_ffn_weights[layer_idx][batch_idx] = all_ffn_weights[layer_idx][batch_idx, tgt_pos[batch_idx], :]  # (batch, ffn_size)
        
        tgt_logits = self.cls(last_hidden)  # (batch, n_vocab)
        tgt_prob = F.softmax(tgt_logits, dim=1)  # (batch, n_vocab)
        
        if not calculate_grad or all_tmp_scores is None:
            # return ffn_weights at a layer and the final layer logits at the [MASK] position and also hidden layer
            return all_ffn_weights, tgt_logits, hidden_states
        else:
            # return final probabilities and grad at a layer at the [MASK] position
            gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, tgt_label]), all_tmp_scores[tgt_layers[0]])
            return tgt_prob.detach().cpu().numpy(), gradient[0].detach().cpu().numpy()