from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from tqdm.contrib import tzip
import numpy as np
from typing import Dict, List, Tuple, Union
import pdb

def print_gpu_memory_usage():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def pop_first_and_fetch(lst):
    val = lst[0]
    lst.pop(0)
    return val

def initialize_encoder_model_and_tokenizer_per_task(model_name, task_type, is_tf=False, num_classification_labels=3):
    if task_type == 'qa':
        if is_tf:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, from_tf=True)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif 'nli' in task_type:
        if is_tf:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True, num_labels=num_classification_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classification_labels)
    else:
        if is_tf:
            model = AutoModelForMaskedLM.from_pretrained(model_name, from_tf=True)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def initialize_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to('cuda')
    return model, tokenizer

class EncoderWrapper:
    """
    Wrapper of the Huggingface Encoder model class that contains several functions specific on downstream tasks and
    and enables to extact predictions from the intermediate layers using LogitLens idea

    @attribute model=a model that we want to wrap
    @attribute tokenizer=tokenizer
    @attribute task_type=downstream task
    """

    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, task_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type
    

    def _tokenize_obj(self, obj_labels: List[str]) -> [list[str], List[int], int]:
        """

        Tokenize object entity

        @param obj_labels: object entity tokens

        @return all_obj_tokens: tokenized object entity
        @return obj_token_lengths: the length of the object entity (excluding padding token)
        @return max_token_len: maximum object entity length (excluding padding token) within one batch
        """
        all_obj_tokens = []
        obj_token_lengths = []
        for obj_label in obj_labels:
            obj_tokens = self.tokenizer(obj_label)["input_ids"][1:-1]
            obj_token_lengths.append(len(obj_tokens))
            all_obj_tokens.append(obj_tokens)
        
        max_token_len = max(obj_token_lengths)

        # add padding
        for i in range(len(all_obj_tokens)):
            num_pad_tokens = max_token_len-obj_token_lengths[i]
            all_obj_tokens[i] += [self.tokenizer.pad_token_id]*num_pad_tokens
        return all_obj_tokens, obj_token_lengths, max_token_len
    

    def _mask_sentences(self, prompts: List[str], obj_token_lengths: List[int]) -> List[str]:
        """
        Replace single mask into tokenizer's n-gram masks

        @param prompts: list of prompts/inputs]
        
        @return new_prompts: list of edited prompts/inputs
        """

        new_prompts = []
        for prompt, obj_token_length in zip(prompts, obj_token_lengths):
            new_mask = " ".join(["<mask>"]*obj_token_length)
            new_prompt = prompt.replace('[Y]', new_mask)
            new_prompt = new_prompt.replace('<mask>', self.tokenizer.mask_token)
            new_prompts.append(new_prompt)
        return new_prompts


    def _calculate_cosine_sim(a: np.array, b: np.array) -> float:
        """
        Calculate cosine sim between two vectors

        @param a: vector 1
        @param b: vector 2

        @return cos_sim: cosine similarity between a and b
        """

        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)
        sim_matrix = np.dot(a, b.T)
        cos_sim = sim_matrix/(np.outer(a_norm, b_norm) + 1e-9)
        cos_sim = cos_sim.mean(axis=-1, keepdims=False)
        cos_sim = cos_sim[0]
        return cos_sim

    
    def measure_encoder_representations_cosine_similarity(self, og_sentences: Union[List[str], List[Tuple[str, str]]]
                                                        , possible_cs_sentences_candidates: Union[List[List[str]], List[List[Tuple[str, str]]]]
                                                        , selected_layers: List[int] = []) -> Dict[int, int]:
        """
        Measure the average cosine similarity between monolingual input representation 
        and possible code-mixed input representations on each encoder layer.

        @param og_sentences: monolingual sentence
        @param possible_cs_sentences: code-mixed parallel translations of one monolingual sentence
        @param selected_layers: encoder layers that we want to analyze

        @return cos_sim_per_layer: average cosine similarity on every chosen layer
        """

        self.model.eval()
        cos_sims_per_layer = dict()
        for og_sentence, possible_cs_sentence_candidates in tzip(og_sentences, possible_cs_sentences_candidates):
            # single input like question answering:
            if isinstance(og_sentence, str):
                batched_input = [og_sentence]+possible_cs_sentence_candidates
                #
                model_inputs = self.tokenizer(batched_input, return_tensors='pt', padding=True, truncation=True).to('cuda')
       
            # double input like nli task where where we pass premise and hypothesis altogether
            else:
                batched_input1 = [og_sentence[0]]
                batched_input1 += [cand[0] for cand in possible_cs_sentence_candidates]
                batched_input2 = [og_sentence[1]]
                batched_input2 += [cand[1] for cand in possible_cs_sentence_candidates]
                model_inputs = self.tokenizer(batched_input1, batched_input2, return_tensors='pt', padding=True, truncation=True).to('cuda')
            
            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]

            for layer in selected_layers:
                assert layer < len(hidden_states)
                if layer not in cos_sims_per_layer:
                    cos_sims_per_layer[layer] = []
                current_hidden_states = hidden_states[layer].detach().cpu().numpy().mean(axis=1, keepdims=False)

                og_representation = current_hidden_states[0]
                og_representation = og_representation[np.newaxis,...]
                cs_representations = current_hidden_states[1:]
                
                cos_sim = self._calculate_cosine_sim(og_representation, cs_representations)
                cos_sims_per_layer[layer].append(cos_sim)

        for layer in cos_sims_per_layer.keys():
            cos_sims_per_layer[layer] = sum(cos_sims_per_layer[layer])/len(cos_sims_per_layer[layer])
        
        return cos_sims_per_layer


    def _find_span(self, input_ids_batch: List[List[int]], target_word_input_ids_batch: List[List[int]], pick_last: bool = False) -> List[List[int]]:
        """"
        Find span range of specified object entities

        @param input_ids_batch: List of token ids in the input [batch, seq_length]
        @param target_word_input_ids_batch: List of token ids in the specified object eneities [batch, 1-max_obj_entity_length]
        @param pick_last: whether to pick last possible span or not

        @return span_positions_per_batch: Span positions in the input token ids of specified object entities [batch, 1-max_obj_entity_length]     
        """
        span_positions_per_batch = []
        for input_ids, target_word_input_ids in zip(input_ids_batch, target_word_input_ids_batch):
            input_ids_lst = [int(token_id) for token_id in input_ids]
            input_ids_valid = [int(token_id) for token_id in input_ids if self.tokenizer.decode(token_id).strip() != '']
            input_ids_valid_indices = [idx for idx, token_id in enumerate(input_ids) if self.tokenizer.decode(token_id).strip() != '']
            target_word_input_ids = [token_id for token_id in target_word_input_ids if self.tokenizer.decode(token_id).strip() != '']
            start_pos = input_ids_valid.index(target_word_input_ids[0])
            possible_spans = []

            for left_bound in range(start_pos, len(input_ids_valid)-1):
                for right_bound in range(len(input_ids_valid)-1, left_bound-1, -1): # 15
                    input_ids_subset = input_ids_valid[left_bound:right_bound+1]

                    if len(input_ids_subset) != len(target_word_input_ids):
                        continue
                    match_values = [input_id_token==target_input_id_token for input_id_token, target_input_id_token in zip(input_ids_subset, target_word_input_ids)]
                    if sum(match_values)==len(match_values):
                        possible_spans.append(input_ids_valid_indices[left_bound:right_bound+1])
            assert len(possible_spans) > 0
                
            if pick_last:
                span_positions_per_batch.append(possible_spans[-1])
            else:
                span_positions_per_batch.append(possible_spans[0])
        return span_positions_per_batch
 
    def _calculate_subj_obj_attention_per_instance(self, tokens_attention: np.array, attention_weights_per_layer: Dict[int, Dict[int, List[float]]]
                                                   , layer: int, obj_indices_span: List[int], subj_indices_span: List[int]):
        """
        Calculates the average of sum of attention weights on all subject tokens attending particular object token across all object tokens

        @param tokens_attention: all attention weights on every head of particular encoder layer for one input
            [head, seq_length, seq_length]
        @param attention_weights_per_layer: average subject-object attention weigthts on every head of one encoder layer for all inputs
        @param layer: layer position
        @param obj_indices_span: any index belongs to part of object entity
        @param subj_indices_span: any index belongs to part of subject entity
        """
        span_attention = tokens_attention[:, obj_indices_span, :][:,:, subj_indices_span] # extract subject tokens attention of each object token z
        assert span_attention.shape[1]==len(obj_indices_span) and span_attention.shape[-1]==len(subj_indices_span)
        span_attention_weight_average_per_head = span_attention.sum(axis=-1, keepdims=False).mean(axis=1, keepdims=False) # sum over the subject tokens get the average over all object tokens
        for head_pos, attention_weight_avg in enumerate(span_attention_weight_average_per_head):
            if head_pos not in attention_weights_per_layer[layer]:
                attention_weights_per_layer[layer][head_pos] = []
            attention_weights_per_layer[layer][head_pos].append(attention_weight_avg)
    
    def __remove_blank_string(self, tokenized_inputs) -> Dict[str, torch.Tensor]:
        """
        Remove any blank string that could be problematic for processing string
        """
        processed_tokenized_inputs = dict()
        all_non_blank_token_ids = []
        max_len = -1
        
        for batch_input_ids in tokenized_inputs['input_ids']:
            non_blank_token_ids = []
            for idx, token_id in enumerate(batch_input_ids):
                if self.tokenizer.decode(token_id).strip() != '':
                    non_blank_token_ids.append(idx)
            all_non_blank_token_ids.append(non_blank_token_ids)
            if len(non_blank_token_ids) > max_len:
                max_len = len(non_blank_token_ids)
        
        for key, batched_vals in tokenized_inputs:
            for batch_idx, batched_items in enumerate(batched_vals):

                batched_items[batch_idx] = [val for idx, val in enumerate(batched_items[batch_idx]) if idx in all_non_blank_token_ids[batch_idx]]
                if len(batched_items[batch_idx]) < max_len:
                    if key == 'input_ids':
                        batched_items[batch_idx] = batched_items[batch_idx]+[self.tokenizer.pad_token_id]*(max_len-len(batched_items[batch_idx]))
                    elif key == 'attention_mask':
                        batched_items[batch_idx] = batched_items[batch_idx]+[0]*(max_len-len(batched_items[batch_idx]))          
                    else:
                        batched_items[batch_idx] = batched_items[batch_idx]+[batched_items[batch_idx][-1]]*(max_len-len(batched_items[batch_idx]))   


    def extract_attention_scores_subj_obj(self, instances: List[Dict], batch_size: int =16, selected_layers: List[int] =[]) -> Dict[int, Dict[int, float]]:
        """
        Calculates the average of sum of attention weights on all subject tokens attending particular object token across all object tokens for one input

        @param instances: all input instances containing dictionary
        @param batch_size: number of instances in one batch
        @param selected_layers: encoder layer indices that we want to analyze

        @return mono_attention_weights_per_layer: average subject-object attention weights for each head and layer 
            given all monolingual inputs
        @return cs_attention_weights_per_layer: average subject-object attention weights for each head and layer 
            given all codemixed inputs
        """

        self.model.eval()
        batch_cnt = len(instances)//batch_size
        
        for i in tqdm(range(0, batch_cnt)):

            batch = instances[i*batch_size:min((i+1)*batch_size, len(instances))]
            
            obj_labels = [instance['obj_label'].strip() for instance in batch]
            
            mono_subj_labels = [instance['subj_label_same_lang'].strip() for instance in batch]
            cs_subj_labels = [instance['subj_label_cross_lang'].strip() for instance in batch]

            # create parallel code-switching statements
            mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']).replace('[Y]', instance['obj_label']) for instance in batch]
            cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']).replace('[Y]', instance['obj_label']) for instance in batch]

            
            all_mono_subj_tokens, mono_subj_token_lengths, _ = self._tokenize_obj(mono_subj_labels)
            for i in range(len(all_mono_subj_tokens)):
                all_mono_subj_tokens[i] = all_mono_subj_tokens[i][:mono_subj_token_lengths[i]]
            
            all_cs_subj_tokens, cs_subj_token_lengths, _ = self._tokenize_obj(cs_subj_labels)
            for i in range(len(all_cs_subj_tokens)):
                all_cs_subj_tokens[i] = all_cs_subj_tokens[i][:cs_subj_token_lengths[i]]
            
            all_obj_tokens, obj_token_lengths, max_obj_token_len = self._tokenize_obj(obj_labels)
            for i in range(len(all_obj_tokens)):

                all_obj_tokens[i] = all_obj_tokens[i][:obj_token_lengths[i]]
            

            mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt')
            cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt')

            

            # find subject entity span positions
            mono_subj_spans = self._find_span(mono_inputs['input_ids'], all_mono_subj_tokens, False)
            cs_subj_spans = self._find_span(cs_inputs['input_ids'], all_cs_subj_tokens, False)

            # find object entity span positions 
            mono_obj_spans = self._find_span(mono_inputs['input_ids'], all_obj_tokens, True)
            cs_obj_spans = self._find_span(cs_inputs['input_ids'], all_obj_tokens, True)

            mono_inputs = mono_inputs.to('cuda')
            cs_inputs = cs_inputs.to('cuda')

            mono_attentions = self.model(**mono_inputs, output_attentions=True).attentions
            cs_attentions = self.model(**cs_inputs, output_attentions=True).attentions

            mono_attention_weights_per_layer = dict()
            cs_attention_weights_per_layer = dict()
            for layer in selected_layers:
                if layer not in mono_attention_weights_per_layer:
                    mono_attention_weights_per_layer[layer] = dict()
                if layer not in cs_attention_weights_per_layer:
                    cs_attention_weights_per_layer[layer] = dict()
                assert layer < len(mono_attentions)
                mono_attentions_spec_layer = mono_attentions[layer].detach().cpu().numpy()
                cs_attention_spec_layer = cs_attentions[layer].detach().cpu().numpy()
                for batch_idx in range(len(batch)):  
                    # calculate subject-object attention for monolingual input
                    mono_tokens_attention = mono_attentions_spec_layer[batch_idx]
                    
                    self._calculate_subj_obj_attention_per_instance(mono_tokens_attention, mono_attention_weights_per_layer, layer
                                                                    , mono_obj_spans[batch_idx], mono_subj_spans[batch_idx])
                    
                    # calculate subject-object attention for codemixed input
                    cs_tokens_attention = cs_attention_spec_layer[batch_idx]
                    cs_start_obj_idx = cs_obj_spans[batch_idx][0]
                    cs_end_obj_idx = cs_obj_spans[batch_idx][-1]
                    cs_start_subj_idx = cs_subj_spans[batch_idx][0] 
                    cs_end_subj_idx = cs_subj_spans[batch_idx][-1]
                    self._calculate_subj_obj_attention_per_instance(cs_tokens_attention, cs_attention_weights_per_layer, layer
                                                                    , cs_obj_spans[batch_idx], cs_subj_spans[batch_idx])

        # accumulate all attentions for every layer
        for layer in selected_layers:
            for head_pos in mono_attention_weights_per_layer[layer].keys():
                mono_attention_weights_per_layer[layer][head_pos] = sum(mono_attention_weights_per_layer[layer][head_pos])/len(mono_attention_weights_per_layer[layer][head_pos])

            for head_pos in cs_attention_weights_per_layer[layer].keys():
                cs_attention_weights_per_layer[layer][head_pos] = sum(cs_attention_weights_per_layer[layer][head_pos])/len(cs_attention_weights_per_layer[layer][head_pos])     
        return mono_attention_weights_per_layer, cs_attention_weights_per_layer            

    def _get_mask_token_positions(self, inputs: Dict[str, torch.Tensor]):
        """
        Get all positions of the first masked tokens for all inputs in batch
        
        @param input: inputs obtained from tokenizer (input_ids, attention_masks, etc)
        
        @return masked_rows: batch indidces [batch_size]
        @return masked_cols: position of first mask token [batch_size]
        """
        
        masked_indices = torch.nonzero(inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=False)
        masked_index = dict()
        masked_rows, masked_cols = [], []
        for pos in masked_indices:
            row_pos = pos[0].item()
            col_pos = pos[1].item()
            if row_pos not in masked_index:
                masked_index[row_pos] = []
            masked_index[row_pos].append(col_pos)
        
        for key in sorted(masked_index.keys()):
            masked_rows.append(key) 
            masked_cols.append(min(masked_index[key]))
        return masked_rows, masked_cols

    def _calculate_joint_proba(self, inputs: Dict[str, torch.Tensor], span_pos_rows: List[List[int]], span_pos_cols: List[List[int]]
                               , ngram_candidate: torch.Tensor, batch_indices: torch.Tensor, current_pos: torch.Tensor
                               , prev_proba: torch.Tensor
                               , layer_idx: int) -> torch.Tensor:
        """
        Calculate joint proba of current topk-frontier with all words in the vocabulary

        @param inputs: input batch from tokenizer
        @param span_pos_rows: batch indices that we want to predict the next token [(1...batch_size), 1]
        @param span_pos_cols: position of previous mask tokens that have been predicted [(1...batch_size), span_length]
        @param ngram_candidate: ngram of k-th candiate [batch_size, span_length-1]
        @param batch_indices: all indices within one batch [batch_size, 1] 
        @param current_pos: mask token position for prediction for one batch [batch_size, 1]
        @param prev_proba: probabilites of ngram_candiate [batch_size, 1]
        @param layer_idx: encoder layer in which we want to extract from

        @return joint_proba: joint probabilites between top-k and tokens in vocabulary [batch_size, 1, 1]
        """

        # add previously-predicted n-gram candidates for some inputs
        inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in inputs.items()}
        inputs_copy['input_ids'][span_pos_rows, span_pos_cols] = ngram_candidate.to('cuda')

        # log proba
        if layer_idx == -1:
            cand_logits = self.model(**inputs_copy).logits
            cand_proba = torch.log(cand_logits.softmax(dim=-1))
        else:

            cand_hidden_states = self.model(**inputs_copy, output_hidden_states=True).hidden_states
            cand_hidden_states = cand_hidden_states[layer_idx]
            cand_proba = self.model.lm_head(cand_hidden_states).softmax(dim=-1) # batch*seq_length*vocab
            cand_proba = torch.log(cand_proba)

        # get log proba of mask token that we want to predict
        cand_proba = cand_proba[batch_indices, current_pos, :].squeeze(1) # [batch*vocab]
    
        # calculate joint log probabilites
        joint_proba = prev_proba + cand_proba
        joint_proba = joint_proba.unsqueeze(1) 
        return joint_proba

    def _calculate_joint_proba_with_causal_intervention(self, 
                                                        mono_inputs: Dict[str, torch.Tensor], cs_inputs: Dict[str, torch.Tensor]
                                                        , mono_span_pos_rows: List[List[int]], cs_span_pos_rows: List[List[int]]
                                                        , mono_span_pos_cols: List[List[int]], cs_span_pos_cols: List[List[int]]
                               , mono_ngram_candidate: torch.Tensor, cs_ngram_candidate: torch.Tensor, batch_indices: torch.Tensor
                               , mono_current_pos: torch.Tensor, cs_current_pos: torch.Tensor
                               , mono_prev_proba: torch.Tensor, cs_prev_proba: torch.Tensor
                               , suppression_constant: float
                               , cs_subject_tokens_positions: List[List[int]]
                               , intervented_layers: List[int]
                               , layer_idx: int) -> torch.Tensor:
        """
        Calculate joint proba of current topk-frontier with all words in the vocabulary

        @param mono_inputs: input batch from tokenizer (monolingual)
        @param cs_inputs: input batch from tokenizer (codemixed)=
        @param mono_span_pos_rows: batch indices that we want to predict the next token [(1...batch_size), 1] (monolingual)
        @param cs_span_pos_rows: batch indices that we want to predict the next token [(1...batch_size), 1] (codemixed)
        @param mono_span_pos_cols: position of previous mask tokens that have been predicted [(1...batch_size), span_length] (monolingual)
        @param cs_span_pos_cols: position of previous mask tokens that have been predicted [(1...batch_size), span_length] (codemixed)
        @param mono_ngram_candidate: ngram of k-th candiate [batch_size, span_length-1] (monolingual)
        @param cs_ngram_candidate: ngram of k-th candiate [batch_size, span_length-1] (codemixed)
        @param batch_indices: all indices within one batch [batch_size, 1] 
        @param mono_current_pos: mask token position for prediction for one batch [batch_size, 1] (monolingual)
        @param cs_current_pos: mask token position for prediction for one batch [batch_size, 1] (codemixed)
        @param mono_prev_proba: probabilites of ngram_candiate [batch_size, 1] (monolingual)
        @param cs_prev_proba: probabilites of ngram_candiate [batch_size, 1] (codemixed)
        @param supression_constant: constant to decrease the activation value between subject and object
        @param cs_subject_tokens_positions: position of subject tokens in codemixed input
        @param intervened_layers: which FFN layers that we want to patch
        @param layer_idx: encoder layer index that we want to get prediction from

        @return mono_joint_proba: joint probabilites between top-k and tokens in vocabulary [batch_size, 1, 1] (monolingual)
        @return cs_joint_proba: joint probabilites between top-k and tokens in vocabulary [batch_size, 1, 1] (codemixed) 
        """

        # add previously-predicted n-gram candidates for some inputs
        mono_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in mono_inputs.items()}
        mono_inputs_copy['input_ids'][mono_span_pos_rows, mono_span_pos_cols] = mono_ngram_candidate.to('cuda')
        cs_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in cs_inputs.items()}
        cs_inputs_copy['input_ids'][cs_span_pos_rows, cs_span_pos_cols] = cs_ngram_candidate.to('cuda')

        # log proba
        mono_ffn_states, mono_logits, mono_hidden_states = self.model(**mono_inputs_copy, tgt_pos=mono_current_pos.squeeze(-1)
                                                                      , tgt_layers=intervented_layers) # batch*seq_length*vocab
        #pdb.set_trace()
        _, cs_logits, cs_hidden_states = self.model(**cs_inputs_copy, tgt_pos=cs_current_pos.squeeze(-1)
                                                    , tgt_layers=intervented_layers, all_tmp_scores=mono_ffn_states
                                                    , suppression_constant=suppression_constant
                                                    , subject_tokens_positions=cs_subject_tokens_positions)
        for key, tensor in mono_inputs.items():
            tensor.detach().cpu()
        
        for key, tensor in cs_inputs.items():
            tensor.detach().cpu()

        if layer_idx == -1:
            mono_cand_proba = mono_logits.softmax(dim=-1)
            mono_cand_proba = torch.log(mono_cand_proba)
            cs_cand_proba = cs_logits.softmax(dim=-1)
            cs_cand_proba = torch.log(cs_cand_proba)
        else:
            mono_hidden_states = mono_hidden_states[layer_idx]
            cs_hidden_states = cs_hidden_states[layer_idx]
            batch_positions = [[batch_idx] for batch_idx in range(len(mono_hidden_states))]
            mono_tgt_positions = [mono_current_pos[batch_idx] for batch_idx in range(len(mono_hidden_states))]
            cs_tgt_positions = [cs_current_pos[batch_idx] for batch_idx in range(len(mono_hidden_states))]
            mono_hidden_states = mono_hidden_states[batch_positions, mono_tgt_positions, :]
            cs_hidden_states = cs_hidden_states[batch_positions, cs_tgt_positions, :]
            mono_logits = self.model.cls(mono_hidden_states)
            mono_cand_proba = torch.log(mono_logits.softmax(dim=-1))
            cs_logits = self.model.cls(cs_hidden_states)
            cs_cand_proba = torch.log(cs_logits.softmax(dim=-1))


        # get log proba of mask token that we want to predict
        mono_cand_proba = mono_cand_proba.squeeze(1) # [batch*vocab]
        cs_cand_proba = cs_cand_proba.squeeze(1) # [batch*vocab]
    
        # calculate joint log probabilites
        mono_cand_proba = mono_prev_proba + mono_cand_proba
        mono_cand_proba = mono_cand_proba.unsqueeze(1)  
        cs_cand_proba = cs_prev_proba + cs_cand_proba
        cs_cand_proba = cs_cand_proba.unsqueeze(1)
        return mono_cand_proba, cs_cand_proba
    
    def _get_next_topk_proba_and_words(self, joint_proba: torch.Tensor, beam_topk: int):
        """
        Determine the proba and token of next top-k candidates based on the joint probability

        @param joint_proba: joint probability [batch*k*vocab]
        @param beam_topk: number of frontiers to select
        
        @return next_topk_log_prob: the joint probabilities of the next top-k candidates # [batch*k]
        @return vocab_indices: the chosen token of the next top-k candidates # [batch*k] 
        @return prefix_indices: the chosen n-gram prefix of the next top-k candidates # [batch*k] 
        """

        vocab_size = joint_proba.shape[-1]
        joint_proba = joint_proba.view(joint_proba.shape[0], -1) # [batch*(k*vocab)
        next_topk_log_prob, next_topk_indices = joint_proba.topk(beam_topk, sorted=True) # [batch*k], [batch*k]
        prefix_indices, vocab_indices = next_topk_indices//vocab_size, next_topk_indices%vocab_size # indices of the prefix candidates, indices of the vocab
        prefix_indices = prefix_indices.cpu()
        return next_topk_log_prob, vocab_indices, prefix_indices

    def _update_topk_ngrams(self, new_indices: np.array, prefix_ngrams: np.array, next_token: np.array, batch_idx: int):
        """
        Update the topk frontier

        @param new_indices: topk tokens that we want to update [batch*k*span_length]
        @param prefix_ngrams: chosen prefix for topk [k*span_length-1]
        @param next_token: next token for top-k [k*1]
        @param batch_idx: position of instance within batch
        """
        # for each instance, update the top-k frontier n-grams
        new_indices[batch_idx, :, :-1] = prefix_ngrams
        new_indices[batch_idx, :, -1] = next_token

    def _add_topk_indices_log_prob_to_batch(self, batch_rank_preds: List[Dict[str, float]]
                                            , new_indices: np.array, next_topk_log_prob: np.array, batch_idx: int, span_length: int):
        """
        Add current top-k log probabilites and ngram into the list of predictions for current batch

        @param batch_rank_preds: all top-k ngrams along with their probabilites
        @param new_indices: ngram of topk [batch_size, k, span_length]
        @param next_topk_log_prob: probabilites of each topk [batch_size, k, 1]
        @param span_length: span length for current iteration
        """
        topk_indices_instance = new_indices[batch_idx]
        topk_log_prob_instance = next_topk_log_prob[batch_idx]
        for curr_topk_log_prob, curr_topk_token_idx in zip(topk_log_prob_instance, topk_indices_instance):
            vocab_ids = curr_topk_token_idx.type(torch.LongTensor)
            decoded_word = self.tokenizer.batch_decode(vocab_ids)
            decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)]) # add id besides the token to distinghuish token
            batch_rank_preds[batch_idx][decoded_word] = curr_topk_log_prob.item()/(span_length+1) # normalize with length

    def _get_mask_positions(self, inputs):
        """
        Provides masked token indices for the beam search process

        @param inputs: batch input obtained from tokenizer
        ]
        """
        masked_rows, masked_cols = self._get_mask_token_positions(inputs)
            
        # all these have batch_size*1
        start_pos = np.array(masked_cols.copy())[..., np.newaxis]
        masked_rows = np.array(masked_rows)[..., np.newaxis] 
        
        return masked_rows, masked_cols, start_pos
    
    def _do_beam_search(self, mono_masked_rows: List[List[int]], mono_masked_cols: List[List[int]]
                        ,cs_masked_rows: List[List[int]], cs_masked_cols: List[List[int]]
                        , mono_log_probs: torch.Tensor, cs_log_probs: torch.Tensor
                        , mono_inputs: Dict[str, torch.Tensor], cs_inputs: Dict[str, torch.Tensor]
                        , beam_topk: int, ranking_topk: int
                       , max_obj_token_len: int, obj_token_lengths: List[int]
                       , mono_rank_preds: List[Dict[str, float]], cs_rank_preds: List[Dict[str, float]]
                       , layer_idx: int = -1):
        """
        Do beam search inference on two inputs (monolingual and  codemixed)

        @param mono_masked_rows: batch indices
        @param mono_masked_cols: position of first mask token in each batch of monolingual inputs
        @param cs_masked_rows: batch indices
        @param cs_masked_cols: position of first mask token in each batch of codemixed inputs
        @param mono_log_probs: first mask token probabilites over all words on monolingual input    [batch_size*seq_length*vocab_size]
        @param cs_log_probs: first mask token probabilites over all words on codemixed input    [batch_size*seq_length*vocab_size] 
        @param mono_inputs: input to be fed into model from tokenized monolingual input
        @param cs_inputs: input to be fed into model from tokenized codemixed input 
        @param beam_topk: beam width used for beam searcj
        @param ranking_topk: number of predictions for the final prediction
        @param max_obj_token_len: maximum object entity length within one batch
        @param obj_token_lengths: length of every object entity
        @param mono_rank_preds: all predictions for monolingual input [total_instances, dict[word, proba]]
        @param cs_rank_preds: all predictions for codemixed input [total_instances, dict[word, proba]]
        @param layer_idx: which encoder layer that we want to extract the predictions from
        """
                
        mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # [batch*seq_length*beam_topk], [batch*seq_length*beam_topk]
        cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True) # [batch*seq_length*beam_topk], [batch*seq_length*beam_topk]
        
        mono_start_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
        cs_start_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1

        mono_current_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
        cs_current_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1
        
        mono_masked_rows = np.array(mono_masked_rows)[..., np.newaxis] # batch_size*1 
        mono_masked_cols = np.array(mono_masked_cols)[..., np.newaxis] # batch_size*1
        cs_masked_rows = np.array(cs_masked_rows)[..., np.newaxis] # batch_size*1
        cs_masked_cols = np.array(cs_masked_cols)[..., np.newaxis] # batch_size*1

        # extract the topk values (probabilities and positions) of top-k of the prediction for first mask token 
        mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_masked_cols, :], mono_topk_indices[mono_masked_rows, mono_masked_cols, :] # [batch*1*k], [batch*1*k] 
        cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_masked_cols, :], cs_topk_indices[cs_masked_rows, cs_masked_cols, :] # [batch*1*k], [batch*1*k]
        
        batch_sz = mono_current_pos.shape[0]

        mono_batch_rank_preds = []
        cs_batch_rank_preds = []
        for batch_idx in range(batch_sz):
            mono_dict = dict()
            cs_dict = dict()
            mono_batch_rank_preds.append(mono_dict)
            cs_batch_rank_preds.append(cs_dict)
        
        for batch_idx in range(batch_sz):
            # get the topk probas and their mapped tokens
            mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0] # [k], [k] 
            cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0] # [k], [k]

            for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                decoded_word = f"{decoded_word}<{curr_mono_topk_token_idx}>" # we add the token id here to distinguish the difference because one token can be mapped into multiple token ids
                mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()
            
            for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                decoded_word = f"{decoded_word}<{curr_cs_topk_token_idx}>" # we add the token id here to distinguish the difference because one token can be mapped into multiple token ids
                cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob.item()

        batch_indices = torch.arange(batch_sz).unsqueeze(-1) # [batch_size, 1]

        mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1)) # [k*batch_size*seq_length]
        mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor) # [k*batch_size*seq_length]
        
        cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1)) # [k*batch_size*seq_length]
        cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor) # [k*batch_size*seq_length]
        
        for span_len in range(1, max_obj_token_len):
            mono_span_pos_rows = [] # [(1...batch_size)*1]
            mono_span_pos_cols = [] # [(1...batch_size)*(1...span_length)]

            cs_span_pos_rows = [] # [(1...batch_size)*1]
            cs_span_pos_cols = [] # [(1...batch_size)*(1...span_length)]

            selected_mono_topk_indices = []
            selected_cs_topk_indices = []

            # set span indices
            for batch_idx in range(batch_sz):

                # this means that all masked tokens havent been predicted yet for one instance
                if obj_token_lengths[batch_idx] >= (span_len+1):
                    mono_current_pos[batch_idx][0] += 1 # shift target position to the right
                    mono_span_pos_rows.append([batch_idx])
        
                    mono_span_pos_cols.append(np.arange(mono_start_pos[batch_idx][0], mono_current_pos[batch_idx][0])) # [(1...batch_size)*(span-1)]
                    selected_mono_topk_indices.append(mono_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0)) # [(1...batch_size)*k*seq_length]
            
                    cs_current_pos[batch_idx][0] += 1 # shift target position to the right
                    cs_span_pos_rows.append([batch_idx])
                    cs_span_pos_cols.append(np.arange(cs_start_pos[batch_idx][0], cs_current_pos[batch_idx][0])) # [(1...batch_size)*(span-1)]
                    selected_cs_topk_indices.append(cs_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0)) # [(1...batch_size)*k*seq_length]
            
            selected_cs_topk_indices = torch.cat(selected_cs_topk_indices, axis=0).permute((1,0,2)) # [k*(1...batch_size)*seq_length]
            selected_mono_topk_indices = torch.cat(selected_mono_topk_indices, axis=0).permute((1,0,2)) # [k*(1...batch_size)*seq_length]
            
            all_mono_joint_proba = []
            all_cs_joint_proba = []

            num_of_cands = len(mono_topk_log_prob)

            for cand_rank in range(num_of_cands):
                mono_joint_proba = self._calculate_joint_proba(mono_inputs, mono_span_pos_rows
                                                                , mono_span_pos_cols, selected_mono_topk_indices[cand_rank]
                                                                , batch_indices, mono_current_pos, mono_topk_log_prob[cand_rank], layer_idx)
                cs_joint_proba = self._calculate_joint_proba(cs_inputs, cs_span_pos_rows
                                                                , cs_span_pos_cols, selected_cs_topk_indices[cand_rank]
                                                                , batch_indices, cs_current_pos, cs_topk_log_prob[cand_rank], layer_idx)
                
                all_mono_joint_proba.append(mono_joint_proba)
                all_cs_joint_proba.append(cs_joint_proba)

            all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # [batch*k*vocab]
            all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # [batch*k*vocab]
            
            next_mono_topk_log_prob, vocab_indices_mono, prefix_indices_mono = self._get_next_topk_proba_and_words(all_mono_joint_proba, beam_topk)
            next_cs_topk_log_prob, vocab_indices_cs, prefix_indices_cs = self._get_next_topk_proba_and_words(all_cs_joint_proba, beam_topk)

            new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
            new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1)) # batch*k*len

            mono_topk_indices = mono_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
            cs_topk_indices = cs_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1

            for batch_idx in range(batch_sz):
                self._update_topk_ngrams(new_mono_indices, mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]], vocab_indices_mono[batch_idx], batch_idx)
                self._update_topk_ngrams(new_cs_indices, cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]], vocab_indices_cs[batch_idx], batch_idx)
                
                # check if this batch still requires to predict mask token
                if obj_token_lengths[batch_idx] >= (span_len+1):
                    self._add_topk_indices_log_prob_to_batch(mono_batch_rank_preds, new_mono_indices, next_mono_topk_log_prob, batch_idx, span_len)
                    self._add_topk_indices_log_prob_to_batch(cs_batch_rank_preds, new_cs_indices, next_cs_topk_log_prob, batch_idx, span_len)
            
            cs_topk_indices = new_cs_indices.permute((1,0,2)).type(torch.LongTensor) # [k*batch*len]
            mono_topk_indices = new_mono_indices.permute((1,0,2)).type(torch.LongTensor) # [k*batch*len]

            mono_topk_log_prob = next_mono_topk_log_prob.permute((1,0)).unsqueeze(-1) # [k*batch*1]
            cs_topk_log_prob = next_cs_topk_log_prob.permute((1,0)).unsqueeze(-1) # [k*batch*1]
                            
        # rank all preds
        for batch_preds in mono_batch_rank_preds:
            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
            selected_words = sorted_batch_preds[:ranking_topk]
            mono_rank_preds.append(selected_words)

        for batch_preds in cs_batch_rank_preds:
            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
            selected_words = sorted_batch_preds[:ranking_topk]
            cs_rank_preds.append(selected_words)
    

    
    def _do_beam_search_with_causal_intervention(self
                        , mono_masked_rows: List[List[int]], mono_masked_cols: List[List[int]]
                        ,cs_masked_rows: List[List[int]], cs_masked_cols: List[List[int]]
                        , mono_log_probs: torch.Tensor, cs_log_probs: torch.Tensor
                        , mono_inputs: Dict[str, torch.Tensor], cs_inputs: Dict[str, torch.Tensor]
                        , beam_topk: int, ranking_topk: int
                       , max_obj_token_len: int, obj_token_lengths: List[int]
                       , suppression_constant: float, cs_subj_token_positions: List[List[int]], intervened_layers: List[int]
                       , mono_rank_preds: List[Dict[str, float]], cs_rank_preds: List[Dict[str, float]], layer_idx: int = -1):
        """
        Do beam search inference on two inputs (monolingual and  codemixed) with some causal intervention

        @param mono_masked_rows: batch indices
        @param mono_masked_cols: position of first mask token in each batch of monolingual inputs
        @param cs_masked_rows: batch indices
        @param cs_masked_cols: position of first mask token in each batch of codemixed inputs
        @param mono_log_probs: first mask token probabilites over all words on monolingual input    [batch_size*seq_length*vocab_size]
        @param cs_log_probs: first mask token probabilites over all words on codemixed input    [batch_size*seq_length*vocab_size] 
        @param mono_inputs: input to be fed into model from tokenized monolingual input
        @param cs_inputs: input to be fed into model from tokenized codemixed input 
        @param beam_topk: beam width used for beam search
        @param ranking_topk: number of predictions for the final prediction
        @param max_obj_token_len: maximum object entity length within one batch
        @param obj_token_lengths: length of every object entity
        @param suppression_consantt: a constant to suppress the activation between subject-object
        @param cs_subj_tokens_positions: subject tokens positions in codemixed input
        @param intervened_layers: which FFN layers that we want to patch
        @param mono_rank_preds: all predictions for monolingual input [total_instances, dict[word, proba]]
        @param cs_rank_preds: all predictions for codemixed input [total_instances, dict[word, proba]]
        @param layer_idx: which encoder layer that we want to get the prediction from
        """
                
        mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # [batch*seq_length*beam_topk], [batch*seq_length*beam_topk]
        cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True) # [batch*seq_length*beam_topk], [batch*seq_length*beam_topk]
        
        mono_start_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
        cs_start_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1

        mono_current_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
        cs_current_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1
        
        mono_masked_rows = np.array(mono_masked_rows)[..., np.newaxis] # batch_size*1 
        mono_masked_cols = np.array(mono_masked_cols)[..., np.newaxis] # batch_size*1
        cs_masked_rows = np.array(cs_masked_rows)[..., np.newaxis] # batch_size*1
        cs_masked_cols = np.array(cs_masked_cols)[..., np.newaxis] # batch_size*1

        # extract the topk values (probabilities and positions) of top-k of the prediction for first mask token 
        mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_masked_cols, :], mono_topk_indices[mono_masked_rows, mono_masked_cols, :] # [batch*1*k], [batch*1*k] 
        cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_masked_cols, :], cs_topk_indices[cs_masked_rows, cs_masked_cols, :] # [batch*1*k], [batch*1*k]
        
        batch_sz = mono_current_pos.shape[0]

        mono_batch_rank_preds = []
        cs_batch_rank_preds = []
        for batch_idx in range(batch_sz):
            mono_dict = dict()
            cs_dict = dict()
            mono_batch_rank_preds.append(mono_dict)
            cs_batch_rank_preds.append(cs_dict)
        
        for batch_idx in range(batch_sz):
            # get the topk probas and their mapped tokens
            mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0] # [k], [k] 
            cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0] # [k], [k]

            for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                decoded_word = f"{decoded_word}<{curr_mono_topk_token_idx}>" # we add the token id here to distinguish the difference because one token can be mapped into multiple token ids
                mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()
            
            for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                decoded_word = f"{decoded_word}<{curr_cs_topk_token_idx}>" # we add the token id here to distinguish the difference because one token can be mapped into multiple token ids
                cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob.item()

        batch_indices = torch.arange(batch_sz).unsqueeze(-1) # [batch_size, 1]

        mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1)) # [k*batch_size*seq_length]
        mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor) # [k*batch_size*seq_length]
        
        cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1)) # [k*batch_size*seq_length]
        cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor) # [k*batch_size*seq_length]
        
        for span_len in range(1, max_obj_token_len):
            mono_span_pos_rows = [] # [(1...batch_size)*1]
            mono_span_pos_cols = [] # [(1...batch_size)*(1...span_length)]

            cs_span_pos_rows = [] # [(1...batch_size)*1]
            cs_span_pos_cols = [] # [(1...batch_size)*(1...span_length)]

            selected_mono_topk_indices = []
            selected_cs_topk_indices = []

            # set span indices
            for batch_idx in range(batch_sz):

                # this means that all masked tokens havent been predicted yet for one instance
                if obj_token_lengths[batch_idx] >= (span_len+1):
                    mono_current_pos[batch_idx][0] += 1 # shift target position to the right
                    mono_span_pos_rows.append([batch_idx])
        
                    mono_span_pos_cols.append(np.arange(mono_start_pos[batch_idx][0], mono_current_pos[batch_idx][0])) # [(1...batch_size)*(span-1)]
                    selected_mono_topk_indices.append(mono_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0)) # [(1...batch_size)*k*seq_length]
            
                    cs_current_pos[batch_idx][0] += 1 # shift target position to the right
                    cs_span_pos_rows.append([batch_idx])
                    cs_span_pos_cols.append(np.arange(cs_start_pos[batch_idx][0], cs_current_pos[batch_idx][0])) # [(1...batch_size)*(span-1)]
                    selected_cs_topk_indices.append(cs_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0)) # [(1...batch_size)*k*seq_length]
            
            selected_cs_topk_indices = torch.cat(selected_cs_topk_indices, axis=0).permute((1,0,2)) # [k*(1...batch_size)*seq_length]
            selected_mono_topk_indices = torch.cat(selected_mono_topk_indices, axis=0).permute((1,0,2)) # [k*(1...batch_size)*seq_length]
            
            all_mono_joint_proba = []
            all_cs_joint_proba = []

            num_of_cands = len(mono_topk_log_prob)

            for cand_rank in range(num_of_cands):
                mono_joint_proba, cs_joint_proba = self._calculate_joint_proba_with_causal_intervention(mono_inputs, cs_inputs, mono_span_pos_rows, cs_span_pos_rows
                                                                , mono_span_pos_cols, cs_span_pos_cols, selected_mono_topk_indices[cand_rank], selected_cs_topk_indices[cand_rank]
                                                                , batch_indices, mono_current_pos, cs_current_pos, mono_topk_log_prob[cand_rank], cs_topk_log_prob[cand_rank]
                                                                , suppression_constant, cs_subj_token_positions, intervened_layers, layer_idx)
                
                all_mono_joint_proba.append(mono_joint_proba)
                all_cs_joint_proba.append(cs_joint_proba)

            all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # [batch*k*vocab]
            all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # [batch*k*vocab]
            
            next_mono_topk_log_prob, vocab_indices_mono, prefix_indices_mono = self._get_next_topk_proba_and_words(all_mono_joint_proba, beam_topk)
            next_cs_topk_log_prob, vocab_indices_cs, prefix_indices_cs = self._get_next_topk_proba_and_words(all_cs_joint_proba, beam_topk)

            new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
            new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1)) # batch*k*len

            mono_topk_indices = mono_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
            cs_topk_indices = cs_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1

            for batch_idx in range(batch_sz):
                self._update_topk_ngrams(new_mono_indices, mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]], vocab_indices_mono[batch_idx], batch_idx)
                self._update_topk_ngrams(new_cs_indices, cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]], vocab_indices_cs[batch_idx], batch_idx)
                
                # check if this batch still requires to predict mask token
                if obj_token_lengths[batch_idx] >= (span_len+1):
                    self._add_topk_indices_log_prob_to_batch(mono_batch_rank_preds, new_mono_indices, next_mono_topk_log_prob, batch_idx, span_len)
                    self._add_topk_indices_log_prob_to_batch(cs_batch_rank_preds, new_cs_indices, next_cs_topk_log_prob, batch_idx, span_len)
            
            cs_topk_indices = new_cs_indices.permute((1,0,2)).type(torch.LongTensor) # [k*batch*len]
            mono_topk_indices = new_mono_indices.permute((1,0,2)).type(torch.LongTensor) # [k*batch*len]

            mono_topk_log_prob = next_mono_topk_log_prob.permute((1,0)).unsqueeze(-1) # [k*batch*1]
            cs_topk_log_prob = next_cs_topk_log_prob.permute((1,0)).unsqueeze(-1) # [k*batch*1]
                            
        # rank all preds
        for batch_preds in mono_batch_rank_preds:
            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
            selected_words = sorted_batch_preds[:ranking_topk]
            mono_rank_preds.append(selected_words)

        for batch_preds in cs_batch_rank_preds:
            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
            selected_words = sorted_batch_preds[:ranking_topk]
            cs_rank_preds.append(selected_words) 

    def inference_cloze_task(self, instances: List[Dict], batch_size: int = 16, selected_layers: List[int] = [], beam_topk: int = 5, ranking_topk: int = 5) -> [Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], List[str]]:
        """
        Do inference for fill-in-the-blank task using beam search.

        @param instances: all input instances containing dictionary
        @param batch_size: num of instances in one batch
        @param selected_layers: encoder layer indices that we want to analyze
        @param beam_topk: width used for beam search
        @param ranking_topk: number of selected candidates

        @return mono_rank_preds: list of top-k predictions on all monolingual inputs just from last layer 
        @return cs_rank_preds: list of top-k predictions on all codemixed inputs just from last layer
        @return mono_rank_preds_per_layer: list of top-k predictions on all monolingual inputs from selected layers
        @return cs_rank_preds_per_layer: list of top-k predictions on all codemixed inputs just from selected layers
        @return labels: list of ground truth labels
        """

        self.model.eval()

        mono_rank_preds_per_layer, cs_rank_preds_per_layer = dict(), dict()
        labels = []
        batch_cnt = len(instances)//batch_size

        mono_rank_preds = []; cs_rank_preds = []

        for layer in selected_layers:
            mono_rank_preds_per_layer[layer] = []
            cs_rank_preds_per_layer[layer] = []

        with torch.no_grad():
            for i in tqdm(range(0, batch_cnt)):
                batch = instances[i*batch_size:min((i+1)*batch_size, len(instances))]
                obj_labels = [instance['obj_label'] for instance in batch]
                
                labels.extend(obj_labels)

                # Do codemixing
                mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']) for instance in batch]
                cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']) for instance in batch]
                
                # Get tokenized object entities
                _, obj_token_lengths, max_obj_token_len = self._tokenize_obj(obj_labels)
                
                # Do n-gram masking
                mono_prompts = self._mask_sentences(mono_prompts, obj_token_lengths)
                cs_prompts = self._mask_sentences(cs_prompts, obj_token_lengths)

                mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
                cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')

                mono_masked_rows, mono_masked_cols = self._get_mask_token_positions(mono_inputs)
                cs_masked_rows, cs_masked_cols = self._get_mask_token_positions(cs_inputs)
                

                # Only getting the final output
                if len(selected_layers) == 0 or -1 in selected_layers:
                    mono_outputs = self.model(**mono_inputs, output_hidden_states=True)
                    cs_outputs = self.model(**cs_inputs, output_hidden_states=True)

                    mono_log_probs = torch.log(mono_outputs['logits'].softmax(dim=-1)) # batch*seq_length*vocab
                    cs_log_probs = torch.log(cs_outputs['logits'].softmax(dim=-1)) # batch*seq_length*vocab

                    self._do_beam_search(mono_masked_rows, mono_masked_cols, cs_masked_rows, cs_masked_cols, mono_log_probs, cs_log_probs, mono_inputs, cs_inputs, beam_topk, ranking_topk, max_obj_token_len, obj_token_lengths, mono_rank_preds, cs_rank_preds)
                    

                else:
                    mono_outputs = self.model(**mono_inputs, output_hidden_states=True)
                    cs_outputs = self.model(**cs_inputs, output_hidden_states=True)

                    mono_hidden_states = mono_outputs.hidden_states[1:]
                    cs_hidden_states = cs_outputs.hidden_states[1:]


                    for layer in  selected_layers:
                        # check if the layer isnt out of bound
                        assert layer < len(mono_hidden_states)

                        mono_logits = self.model.lm_head(mono_hidden_states[layer])
                        cs_logits = self.model.lm_head(cs_hidden_states[layer])
                        
                        
                        mono_log_probs = torch.log(mono_logits.softmax(dim=-1)) # [batch*seq_length*vocab]
                        cs_log_probs = torch.log(cs_logits.softmax(dim=-1))

                        self._do_beam_search(mono_masked_rows, mono_masked_cols, cs_masked_rows, cs_masked_cols, mono_log_probs, cs_log_probs, mono_inputs, cs_inputs, beam_topk, ranking_topk, max_obj_token_len, obj_token_lengths, mono_rank_preds_per_layer[layer], cs_rank_preds_per_layer[layer], layer) 
                        
        if len(selected_layers) == 0 or -1 in selected_layers:
            return mono_rank_preds, cs_rank_preds, labels
        else:
            return mono_rank_preds_per_layer, cs_rank_preds_per_layer, labels
    
    def inference_cloze_task_with_causal_intervention(self, instances: List[Dict], batch_size: int = 16
                                                      , selected_layers: List[int] = [], beam_topk: int = 5
                                                      , ranking_topk: int = 5
                                                      , intervened_ffn_layers: List[int] = []
                                                      , suppression_constant: float = None) -> [Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], List[str]]:
        """e
        Do inference for fill-in-the-blank task using beam search but .

        @param instances: all input instances containing dictionary
        @param batch_size: num of instances in one batch
        @param selected_layers: encoder layer indices that we want to analyze
        @param beam_topk: width used for beam search
        @param ranking_topk: number of selected candidates
        @param intervened_ffn_layers: which ffn layers that we want to manipulate
        @param suppression_constnat: subject-object activation weights suppression constant

        @return mono_rank_preds: list of top-k predictions on all monolingual inputs just from last layer 
        @return cs_rank_preds: list of top-k predictions on all codemixed inputs just from last layer
        @return mono_rank_preds_per_layer: list of top-k predictions on all monolingual inputs from selected layers
        @return cs_rank_preds_per_layer: list of top-k predictions on all codemixed inputs just from selected layers
        @return labels: list of ground truth labels
        """

        self.model.eval()

        mono_rank_preds_per_layer, cs_rank_preds_per_layer = dict(), dict()
        labels = []
        batch_cnt = len(instances)//batch_size

        mono_rank_preds = []; cs_rank_preds = []

        for layer in selected_layers:
            mono_rank_preds_per_layer[layer] = []
            cs_rank_preds_per_layer[layer] = []

        with torch.no_grad():
            for i in tqdm(range(0, batch_cnt)):
                batch = instances[i*batch_size:min((i+1)*batch_size, len(instances))]
                obj_labels = [instance['obj_label'] for instance in batch]
                cs_subj_labels = [instance['subj_label_cross_lang'] for instance in batch]
                
                labels.extend(obj_labels)

                # Do codemixing
                mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']) for instance in batch]
                cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']) for instance in batch]

            
                # Get tokenized object entities
                all_obj_tokens, obj_token_lengths, max_obj_token_len = self._tokenize_obj(obj_labels)
                
                # Do n-gram masking
                mono_prompts = self._mask_sentences(mono_prompts, obj_token_lengths)
                cs_prompts = self._mask_sentences(cs_prompts, obj_token_lengths)

                mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
                cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')

                # Find subject and object positions
                mono_subj_labels = [instance['subj_label_same_lang'] for instance in batch]
                all_mono_subj_tokens, mono_subj_token_lengths, _ = self._tokenize_obj(mono_subj_labels)
                
                cs_subj_labels = [instance['subj_label_cross_lang'] for instance in batch]
                all_cs_subj_tokens, cs_subj_token_lengths, _ = self._tokenize_obj(cs_subj_labels)

                # dismiss padding/special token
                for i in range(len(all_cs_subj_tokens)):
                    all_cs_subj_tokens[i] = all_cs_subj_tokens[i][:cs_subj_token_lengths[i]]

                cs_subj_spans = self._find_span(cs_inputs['input_ids'], all_cs_subj_tokens, False)

                mono_masked_rows, mono_masked_cols = self._get_mask_token_positions(mono_inputs)
                cs_masked_rows, cs_masked_cols = self._get_mask_token_positions(cs_inputs)
                

                # Only getting the final output
                if len(selected_layers) == 0 or -1 in selected_layers:
                    mono_ffn_weights, mono_outputs, _ = self.model(**mono_inputs, tgt_layers = intervened_ffn_layers, tgt_pos = mono_masked_cols)
                    _ , cs_outputs, _ = self.model(**cs_inputs, tgt_layers = intervened_ffn_layers, tgt_pos = cs_masked_cols, all_tmp_scores = mono_ffn_weights
                                            , suppression_constant = suppression_constant, subject_tokens_positions = cs_subj_spans)

                    mono_log_probs = torch.log(mono_outputs.softmax(dim=-1)) # batch*seq_length*vocab
                    cs_log_probs = torch.log(cs_outputs.softmax(dim=-1)) # batch*seq_length*vocab

                    self._do_beam_search_with_causal_intervention(mono_masked_rows, mono_masked_cols, cs_masked_rows, cs_masked_cols, mono_log_probs, cs_log_probs
                                         , mono_inputs, cs_inputs, beam_topk, ranking_topk, max_obj_token_len, obj_token_lengths, suppression_constant, cs_subj_spans, intervened_ffn_layers, mono_rank_preds, cs_rank_preds)
                    

                else:
                    mono_ffn_weights, mono_outputs, mono_hidden_states = self.model(**mono_inputs, tgt_pos = mono_masked_cols, tgt_layers = intervened_ffn_layers)
                    _, cs_outputs, cs_hidden_states = self.model(**cs_inputs, tgt_layers = intervened_ffn_layers, tgt_pos = cs_masked_cols, all_tmp_scores = mono_ffn_weights
                                            , suppression_constant = suppression_constant, subject_tokens_positions = cs_subj_spans)

                    for layer in  selected_layers:
                        # check if the layer isnt out of bound
                        assert layer < len(mono_hidden_states)

                        mono_logits = self.model.cls(mono_hidden_states[layer])
                        cs_logits = self.model.cls(cs_hidden_states[layer])
                        
                        mono_log_probs = torch.log(mono_logits.softmax(dim=-1)) # [batch*seq_length*vocab]
                        cs_log_probs = torch.log(cs_logits.softmax(dim=-1))

                        self._do_beam_search_with_causal_intervention(mono_masked_rows, mono_masked_cols, cs_masked_rows, cs_masked_cols
                                                                       , mono_log_probs, cs_log_probs, mono_inputs, cs_inputs, beam_topk
                                                                       , ranking_topk, max_obj_token_len, obj_token_lengths, suppression_constant, cs_subj_spans, intervened_ffn_layers
                                                                       , mono_rank_preds_per_layer[layer], cs_rank_preds_per_layer[layer], layer) 
                        
        if len(selected_layers) == 0 or -1 in selected_layers:
            return mono_rank_preds, cs_rank_preds, labels
        else:
            return mono_rank_preds_per_layer, cs_rank_preds_per_layer, labels

                               
    def inference_per_layer(self, dl, selected_layers):
        self.model.eval()
        source_preds_per_layer = dict()
        target_preds_per_layer = dict()
        with torch.no_grad():
            for batch in tqdm(dl):
                if self.task_type == 'qa':
                    
                    query = [instance[0] for instance in batch]
                    same_lang_ctx = [instance[1][0] for instance in batch]
                    cross_lang_ctx = [instance[1][1] for instance in batch]
                    
                    source_input = self.tokenizer(query, same_lang_ctx, return_tensors='pt', padding=True, truncation=True).to("cuda")
                    target_input = self.tokenizer(query, cross_lang_ctx, return_tensors='pt', padding=True, truncation=True).to("cuda")
                
                    source_outputs = self.model.base_model(**source_input, output_hidden_states=True)
                    target_outputs = self.model.base_model(**target_input, output_hidden_states=True)

                    source_hidden_states = source_outputs.hidden_states[1:]
                    target_hidden_states = target_outputs.hidden_states[1:]
                    
                    source_last_hidden_states = source_outputs.last_hidden_state
                    target_last_hidden_states = target_outputs.last_hidden_state

                    assert torch.equal(source_last_hidden_states, source_hidden_states[-1])
                    assert torch.equal(target_last_hidden_states, target_hidden_states[-1])
                    
                    for layer in selected_layers:
                        assert layer < len(source_hidden_states)
                        source_input_ids = source_input['input_ids']
                        source_logits = self.model.qa_outputs(source_hidden_states[layer])
                        source_start_logits, source_end_logits = source_logits.split(1, dim=-1)
                        source_start_logits = source_start_logits.squeeze(-1).contiguous()
                        source_end_logits = source_end_logits.squeeze(-1).contiguous()
                        source_answer_start_indices = source_start_logits.argmax(dim=-1)
                        source_answer_end_indices = source_end_logits.argmax(dim=-1)
                        for idx, (start_idx, end_idx) in enumerate(zip(source_answer_start_indices, source_answer_end_indices)):
                            pred = source_input_ids[idx, start_idx:end_idx+1]
                            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
                            if layer not in source_preds_per_layer:
                                source_preds_per_layer[layer] = []
                            source_preds_per_layer[layer].append(pred)
                        
                        target_input_ids = target_input['input_ids']
                        target_logits = self.model.qa_outputs(target_hidden_states[layer])
                        target_start_logits, target_end_logits = target_logits.split(1, dim=-1)
                        target_start_logits = target_start_logits.squeeze(-1).contiguous()
                        target_end_logits = target_end_logits.squeeze(-1).contiguous()
                        target_answer_start_indices = target_start_logits.argmax(dim=-1)
                        target_answer_end_indices = target_end_logits.argmax(dim=-1)
                        for idx, (start_idx, end_idx) in enumerate(zip(target_answer_start_indices, target_answer_end_indices)):
                            pred = target_input_ids[idx, start_idx:end_idx+1]
                            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
                            if layer not in target_preds_per_layer:
                                target_preds_per_layer[layer] = []
                            target_preds_per_layer[layer].append(pred)
                            
                
                # nli task
                else:
                    premises = [instance[0] for instance in batch]
                    hypotheses = [instance[1] for instance in batch]
                    source_input = self.tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True).to("cuda")

                    source_outputs = self.model.base_model(**source_input, output_hidden_states=True)
                    source_hidden_states = source_outputs.hidden_states
                    source_last_hidden_state = source_outputs.last_hidden_state


                    for layer in selected_layers:
                        assert layer < len(source_hidden_states)
                        pooled_output = self.model.pooler(source_hidden_states[layer])
                        logits = self.model.classifier(pooled_output)
                        pred = logits.argmax(dim=-1).detach().cpu().numpy()
                        if layer not in source_preds_per_layer:
                            source_preds_per_layer[layer] = []
                        
                        source_preds_per_layer[layer].extend(pred)
                        if layer == 12:
                            curr_pred = pred
                            pooled_output = self.model.pooler(source_last_hidden_state)
                            logits = self.model.classifier(pooled_output)
                            ref_pred = logits.argmax(dim=-1).detach().cpu().numpy() 
                            assert sum(np.equal(curr_pred, ref_pred))==len(curr_pred)
                    
        return source_preds_per_layer, target_preds_per_layer

    
    def inference(self, dl):
        self.model.eval()
        source_preds = []
        target_preds = []
        with torch.no_grad():
            for batch in tqdm(dl):
                if self.task_type == 'qa':
                    query = [instance[0] for instance in batch]
                    same_lang_ctx = [instance[1][0] for instance in batch]
                    cross_lang_ctx = [instance[1][1] for instance in batch]
                    source_input = self.tokenizer(query, same_lang_ctx, return_tensors='pt', padding=True, truncation=True).to("cuda")
                    target_input = self.tokenizer(query, cross_lang_ctx, return_tensors='pt', padding=True, truncation=True).to("cuda")
                    
                    source_outputs = self.model(**source_input)
                    target_outputs = self.model(**target_input)

                    source_answer_start_indices = source_outputs.start_logits.argmax(dim=-1)
                    source_answer_end_indices = source_outputs.end_logits.argmax(dim=-1)

                    target_answer_start_indices = target_outputs.start_logits.argmax(dim=-1)
                    target_answer_end_indices = target_outputs.end_logits.argmax(dim=-1)

                    for idx, (start_idx, end_idx) in enumerate(zip(source_answer_start_indices, source_answer_end_indices)):
                        source_predicted = source_input.input_ids[idx, start_idx:end_idx+1]
                        source_predicted = self.tokenizer.decode(source_predicted, skip_special_tokens=True)
                        source_preds.append(source_predicted)
                    
                    for idx, (start_idx, end_idx) in enumerate(zip(target_answer_start_indices, target_answer_end_indices)):
                        target_predicted = target_input.input_ids[idx, start_idx:end_idx+1]
                        target_predicted = self.tokenizer.decode(target_predicted, skip_special_tokens=True)
                        target_preds.append(target_predicted)
                

                # nli task
                else:
                    premises = [instance[0] for instance in batch]
                    hypotheses = [instance[1] for instance in batch]
                    input = self.tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True).to("cuda")

                    outputs = self.model(**input)
                    logits = outputs.logits
                    predicted = logits.argmax(dim=-1).detach().cpu().numpy()
                    if self.task_type == 'nli_qa':
                        predicted = [int(output) > 0 for output in predicted]
                    source_preds.extend(predicted)


        return source_preds, target_preds
                    


class DecoderLensWrapper:
    def __init__(self, model, tokenizer, source_len=512, target_len=50):
       self.tokenizer = tokenizer
       self.model = model
       self.source_len = source_len
       self.target_len = target_len
    
    def _shift_right(self, input_ids):
        '''
        Shift the label input ids to the right. Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/mt5/modeling_mt5.py#L857-L882 with some adjustments
        '''
        decoder_start_token_id = self.model.config.decoder_start_token_id
        pad_token_id = self.model.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    
    
    def get_encoder_outputs(self, inputs):
       encoder_outputs = self.model.encoder(
           **inputs,
           output_hidden_states=True,
           output_attentions=True,
           return_dict=True
       )
       return encoder_outputs[0], encoder_outputs[1][1:], encoder_outputs[2]

    def _tokenize_obj(self, obj_labels: List[str]) -> [List[str], List[str], List[int], List[int], List[List[int]]]:
        """
        Tokenize all objects which then to be fed into the decoder part

        @param: obj_labels: list of object entities per instance

        @return all_obj_tokens: list of object tokens (w/o the sentinel tokens)
        @return all_labels: list ofobject tokens (w/ the sentinel tokens)
        @return obj_token_lengths: length of each elment in all_obj_tokens
        @return label_token_lengths: length of each elment in all_labels
        @return all_attn_masks: all attention masks for input
        """

        all_obj_tokens = []
        obj_token_lengths = []
        label_token_lengths = []
        all_attn_masks = []
        all_labels = []
        for obj_label in obj_labels:
            obj_tokens = self.tokenizer(f"<extra_id_0> {obj_label} <extra_id_1>")
            attn_mask = obj_tokens['attention_mask']
            obj_tokens = obj_tokens['input_ids']
            label_token_lengths.append(len(obj_tokens))
            all_attn_masks.append(attn_mask)
            label = obj_tokens
            all_labels.append(label)
            obj_tokens = obj_tokens[1:-2]
            obj_token_lengths.append(len(obj_tokens))
            all_obj_tokens.append(obj_tokens)
        
        max_obj_token_len = max(obj_token_lengths)
        max_label_token_len = max(label_token_lengths)

        # add padding
        for i in range(len(all_obj_tokens)):
            num_pad_tokens = max_obj_token_len-obj_token_lengths[i]
            all_obj_tokens[i] += [self.tokenizer.pad_token_id]*num_pad_tokens
            
        for i in range(len(all_labels)):
            num_pad_tokens = max_label_token_len-label_token_lengths[i]
            all_labels[i] += [self.tokenizer.pad_token_id]*num_pad_tokens
            all_attn_masks[i] += [0]*num_pad_tokens

        return all_obj_tokens, all_labels, obj_token_lengths, label_token_lengths, all_attn_masks
        
    def _mask_sentences(self, prompts: List[str]) -> List[str]:
        """
        Replace all default object mask tokens with one sentinel token

        @param prompts: A list of prompts

        @return new_prompts: A list of modified prompts having the sentinel token
        """

        new_prompts = []
        for prompt in prompts:
            new_mask = "<extra_id_0>"
            new_prompt = prompt.replace('[Y]', new_mask)
            new_prompts.append(new_prompt)
        return new_prompts
    
    def measure_encoder_representations_cosine_similarity(self, og_sentences, possible_cs_sentences_candidates, selected_layers=[]):
        self.model.eval()
        cos_sims_per_layer = dict()
        encoder_model = self.model.encoder
        for og_sentence, possible_cs_sentence_candidates in tzip(og_sentences, possible_cs_sentences_candidates):
            # single input:
            if isinstance(og_sentence, str):
                batched_input = [og_sentence]+possible_cs_sentence_candidates
                model_inputs = self.tokenizer(batched_input, return_tensors='pt', padding=True, truncation=True).to('cuda')

            # double input
            else:
                batched_input1 = [og_sentence[0]]
                batched_input1 += [cand[0] for cand in possible_cs_sentence_candidates]
                batched_input2 = [og_sentence[1]]
                batched_input2 += [cand[1] for cand in possible_cs_sentence_candidates]
                model_inputs = self.tokenizer(batched_input1, batched_input2, return_tensors='pt', padding=True, truncation=True).to('cuda')
            with torch.no_grad():
                outputs = encoder_model(**model_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
            for layer in selected_layers:
                assert layer < len(hidden_states)
                if layer not in cos_sims_per_layer:
                    cos_sims_per_layer[layer] = []
                current_hidden_states = hidden_states[layer].detach().cpu().numpy().mean(axis=1, keepdims=False)

                og_representation = current_hidden_states[0]
                og_representation = og_representation[np.newaxis,...]
                cs_representations = current_hidden_states[1:]
                
                og_norm = np.linalg.norm(og_representation, axis=1)
                cs_norm = np.linalg.norm(cs_representations, axis=1)

                sim_matrix = np.dot(og_representation, cs_representations.T)
                
                cos_sim = sim_matrix/(np.outer(og_norm, cs_norm) + 1e-9)
                cos_sim = cos_sim.mean(axis=-1, keepdims=False)
                cos_sims_per_layer[layer].append(cos_sim[0])
        for layer in cos_sims_per_layer.keys():
            cos_sims_per_layer[layer] = sum(cos_sims_per_layer[layer])/len(cos_sims_per_layer[layer])
        return cos_sims_per_layer

    def _find_span(self, input_ids_batch, target_word_input_ids_batch, pick_last=False):
        """"
        Find span range of specified object entities

        @param input_ids_batch: List of token ids in the input [batch, seq_length]
        @param target_word_input_ids_batch: List of token ids in the specified object eneities [batch, 1-max_obj_entity_length]
        @param pick_last: whether to pick last possible span or not

        @return span_positions_per_batch: Span positions in the input token ids of specified object entities [batch, 1-max_obj_entity_length]     
        """
        span_positions_per_batch = []
        for input_ids, target_word_input_ids in zip(input_ids_batch, target_word_input_ids_batch):
            input_ids_lst = [int(token_id) for token_id in input_ids]
            start_pos = input_ids_lst.index(target_word_input_ids[0])
            possible_spans = []

            for left_bound in range(start_pos, len(input_ids)):
                for right_bound in range(len(input_ids)-1, left_bound-1, -1):
                    input_ids_subset = input_ids[left_bound:right_bound+1]
                    if len(input_ids_subset) != len(target_word_input_ids):
                        continue
                    match_values = [input_id_token==target_input_id_token for input_id_token, target_input_id_token in zip(input_ids_subset, target_word_input_ids)]
                    if sum(match_values)==len(match_values):
                        possible_spans.append([i for i in range(left_bound,right_bound+1)])
            
            assert len(possible_spans) > 0
                
            if pick_last:
                span_positions_per_batch.append(possible_spans[-1])
            else:
                span_positions_per_batch.append(possible_spans[0])
        return span_positions_per_batch
    
    def _calculate_subj_obj_attention_per_instance(self, tokens_attention: np.array, attention_weights_per_layer: Dict[int, Dict[int, List[float]]]
                                                   , layer: int, obj_span_indices: List[int], subj_span_indices: List[int]):
        """
        Calculates the average of sum of attention weights on all subject tokens attending particular object token across all object tokens

        @param tokens_attention: all attention weights on every head of particular encoder layer for one input
            [head, seq_length, seq_length]
        @param attention_weights_per_layer: average subject-object attention weigthts on every head of one encoder layer for all inputs
        @param layer: layer position
        @param obj_span_indices: indices of object entity
        @param subj_span_indices: indices of subject entity
        """
        
        span_attention = tokens_attention[:, obj_span_indices, :][:, :, subj_span_indices] # extract subject tokens attention of each object token z
        span_attention_weight_average_per_head = span_attention.sum(axis=-1, keepdims=False).mean(axis=1, keepdims=False) # sum over the subject tokens get the average over all object tokens
        for head_pos, attention_weight_avg in enumerate(span_attention_weight_average_per_head):
            if head_pos not in attention_weights_per_layer[layer]:
                attention_weights_per_layer[layer][head_pos] = []
            attention_weights_per_layer[layer][head_pos].append(attention_weight_avg)

    def extract_attention_scores_subj_obj(self, instances: List[Dict], batch_size: int = 16, selected_layers: List[int] = []):
        """
        Calculates the average of sum of attention weights on all subject tokens attending particular object token across all object tokens for one input

        @param instances: all input instances containing dictionary
        @param batch_size: number of instances in one batch
        @param selected_layers: encoder layer indices that we want to analyze

        @return mono_attention_weights_per_layer: average subject-object attention weights for each head and layer 
            given all monolingual inputs
        @return cs_attention_weights_per_layer: average subject-object attention weights for each head and layer 
            given all codemixed inputs
        """

        self.model.eval()
        encoder_model = self.model.encoder
        batch_cnt = len(instances)//batch_size
        
        for i in tqdm(range(0, batch_cnt), desc='batch'):

            batch = instances[i*batch_size:min((i+1)*batch_size, len(instances))]
            
            obj_labels = [instance['obj_label'].strip() for instance in batch]
            
            mono_subj_labels = [instance['subj_label_same_lang'].strip() for instance in batch]
            cs_subj_labels = [instance['subj_label_cross_lang'].strip() for instance in batch]

            # create parallel code-switching statements
            mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']).replace('[Y]', instance['obj_label']) for instance in batch]
            cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']).replace('[Y]', instance['obj_label']) for instance in batch]

            all_mono_subj_tokens, _, mono_subj_token_lengths, _, _ = self._tokenize_obj(mono_subj_labels)
            for i in range(len(all_mono_subj_tokens)):
                all_mono_subj_tokens[i] = all_mono_subj_tokens[i][:mono_subj_token_lengths[i]]
            
            all_cs_subj_tokens, _, cs_subj_token_lengths, _, _  = self._tokenize_obj(cs_subj_labels)
            for i in range(len(all_cs_subj_tokens)):
                all_cs_subj_tokens[i] = all_cs_subj_tokens[i][:cs_subj_token_lengths[i]]
            
            all_obj_tokens, _, obj_token_lengths, _, _ = self._tokenize_obj(obj_labels)
            for i in range(len(all_obj_tokens)):
                all_obj_tokens[i] = all_obj_tokens[i][:obj_token_lengths[i]]
            
            mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt')
            cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt')

            
            mono_inputs_new = dict()
            for key, val in mono_inputs.items():
                val = torch.Tensor([val[idx] for idx in mono_input_space_indices]).to(val.device)
                mono_inputs_new[key] = val

            cs_inputs_new = dict()
            for key, val in cs_inputs.items():
                val = torch.Tensor([val[idx] for idx in cs_input_space_indices]).to(val.device)
                cs_inputs_new[key] = val
            
            mono_inputs = mono_inputs_new
            cs_inputs = cs_inputs_new


            # find subject entity span positions
            mono_subj_spans = self._find_span(mono_inputs['input_ids'], all_mono_subj_tokens, False)
            cs_subj_spans = self._find_span(cs_inputs['input_ids'], all_cs_subj_tokens, False)

            # find object entity span positions
            mono_obj_spans = self._find_span(mono_inputs['input_ids'], all_obj_tokens, True)
            cs_obj_spans = self._find_span(cs_inputs['input_ids'], all_obj_tokens, True)

            mono_inputs = mono_inputs.to('cuda')
            cs_inputs = mono_inputs.to('cuda')

            mono_attentions = encoder_model(**mono_inputs, output_attentions=True).attentions
            cs_attentions = encoder_model(**cs_inputs, output_attentions=True).attentions

            mono_attention_weights_per_layer = dict()
            cs_attention_weights_per_layer = dict()

            for layer in selected_layers:
                if layer not in mono_attention_weights_per_layer:
                    mono_attention_weights_per_layer[layer] = dict()
                if layer not in cs_attention_weights_per_layer:
                    cs_attention_weights_per_layer[layer] = dict()
                assert layer < len(mono_attentions)
                mono_attentions_spec_layer = mono_attentions[layer].detach().cpu().numpy()
                cs_attention_spec_layer = cs_attentions[layer].detach().cpu().numpy()
                for batch_idx in range(len(batch)):
                    # calculate subject-object attention for monolingual input
                    mono_tokens_attention = mono_attentions_spec_layer[batch_idx]
                    mono_start_obj_idx = mono_obj_spans[batch_idx][0]
                    mono_end_obj_idx = mono_obj_spans[batch_idx][-1]
                    mono_start_subj_idx = mono_subj_spans[batch_idx][0] 
                    mono_end_subj_idx = mono_subj_spans[batch_idx][-1]
                    self._calculate_subj_obj_attention_per_instance(mono_tokens_attention, mono_attention_weights_per_layer, layer
                                                                    , mono_start_obj_idx, mono_end_obj_idx, mono_start_subj_idx, mono_end_subj_idx)
                    
                    # calculate subject-object attention for codemixed input
                    cs_tokens_attention = cs_attention_spec_layer[batch_idx]
                    cs_start_obj_idx = cs_obj_spans[batch_idx][0]
                    cs_end_obj_idx = cs_obj_spans[batch_idx][-1]
                    cs_start_subj_idx = cs_subj_spans[batch_idx][0] 
                    cs_end_subj_idx = cs_subj_spans[batch_idx][-1]
                    self._calculate_subj_obj_attention_per_instance(cs_tokens_attention, cs_attention_weights_per_layer, layer
                                                                    , cs_start_obj_idx, cs_end_obj_idx, cs_start_subj_idx, cs_end_subj_idx)
          
        # accumulate all attentions for every layer
        for layer in selected_layers:
            for head_pos in mono_attention_weights_per_layer[layer].keys():
                mono_attention_weights_per_layer[layer][head_pos] = sum(mono_attention_weights_per_layer[layer][head_pos])/len(mono_attention_weights_per_layer[layer][head_pos])

            for head_pos in cs_attention_weights_per_layer[layer].keys():
                cs_attention_weights_per_layer[layer][head_pos] = sum(cs_attention_weights_per_layer[layer][head_pos])/len(cs_attention_weights_per_layer[layer][head_pos])     
        return mono_attention_weights_per_layer, cs_attention_weights_per_layer            

    def _get_necessary_mask_positions(self, outputs: torch.Tensor, label_token_lengths: List[int], obj_token_lengths: List[int]):
        """
        get necessary position indices for several mask tokens
        @param outputs: predicted token output logits [batch_size*seq_length*vocab_size]
        @param label_token_lengths: length of each object entity (including the sentinel tokens)
        @param obj_token_lengths: length of each object entity (excluding the sentinel tokens)

        @return masked_indices: span indices
        @return masked_start_idx: first predicted object token's position
        @return masked_current_idx: current predicted object token's position
        @return masked_batch_idx: batch indices
        """
        masked_indices = []
        masked_start_idx = []
        masked_current_idx = []
        masked_batch_indices = []

        for idx, output in enumerate(outputs):
            masked_indices.append(list(range(output.size()[1]))[1:-2-(output.size()[1]-label_token_lengths[idx])]) #extract object in-between the sentinel tokens
            masked_start_idx.append(min(masked_indices[idx]))
            masked_current_idx.append(min(masked_indices[idx]))
            masked_batch_indices.append(idx)
            assert len(masked_indices[idx]) == obj_token_lengths[idx]
        
        return masked_indices, masked_start_idx, masked_current_idx, masked_batch_indices

    def _calculate_joint_proba(self, log_probs: torch.Tensor, span_pos_rows: List[List[int]], span_pos_cols: List[List[int]]
                               , prev_proba: torch.Tensor) -> torch.Tensor:
        """
        Calculate joint proba of current topk-frontier with all words in the vocabulary

        @param log_probs: log probabilites tensor for one batch [batch_size, seq_length, vocab]
        @param span_pos_rows: batch indices that we want to predict the next token [(1...batch_size), 1]
        @param span_pos_cols: position of previous mask tokens that have been predicted [(1...batch_size), span_length]
        @param prev_proba: probabilites of ngram_candiate [batch_size, 1]

        @return joint_proba: joint probabilites between top-k and tokens in vocabulary [batch_size, 1, 1]
        """
        cand_proba = log_probs[span_pos_rows, span_pos_cols, :].squeeze(1) # [batch*vocab]
        joint_proba = prev_proba + cand_proba # batch*vocab
        joint_proba = joint_proba.unsqueeze(1) 
        return joint_proba
    

    def _get_next_topk_proba_and_words(self, joint_proba: torch.Tensor, beam_topk: int):
        """
        Determine the proba and token of next top-k candidates based on the joint probability

        @param joint_proba: joint probability [batch*k*vocab]
        @param beam_topk: number of frontiers to select
        
        @return next_topk_log_prob: the joint probabilities of the next top-k candidates # [batch*k]
        @return vocab_indices: the chosen token of the next top-k candidates # [batch*k] 
        @return prefix_indices: the chosen n-gram prefix of the next top-k candidates # [batch*k] 
        """

        vocab_size = joint_proba.shape[-1]
        joint_proba = joint_proba.view(joint_proba.shape[0], -1) # [batch*(k*vocab)
        next_topk_log_prob, next_topk_indices = joint_proba.topk(beam_topk, sorted=True) # [batch*k], [batch*k]
        prefix_indices, vocab_indices = next_topk_indices//vocab_size, next_topk_indices%vocab_size # indices of the prefix candidates, indices of the vocab
        prefix_indices = prefix_indices.cpu()
        return next_topk_log_prob, vocab_indices, prefix_indices


    def _update_topk_ngrams(self, new_indices: np.array, prefix_ngrams: np.array, next_token: np.array, batch_idx: int):
        """
        Update the topk frontier

        @param new_indices: topk tokens that we want to update [batch*k*span_length]
        @param prefix_ngrams: chosen prefix for topk [k*span_length-1]
        @param next_token: next token for top-k [k*1]
        @param batch_idx: position of instance within batch
        """
        # for each instance, update the top-k frontier n-grams
        new_indices[batch_idx, :, :-1] = prefix_ngrams
        new_indices[batch_idx, :, -1] = next_token

    def _add_topk_indices_log_prob_to_batch(self, batch_rank_preds: List[Dict[str, float]]
                                            , new_indices: np.array, next_topk_log_prob: np.array, batch_idx: int, span_length: int):
        """
        Add current top-k log probabilites and ngram into the list of predictions for current batch

        @param batch_rank_preds: all top-k ngrams along with their probabilites
        @param new_indices: ngram of topk [batch_size, k, span_length]
        @param next_topk_log_prob: probabilites of each topk [batch_size, k, 1]
        @param span_length: span length for current iteration
        """
        topk_indices_instance = new_indices[batch_idx]
        topk_log_prob_instance = next_topk_log_prob[batch_idx]
        for curr_topk_log_prob, curr_topk_token_idx in zip(topk_log_prob_instance, topk_indices_instance):
            vocab_ids = curr_topk_token_idx.type(torch.LongTensor)
            decoded_word = self.tokenizer.batch_decode(vocab_ids)
            decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)]) # add id besides the token to distinghuish token
            batch_rank_preds[batch_idx][decoded_word] = curr_topk_log_prob.item()/(span_length+1) # normalize with length

    def _beam_search(self, mono_outputs: torch.Tensor, cs_outputs: torch.Tensor
                     , label_token_lengths: List[int], obj_token_lengths: List[int]
                     , batch_sz: int, beam_topk: int, ranking_topk: int
                     , mono_rank_preds: List[List[str]], cs_rank_preds: List[List[str]]): 
        """"
        Do beam search for one batch

        @param mono_outputs: output logits from the monolingual input [batch_size*seq_length*vocab]
        @param cs_outputs: output logits from the codemixed input [batch_size*seq_length*vocab]
        @param label_token_lengths: length of each object entity (including the sentinel tokens)
        @param obj_token_lengths: length of each object entity (excluding the sentinel tokens)
        @param batch_sz: batch size
        @param beam_topk: beam search width
        @param ranking_topk: number of candidates selected for final prediction
        @param mono_rank_preds: final top-k predictions for all monolingual inputs
        @param cs_rank_preds: final top-k predictions for all codemixed inputs
        """

        # get mask token position indices
        mono_masked_indices, mono_masked_start_idx, mono_masked_current_idx, mono_masked_batch_indices = self._get_necessary_mask_positions(mono_outputs, label_token_lengths, obj_token_lengths)
        cs_masked_indices, cs_masked_start_idx, cs_masked_current_idx, cs_masked_batch_indices = self._get_necessary_mask_positions(cs_outputs, label_token_lengths, obj_token_lengths)
        
        max_obj_token_len = max(obj_token_lengths)

        mono_log_probs = torch.log(mono_outputs.softmax(dim=-1)) # [batch_size*seq_length*vocab]
        cs_log_probs = torch.log(cs_outputs.softmax(dim=-1)) # [batch_size*seq_length*vocab]

        mono_masked_rows = np.array(mono_masked_batch_indices)[..., np.newaxis]
        mono_masked_current_cols = np.array(mono_masked_start_idx)[..., np.newaxis]

        cs_masked_rows = np.array(cs_masked_batch_indices)[..., np.newaxis]
        cs_masked_current_cols = np.array(cs_masked_current_idx)[..., np.newaxis]

        mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # [batch_size*seq_length*k], [batch_size*seq_length*k] 
        cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True) # [batch_size*seq_length*k], [batch_size*seq_length*k] 

        mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_masked_current_cols, :], mono_topk_indices[mono_masked_rows, mono_masked_current_cols, :] # [bach*1*k], [bach*1*k] 
        cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_masked_current_cols, :], cs_topk_indices[cs_masked_rows, cs_masked_current_cols, :] # [bach*1*k], [bach*1*k]

        mono_batch_rank_preds = []
        cs_batch_rank_preds = []
        for _ in range(batch_sz):
            mono_dict = dict()
            cs_dict = dict()
            mono_batch_rank_preds.append(mono_dict)
            cs_batch_rank_preds.append(cs_dict)

        for batch_idx in range(batch_sz):
            mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0] # [k], [k]
            cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0] # [k], [k]
            
            # collect all predictions for first object token
            for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                decoded_word = f"{decoded_word}<{curr_mono_topk_token_idx}>"
                mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()
            
            for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                decoded_word = f"{decoded_word}<{curr_cs_topk_token_idx}>"
                cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob
        
        mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1)) # [k*batch*1]
        cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1)) # [k*batch*1]

        mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor)
        cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor)       

        
        for span_len in range(1, max_obj_token_len):
            # set span indices
            for batch_idx in range(batch_sz):
                if obj_token_lengths[batch_idx] >= (span_len+1): # this means that this instance still has some object token to predict
                    mono_masked_current_cols[batch_idx][0] += 1
                    cs_masked_current_cols[batch_idx][0] += 1


            all_mono_joint_proba = []
            all_cs_joint_proba = []
            for cand_rank in range(len(mono_topk_log_prob)): 
                mono_joint_proba = self._calculate_joint_proba(mono_log_probs, mono_masked_rows, mono_masked_current_cols, mono_topk_log_prob[cand_rank])   
                all_mono_joint_proba.append(mono_joint_proba)

                cs_joint_proba = self._calculate_joint_proba(cs_log_probs, cs_masked_rows, cs_masked_current_cols, cs_topk_log_prob[cand_rank])
                all_cs_joint_proba.append(cs_joint_proba)
            
            all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # batch*k*vocab
            all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # batch*k*vocab

            next_mono_topk_log_prob, vocab_indices_mono, prefix_indices_mono = self._get_next_topk_proba_and_words(
                all_mono_joint_proba, beam_topk
            )
            next_cs_topk_log_prob, vocab_indices_cs, prefix_indices_cs = self._get_next_topk_proba_and_words(
                all_mono_joint_proba, beam_topk
            )

            new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
            new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1))

            mono_topk_indices = mono_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
            cs_topk_indices = cs_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
            
            for batch_idx in range(batch_sz):
                self._update_topk_ngrams(new_mono_indices, mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]], vocab_indices_mono[batch_idx], batch_idx)
                self._update_topk_ngrams(new_cs_indices, cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]], vocab_indices_cs[batch_idx], batch_idx)

                if obj_token_lengths[batch_idx] >= (span_len+1):
                    self._add_topk_indices_log_prob_to_batch(mono_batch_rank_preds, new_mono_indices, next_mono_topk_log_prob, batch_idx, span_len)
                    self._add_topk_indices_log_prob_to_batch(cs_batch_rank_preds, new_cs_indices, next_cs_topk_log_prob, batch_idx, span_len)

            cs_topk_indices = new_cs_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len
            mono_topk_indices = new_mono_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len

            mono_topk_log_prob = next_mono_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
            cs_topk_log_prob = next_cs_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
        
        # rank all preds
        for batch_preds in mono_batch_rank_preds:
            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
            selected_words = sorted_batch_preds[:ranking_topk]
            mono_rank_preds.append(selected_words)
        
        for batch_preds in cs_batch_rank_preds:
            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
            selected_words = sorted_batch_preds[:ranking_topk]
            cs_rank_preds.append(selected_words)

    def inference_cloze_task(self, instances: List[Dict], batch_size: int = 16, selected_layers: List[int] = [], beam_topk: int = 5, ranking_topk: int = 5)  -> [Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], List[str]]:
        """
        Do inference for fill-in-the-blank task using beam search

        @param instances: all instances
        @param batch_size: number of instances in a batch
        @param selected_layers: layer indices that we want to analyze
        @param beam_topk: beam search width
        @param ranking_topk: number of candidates selected for final prediction

        @return mono_rank_preds: list of top-k predictions on all monolingual inputs just from last layer 
        @return cs_rank_preds: list of top-k predictions on all codemixed inputs just from last layer
        @return mono_rank_preds_per_layer: list of top-k predictions on all monolingual inputs from selected layers
        @return cs_rank_preds_per_layer: list of top-k predictions on all codemixed inputs just from selected layers
        @return labels: list of ground truth labels
        """

        self.model.eval()
        mono_rank_preds, cs_rank_preds = [], []
        mono_rank_preds_per_layer, cs_rank_preds_per_layer = dict(), dict()
        labels = []
        batch_cnt = len(instances)//batch_size

        for layer in selected_layers:
            mono_rank_preds_per_layer[layer] = []
            cs_rank_preds_per_layer[layer] = []

        with torch.no_grad():
            for i in tqdm(range(0, batch_cnt)):
                batch = instances[i*batch_size:min(len(instances), (i+1)*batch_size)]
                obj_labels = [instance['obj_label'] for instance in batch]
                labels.extend(obj_labels)
                mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']) for instance in batch]
                cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']) for instance in batch]
                _, all_label_tokens, obj_token_lengths, label_token_lengths, all_attn_masks = self._tokenize_obj(obj_labels)
                
                # replace default mask token with single sentinel token
                mono_prompts = self._mask_sentences(mono_prompts)
                cs_prompts = self._mask_sentences(cs_prompts)
                
                mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
                cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')


               
                all_label_tokens = torch.Tensor(all_label_tokens).to('cuda').long()
                all_attn_masks = torch.Tensor(all_attn_masks).to('cuda').long()
                
                mono_input_ids = mono_inputs['input_ids']
                cs_input_ids = cs_inputs['input_ids']
                
                batch_sz = len(batch)

                # only obtains the prediction from last layer
                if len(selected_layers) == 0 or -1 in selected_layers:
                    mono_outputs = self.model(input_ids=mono_input_ids, labels=all_label_tokens)
                    mono_outputs = mono_outputs.logits # [batch_size*seq_length*vocab]
                    
                    cs_outputs = self.model(input_ids=cs_input_ids, labels=all_label_tokens)
                    cs_outputs = cs_outputs.logits # [batch_size*seq_length*vocab]
                    
                    self._beam_search(mono_outputs, cs_outputs, label_token_lengths, obj_token_lengths,
                                      batch_sz, beam_topk, ranking_topk, mono_rank_preds, cs_rank_preds)
                    
                else:
                    _, mono_enc_hidden_states, _ = self.get_encoder_outputs(mono_inputs)
                    _, cs_enc_hidden_states, _ = self.get_encoder_outputs(cs_inputs)
                    decoder_input_ids = self._shift_right(all_label_tokens)
                    for layer in selected_layers:
                        assert layer < len(mono_enc_hidden_states)
                        
                        if layer not in mono_rank_preds_per_layer:
                            mono_rank_preds_per_layer[layer] = []
                        
                        if layer not in cs_rank_preds_per_layer:
                            cs_rank_preds_per_layer[layer] = []

                        mono_outputs = self.model.decoder(
                            input_ids=decoder_input_ids,
                            attention_mask=all_attn_masks,
                            encoder_hidden_states=mono_enc_hidden_states[layer],
                            encoder_attention_mask=mono_inputs['attention_mask']
                        )[0]
                        cs_outputs = self.model.decoder(
                            input_ids=decoder_input_ids,
                            attention_mask=all_attn_masks,
                            encoder_hidden_states=cs_enc_hidden_states[layer],
                            encoder_attention_mask=cs_inputs['attention_mask']
                        )[0]

                        mono_outputs = self.model.lm_head(mono_outputs)
                        cs_outputs = self.model.lm_head(cs_outputs)

                        self._beam_search(mono_outputs, cs_outputs, label_token_lengths, obj_token_lengths, batch_sz, beam_topk, ranking_topk, mono_rank_preds_per_layer[layer], cs_rank_preds_per_layer[layer])
                          
        if len(selected_layers) == 0 or -1 in selected_layers:
            return mono_rank_preds, cs_rank_preds, labels
       
        else:
            return mono_rank_preds_per_layer, cs_rank_preds_per_layer, labels

    
    def inference_cloze_task_with_causal_intervention(self, instances: List[Dict], batch_size: int = 16
                                                      , selected_layers: List[int] = [], beam_topk: int = 5
                                                      , ranking_topk: int = 5, intervened_layers: List[int] = None, suppression_constant: float = None)  -> [Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], Union[List[Dict[str, float]], Dict[int, List[Dict[str, float]]]], List[str]]:
        """
        Do inference for fill-in-the-blank task using beam search that includes causal intervention

        @param instances: all instances
        @param batch_size: number of instances in a batch
        @param selected_layers: layer indices that we want to analyze
        @param beam_topk: beam search width
        @param ranking_topk: number of candidates selected for final prediction
        @param intervened_layers: which FFN layers we want to manipulate
        @param suppression_constant: constant to suppress the subject-object activation values
        
        @return mono_rank_preds: list of top-k predictions on all monolingual inputs just from last layer 
        @return cs_rank_preds: list of top-k predictions on all codemixed inputs just from last layer
        @return mono_rank_preds_per_layer: list of top-k predictions on all monolingual inputs from selected layers
        @return cs_rank_preds_per_layer: list of top-k predictions on all codemixed inputs just from selected layers
        @return labels: list of ground truth labels
        """

        self.model.eval()
        mono_rank_preds, cs_rank_preds = [], []
        mono_rank_preds_per_layer, cs_rank_preds_per_layer = dict(), dict()
        labels = []
        batch_cnt = len(instances)//batch_size

        for layer in selected_layers:
            mono_rank_preds_per_layer[layer] = []
            cs_rank_preds_per_layer[layer] = []

        with torch.no_grad():
            for i in tqdm(range(0, batch_cnt)):
                batch = instances[i*batch_size:min(len(instances), (i+1)*batch_size)]
                obj_labels = [instance['obj_label'] for instance in batch]
                cs_subj_labels = [instance['subj_label_cross_lang'] for instance in batch] 

                labels.extend(obj_labels)
                mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']) for instance in batch]
                cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']) for instance in batch]
                _, all_label_tokens, obj_token_lengths, label_token_lengths, all_attn_masks = self._tokenize_obj(obj_labels)
                
                # replace default mask token with single sentinel token
                mono_prompts = self._mask_sentences(mono_prompts)
                cs_prompts = self._mask_sentences(cs_prompts)
                
                mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
                cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')

                all_cs_subj_tokens, _, cs_subj_token_lengths, _, _  = self._tokenize_obj(cs_subj_labels)
                for i in range(len(all_cs_subj_tokens)):
                    all_cs_subj_tokens[i] = all_cs_subj_tokens[i][:cs_subj_token_lengths[i]]

                all_label_tokens = torch.Tensor(all_label_tokens).to('cuda').long()
                all_attn_masks = torch.Tensor(all_attn_masks).to('cuda').long()
                
                mono_input_ids = mono_inputs['input_ids']
                cs_input_ids = cs_inputs['input_ids']

                cs_subj_spans = self._find_span(cs_input_ids, all_cs_subj_tokens, False)

                mono_inputs_cpu = self.tokenizer(mono_prompts, padding=True, truncation=True)
                cs_inputs_cpu = self.tokenizer(cs_prompts, padding=True, truncation=True)

                # gather all first sentinel token (used for causal intervention)
                mono_input_ids_cpu = mono_inputs_cpu['input_ids']
                mono_encoder_mask_positions = []
                encoder_sentinel_token_id = 250099 
                for batch_idx in range(len(mono_input_ids_cpu)):
                    mask_position = mono_input_ids_cpu[batch_idx].index(encoder_sentinel_token_id)
                    mono_encoder_mask_positions.append(mask_position) 
                cs_input_ids_cpu = cs_inputs_cpu['input_ids']
                cs_encoder_mask_positions = []
                for batch_idx in range(len(cs_input_ids_cpu)):
                    mask_position = cs_input_ids_cpu[batch_idx].index(encoder_sentinel_token_id)
                    cs_encoder_mask_positions.append(mask_position) 
                
                batch_sz = len(batch)

                # only obtains the prediction from last layer
                if len(selected_layers) == 0 or -1 in selected_layers:
                    mono_outputs, mono_ffn_states, _ = self.model(input_ids=mono_input_ids, labels=all_label_tokens, encoder_tgt_pos=mono_encoder_mask_positions
                                                                                          , tgt_layers=intervened_layers)
                                                                                     
                    
                    cs_outputs, _, _ = self.model(input_ids=cs_input_ids, labels=all_label_tokens,
                                                  tgt_layers=intervened_layers, suppression_constant=suppression_constant, encoder_tgt_pos=cs_encoder_mask_positions
                                                  , subject_tokens_positions=cs_subj_spans, all_modified_activation_values=mono_ffn_states)
                    
                    self._beam_search(mono_outputs, cs_outputs, label_token_lengths, obj_token_lengths,
                                      batch_sz, beam_topk, ranking_topk, mono_rank_preds, cs_rank_preds)
                    
                else:
                    _, mono_ffn_states, mono_enc_hidden_states = self.model(input_ids=mono_input_ids, labels=all_label_tokens
                                                                        , tgt_layers=intervened_layers, encoder_tgt_pos=mono_encoder_mask_positions) 
                    _, _, cs_enc_hidden_states = self.model(input_ids=cs_input_ids, labels=all_label_tokens,
                                tgt_layers=intervened_layers, suppression_constant=suppression_constant
                                , subject_tokens_positions=cs_subj_spans, all_modified_activation_values=mono_ffn_states, encoder_tgt_pos=cs_encoder_mask_positions)
                    decoder_input_ids = self._shift_right(all_label_tokens)
                    for layer in selected_layers:
                        assert layer < len(mono_enc_hidden_states)
                        
                        if layer not in mono_rank_preds_per_layer:
                            mono_rank_preds_per_layer[layer] = []
                        
                        if layer not in cs_rank_preds_per_layer:
                            cs_rank_preds_per_layer[layer] = []

                        mono_outputs = self.model.decoder(
                            input_ids=decoder_input_ids,
                            attention_mask=all_attn_masks,
                            encoder_hidden_states=mono_enc_hidden_states[layer],
                            encoder_attention_mask=mono_inputs['attention_mask']
                        )[0]
                        cs_outputs = self.model.decoder(
                            input_ids=decoder_input_ids,
                            attention_mask=all_attn_masks,
                            encoder_hidden_states=cs_enc_hidden_states[layer],
                            encoder_attention_mask=cs_inputs['attention_mask']
                        )[0]

                        mono_outputs = self.model.lm_head(mono_outputs)
                        cs_outputs = self.model.lm_head(cs_outputs)

                        self._beam_search(mono_outputs, cs_outputs, label_token_lengths, obj_token_lengths, batch_sz, beam_topk, ranking_topk, mono_rank_preds_per_layer[layer], cs_rank_preds_per_layer[layer])
                          
        if len(selected_layers) == 0 or -1 in selected_layers:
            return mono_rank_preds, cs_rank_preds, labels
       
        else:
            return mono_rank_preds_per_layer, cs_rank_preds_per_layer, labels  
    def classify(self, premise_hyphotesis_pairs, all_choices, is_binary):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for pair in tqdm(premise_hyphotesis_pairs):
                labels.append(pair[2])
                text_input = f"{pair[0]}\nQuestion: {pair[1]} {all_choices[0]}, {all_choices[-1]}, or {all_choices[1]}?"
                source_input = self.tokenizer(text_input, return_tensors="pt", padding=False).to("cuda")
                min_loss = float('inf')
                pred = -1
                for idx, candidate in enumerate(all_choices):
                    target_input = self.tokenizer(candidate, return_tensors="pt", padding=False).to("cuda")
                    source_input_ids = source_input['input_ids']
                    attention_mask = source_input['attention_mask']
                    target_input_ids = target_input['input_ids']
                    outputs = self.model(input_ids = source_input_ids, attention_mask = attention_mask, labels=target_input_ids)
                    loss = outputs[0].item()
                    if loss < min_loss:
                        min_loss = loss
                        pred = idx
                if is_binary:
                    pred = pred != all_choices[0]
                preds.append(pred)
        return preds, labels    

    def classify_on_particular_hidden_states(self, premise_hyphotesis_pairs, all_choices, is_binary, selected_layers):
        self.model.eval()
        layerwise_preds = dict()
        with torch.no_grad():
            for pair in tqdm(premise_hyphotesis_pairs):
                text_input = f"{pair[0]}\nQuestion: {pair[1]} {all_choices[0]}, {all_choices[-1]}, or {all_choices[1]}?"
                source_input = self.tokenizer(text_input, return_tensors="pt", padding=False).to("cuda")
                _, enc_hidden_states, _ = self.get_encoder_outputs(source_input) # per batch
                min_loss = float('inf')
                pred = -1
                for layer in selected_layers:
                    assert layer < len(enc_hidden_states)
                    for idx, candidate in enumerate(all_choices):
                        target_input = self.tokenizer(candidate, return_tensors="pt", padding=False).to("cuda")
                        target_input_ids = target_input['input_ids']
                        target_attn_mask = target_input['attention_mask']
                        decoder_input_ids = self._shift_right(target_input_ids)
                        decoder_outputs = self.model.decoder(
                                input_ids=decoder_input_ids,
                                attention_mask=target_attn_mask,
                                encoder_hidden_states=enc_hidden_states[layer],
                                encoder_attention_mask=source_input['attention_mask']
                            )
                        sequence_output = decoder_outputs[0]
                        lm_logits = self.model.lm_head(sequence_output) # batch_size x seq_length x voc_size
                        loss = None
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        target_input_ids = target_input_ids.to(lm_logits.device)
                        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), target_input_ids.view(-1)).item()      
                        if loss < min_loss:
                            min_loss = loss
                            pred = idx
                    if is_binary:
                        pred = pred != all_choices[0]
                    if layer not in layerwise_preds:
                        layerwise_preds[layer] = []
                    layerwise_preds[layer].append(pred)

        return layerwise_preds
      

    def decode(self, queries_dl):
        source_preds = []
        target_preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(queries_dl):
                source_queries = [instance[0] for instance in batch]
                target_queries = [instance[1] for instance in batch]

                source_inputs = self.tokenizer(source_queries, return_tensors="pt", padding=True, max_length=self.source_len).to("cuda")
                target_inputs = self.tokenizer(target_queries, return_tensors="pt", padding=True, max_length=self.target_len).to("cuda")


                source_generated_ids, target_generated_ids = self.model.generate(**source_inputs), self.model.generate(**target_inputs)

                source_pred = self.tokenizer.batch_decode(source_generated_ids, skip_special_tokens=True)
                target_pred = self.tokenizer.batch_decode(target_generated_ids, skip_special_tokens=True)
                source_pred = [instance.replace("Answer:", "").strip() for instance in source_pred]
                target_pred = [instance.replace("Answer:", "").strip() for instance in target_pred]
                source_preds += source_pred
                target_preds += target_pred
        return source_preds, target_preds

    def get_encoder_representation(self, queries_dl, selected_layers):
        source_hidden_states_per_layer = dict()
        target_hidden_states_per_layer = dict()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(queries_dl):
                source_batch = [instance[0] for instance in batch]
                target_batch = [instance[1] for instance in batch]
                
                tokenized_src_source_batch = self.tokenizer(source_batch, return_tensors='pt', truncation=True, padding=True).to("cuda")
                tokenized_src_target_batch = self.tokenizer(target_batch, return_tensors='pt', truncation=True, padding=True).to("cuda")
                
                _, source_enc_hidden_states, _ = self.get_encoder_outputs(tokenized_src_source_batch)
                _, target_enc_hidden_states, _ = self.get_encoder_outputs(tokenized_src_target_batch)

                for layer_idx in selected_layers:
                    if layer_idx not in source_hidden_states_per_layer:
                        source_hidden_states_per_layer[layer_idx] = []
                    if layer_idx not in target_hidden_states_per_layer:
                        target_hidden_states_per_layer[layer_idx] = []
                    source_hidden_states_per_layer[layer_idx].append(source_enc_hidden_states[layer_idx].mean(axis=-2, keepdims=False))
                    target_hidden_states_per_layer[layer_idx].append(target_enc_hidden_states[layer_idx].mean(axis=-2, keepdims=False))

        for layer_idx in source_hidden_states_per_layer.keys():
            source_hidden_states_per_layer[layer_idx] = torch.cat(source_hidden_states_per_layer[layer_idx], dim=0)
        
        for layer_idx in target_hidden_states_per_layer.keys():
            target_hidden_states_per_layer[layer_idx] = torch.cat(target_hidden_states_per_layer[layer_idx], dim=0)

        return source_hidden_states_per_layer, target_hidden_states_per_layer # layer_size, num_instances, dim

    def decode_on_particular_hidden_states(self, queries_dl, selected_layers):
        self.model.eval()
        source_preds = dict();target_preds = dict()
        all_enc_hidden_states, attentions, tgt_batches = [], [], []
        with torch.no_grad():
            for batch in tqdm(queries_dl):
                source_batch = [instance[0] for instance in batch]
                target_batch = [instance[1] for instance in batch]


                tokenized_src_source_batch = self.tokenizer(source_batch, return_tensors='pt', truncation=True, padding=True, max_length=256).to("cuda")
                tokenized_src_target_batch = self.tokenizer(target_batch, return_tensors='pt', truncation=True, padding=True, max_length=256).to("cuda")
                
                tokenized_tgt_batch = self.tokenizer(['']*len(batch), return_tensors='pt', truncation=True, padding=True).to("cuda")
                _, source_enc_hidden_states, source_attns = self.get_encoder_outputs(tokenized_src_source_batch)
                _, target_enc_hidden_states, target_attns = self.get_encoder_outputs(tokenized_src_target_batch)
                

                for layer in tqdm(selected_layers):
                    # Generate using intermediate encoder states
                    source_decoder_output_ids = self.model.generate(
                        input_ids=tokenized_tgt_batch.input_ids,
                        attention_mask=tokenized_tgt_batch.attention_mask,
                        encoder_outputs=BaseModelOutput(last_hidden_state=source_enc_hidden_states[layer], hidden_states=source_enc_hidden_states, attentions=source_attns),
                        max_length=50,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    target_decoder_output_ids = self.model.generate(
                        input_ids=tokenized_tgt_batch.input_ids,
                        attention_mask=tokenized_tgt_batch.attention_mask,
                        encoder_outputs=BaseModelOutput(last_hidden_state=target_enc_hidden_states[layer], hidden_states=target_enc_hidden_states, attentions=target_attns),
                        max_length=50,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    source_outputs = self.tokenizer.batch_decode(source_decoder_output_ids, skip_special_tokens=True)
                    target_outputs = self.tokenizer.batch_decode(target_decoder_output_ids, skip_special_tokens=True)

                    if layer not in source_preds:
                        source_preds[layer] = []
                    source_preds[layer].extend(source_outputs)
                

                    if layer not in target_preds:
                        target_preds[layer] = []
                    target_preds[layer].extend(target_outputs)
        return source_preds, target_preds


    def decode_on_particular_hidden_state(self, enc_outputs, attentions, tgt_batches, layer_idx):
        self.model.eval()

        all_source_decoder_outputs = []
        with torch.no_grad():
            for enc_hidden_states, attns, tgt_batch in zip(enc_outputs, attentions, tgt_batches):
                assert layer_idx < len(enc_hidden_states[0])
                selected_source_enc_hidden_states = enc_hidden_states[0][layer_idx]
                selected_target_enc_hidden_states = enc_hidden_states[1][layer_idx]


                # Generate using intermediate encoder states
                source_decoder_output_ids = self.model.generate(
                    input_ids=tgt_batch.input_ids,
                    attention_mask=tgt_batch.attention_mask,
                    encoder_outputs=BaseModelOutput(last_hidden_state=selected_source_enc_hidden_states, hidden_states=enc_hidden_states[0], attentions=attns[0]),
                    max_length=50,
                    output_hidden_states=True,
                    return_dict=True,
                )
                target_decoder_output_ids = self.model.generate(
                    input_ids=tgt_batch.input_ids,
                    attention_mask=tgt_batch.attention_mask,
                    encoder_outputs=BaseModelOutput(last_hidden_state=selected_target_enc_hidden_states, hidden_states=enc_hidden_states[1], attentions=attns[1]),
                    max_length=50,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                source_outputs = self.tokenizer.batch_decode(source_decoder_output_ids, skip_special_tokens=True)
                target_outputs = self.tokenizer.batch_decode(target_decoder_output_ids, skip_special_tokens=True)
     
                all_source_decoder_outputs += source_outputs
                all_target_decoder_outputs += target_outputs

        # decode all outputs from logits into texts
        return all_source_decoder_outputs, all_target_decoder_outputs