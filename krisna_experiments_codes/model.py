import evaluate
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutput
import pdb
from tqdm.contrib import tzip
import numpy as np

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
    def __init__(self, model, tokenizer, task_type):
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type
    
    def _tokenize_obj(self, obj_labels):
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
    
    def _mask_sentences(self, prompts, obj_token_lengths):
        new_prompts = []
        for prompt, obj_token_length in zip(prompts, obj_token_lengths):
            new_mask = " ".join(["<mask>"]*obj_token_length)
            new_prompt = prompt.replace('[Y]', new_mask)
            new_prompt = new_prompt.replace('<mask>', self.tokenizer.mask_token)
            new_prompts.append(new_prompt)
        return new_prompts
      
    def inference_cloze_task(self, instances, batch_size=16, selected_layers=[], beam_topk=3, ranking_topk=5):
        self.model.eval()

        mono_rank_preds_per_layer, cs_rank_preds_per_layer = dict(), dict()
        labels = []
        batch_cnt = len(instances)//batch_size

        mono_rank_preds = []; cs_rank_preds = []
        mono_rank_preds_per_layer = dict(); cs_rank_preds_per_layer = dict()

        for layer in selected_layers:
            mono_rank_preds_per_layer[layer] = mono_rank_preds
            cs_rank_preds_per_layer[layer] = cs_rank_preds

        with torch.no_grad():
            for i in tqdm(range(0, batch_cnt), desc='batch'):
                batch = instances[i*batch_size:min((i+1)*batch_size, len(instances))]
                obj_labels = [instance['obj_label'] for instance in batch]
                labels.extend(obj_labels)
                mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']) for instance in batch]
                cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']) for instance in batch]
                all_obj_tokens, obj_token_lengths, max_obj_token_len = self._tokenize_obj(obj_labels)
                mono_prompts = self._mask_sentences(mono_prompts, obj_token_lengths)
                cs_prompts = self._mask_sentences(cs_prompts, obj_token_lengths)

                mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
                cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')

                if len(selected_layers) == 0 or -1 in selected_layers:
                    mono_outputs = self.model(**mono_inputs, output_hidden_states=True)
                    cs_outputs = self.model(**cs_inputs, output_hidden_states=True)

                    mono_log_probs = torch.log(mono_outputs['logits'].softmax(dim=-1)) # batch*seq*vocab
                    cs_log_probs = torch.log(cs_outputs['logits'].softmax(dim=-1))
                    ####pdb.set_trace()
                    
                    mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # indices return indices in vocabulary
                    cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True)
                    ####pdb.set_trace()

                    mono_masked_indices = torch.nonzero(mono_inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=False)
                    mono_masked_index = dict()
                    mono_masked_index_list = []
                    mono_masked_rows, mono_masked_cols = [], []
                    for pos in mono_masked_indices:
                        row_pos = pos[0].item()
                        col_pos = pos[1].item()
                        if row_pos not in mono_masked_index:
                            mono_masked_index[row_pos] = []
                        mono_masked_index[row_pos].append(col_pos)
                    ####pdb.set_trace()
                    for key in sorted(mono_masked_index.keys()):
                        mono_masked_rows.append(key) 
                        mono_masked_cols.append(min(mono_masked_index[key]))
                    ####pdb.set_trace()
                

                    cs_masked_indices = torch.nonzero(cs_inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=False)
                    ####pdb.set_trace()
                    cs_masked_index = dict()
                    cs_masked_index_list = []
                    cs_masked_rows, cs_masked_cols = [], []
                    for pos in cs_masked_indices:
                        row_pos = pos[0].item()
                        col_pos = pos[1].item()
                        if row_pos not in cs_masked_index:
                            cs_masked_index[row_pos] = []
                        cs_masked_index[row_pos].append(col_pos)
                    ####pdb.set_trace()
                    for key in sorted(cs_masked_index.keys()):
                        cs_masked_rows.append(key) 
                        cs_masked_cols.append(min(cs_masked_index[key]))
                    ####pdb.set_trace()
                    
                    # all these have batch_size*1
                    mono_start_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
                    cs_start_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1

                    mono_current_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
                    cs_current_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1
                    
                    mono_masked_rows = np.array(mono_masked_rows)[..., np.newaxis] 
                    mono_masked_cols = np.array(mono_masked_cols)[..., np.newaxis]
                    cs_masked_rows = np.array(cs_masked_rows)[..., np.newaxis]
                    cs_masked_cols = np.array(cs_masked_cols)[..., np.newaxis]

                    ####pdb.set_trace()

                    

                    mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_masked_cols, :], mono_topk_indices[mono_masked_rows, mono_masked_cols, :] # bach*1*k
                    cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_masked_cols, :], cs_topk_indices[cs_masked_rows, cs_masked_cols, :]
                    []
                    
                    batch_sz = mono_current_pos.shape[0]

                    mono_batch_rank_preds = []
                    cs_batch_rank_preds = []
                    for batch_idx in range(batch_sz):
                        mono_dict = dict()
                        cs_dict = dict()
                        mono_batch_rank_preds.append(mono_dict)
                        cs_batch_rank_preds.append(cs_dict)
                    
                    for batch_idx in range(batch_sz):
                        mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0]
                        cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0]
                        
                        for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                           
                            decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                            decoded_word = f"{decoded_word}<{curr_mono_topk_token_idx}>"
                            #pdb.set_trace()
                            mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()
                            #pdb.set_trace()
                        
                        for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                            decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                            decoded_word = f"{decoded_word}<{curr_cs_topk_token_idx}>"
                            cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob.item()
                            #pdb.set_trace()

                    batch_indices = torch.arange(batch_sz).unsqueeze(-1)
                    #pdb.set_trace()


                    mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1)) # k b
                    mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor)
                    ####pdb.set_trace()

                    cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1)) # k*batch_size*1
                    cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor) # k*batch_size*1
                    ####pdb.set_trace()

                    for span_len in range(1, max_obj_token_len):
                        mono_span_pos_rows = []
                        mono_span_pos_cols = []

                        cs_span_pos_rows = []
                        cs_span_pos_cols = []

                        selected_mono_topk_indices = []
                        selected_cs_topk_indices = []

                        mono_span_pos = np.zeros((batch_sz, span_len))
                        cs_span_pos = np.zeros((batch_sz, span_len))
                        ####pdb.set_trace()

                        # set span indices
                        for batch_idx in range(batch_sz):
                            if obj_token_lengths[batch_idx] >= (span_len+1):
                                mono_current_pos[batch_idx][0] += 1
                                mono_span_pos_rows.append([batch_idx])
                                mono_span_pos_cols.append(np.arange(mono_start_pos[batch_idx][0], mono_current_pos[batch_idx][0])) # batch_size*span
                                selected_mono_topk_indices.append(mono_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0))
                        
                                cs_current_pos[batch_idx][0] += 1
                                cs_span_pos_rows.append([batch_idx])
                                cs_span_pos_cols.append(np.arange(cs_start_pos[batch_idx][0], cs_current_pos[batch_idx][0])) # batch_size*span
                                selected_cs_topk_indices.append(cs_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0))
                        #pdb.set_trace()
                        
                        selected_cs_topk_indices = torch.cat(selected_cs_topk_indices, axis=0).permute((1,0,2))
                        selected_mono_topk_indices = torch.cat(selected_mono_topk_indices, axis=0).permute((1,0,2))
                        #pdb.set_trace()
                        

                        all_mono_joint_proba = []
                        all_cs_joint_proba = []

                        for cand_rank in range(len(mono_topk_log_prob)):
                            
                            mono_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in mono_inputs.items()}
                            mono_inputs_copy['input_ids'][mono_span_pos_rows, mono_span_pos_cols] = selected_mono_topk_indices[cand_rank].to('cuda') # batch*seq_length*
                            #pdb.set_trace()
                            
                            cs_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in cs_inputs.items()}
                            cs_inputs_copy['input_ids'][cs_span_pos_rows, cs_span_pos_cols] = selected_cs_topk_indices[cand_rank].to('cuda') # batch*seq_length*
                            #pdb.set_trace()

                            mono_cand_proba = self.model(**mono_inputs_copy).logits.softmax(dim=-1) # batch*seq_length*vocab
                            
                            for key, tensor in mono_inputs.items():
                                tensor.detach().cpu()
                            mono_cand_proba = torch.log(mono_cand_proba)
                            #pdb.set_trace()
                            cs_cand_proba = self.model(**cs_inputs_copy).logits.softmax(dim=-1) # batch*seq_length*vocab
                            
                            cs_cand_proba = torch.log(cs_cand_proba)
                            for key, tensor in cs_inputs.items():
                                tensor.detach().cpu()
                            #pdb.set_trace()

                            mono_cand_proba = mono_cand_proba[batch_indices, mono_current_pos, :].squeeze(1) # batch*vocab
                            cs_cand_proba = cs_cand_proba[batch_indices, cs_current_pos, :].squeeze(1) #batch*vocab
                            #pdb.set_trace()

                            mono_prev_proba = mono_topk_log_prob[cand_rank] # batch*len
                            cs_prev_proba = cs_topk_log_prob[cand_rank] # batch*1
                            #pdb.set_trace()

                            mono_joint_proba = mono_prev_proba + mono_cand_proba # batch*vocab
                            mono_joint_proba = mono_joint_proba.unsqueeze(1)
                            all_mono_joint_proba.append(mono_joint_proba)
                            cs_joint_proba = cs_prev_proba + cs_cand_proba # batch*vocab
                            cs_joint_proba = cs_joint_proba.unsqueeze(1)
                            all_cs_joint_proba.append(cs_joint_proba)
                            #pdb.set_trace()
                        
                        all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # batch*k*vocab
                        all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # batch*k*vocab
                        #pdb.set_trace()

                        vocab_size = all_cs_joint_proba.shape[-1]
                        #pdb.set_trace()

                        all_mono_joint_proba = all_mono_joint_proba.view(all_mono_joint_proba.shape[0], -1) #batch*(k*vocab)
                        all_cs_joint_proba = all_cs_joint_proba.view(all_cs_joint_proba.shape[0], -1) #batch*(k*vocab)
                        #pdb.set_trace()

                        next_mono_topk_log_prob, next_mono_topk_indices = all_mono_joint_proba.topk(beam_topk, sorted=True) #batch*k
                        prefix_indices_mono, vocab_indices_mono = next_mono_topk_indices//vocab_size, next_mono_topk_indices%vocab_size
                        prefix_indices_mono = prefix_indices_mono.cpu()
                        #pdb.set_trace()


                        next_cs_topk_log_prob, next_cs_topk_indices = all_cs_joint_proba.topk(beam_topk, sorted=True)
                        prefix_indices_cs, vocab_indices_cs = next_cs_topk_indices//vocab_size, next_cs_topk_indices%vocab_size
                        prefix_indices_cs = prefix_indices_cs.cpu()
                        #pdb.set_trace()

                        new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
                        new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1))
                        #pdb.set_trace()

                        mono_topk_indices = mono_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
                        cs_topk_indices = cs_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
                        #pdb.set_trace()

                        for batch_idx in range(batch_sz):
                            #pdb.set_trace()
                            new_mono_indices[batch_idx, :, :-1] = mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]]
                            new_mono_indices[batch_idx, :, -1] = vocab_indices_mono[batch_idx]
                            #pdb.set_trace()

                            new_cs_indices[batch_idx, :, :-1] = cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]]
                            new_cs_indices[batch_idx, :, -1] =  vocab_indices_cs[batch_idx]
                            #pdb.set_trace()

                            if obj_token_lengths[batch_idx] >= (span_len+1):
                                mono_topk_indices_instance = new_mono_indices[batch_idx]
                                mono_topk_log_prob_instance = next_mono_topk_log_prob[batch_idx]
                                for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                                    
                                    vocab_ids = curr_mono_topk_token_idx.type(torch.LongTensor)
                                    decoded_word =  self.tokenizer.batch_decode(vocab_ids)
                                    decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)])

                                    #pdb.set_trace()
                                    mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()/(span_len+1)
                                    #pdb.set_trace()
                        
                                cs_topk_indices_instance = new_cs_indices[batch_idx]
                                cs_topk_log_prob_instance = next_cs_topk_log_prob[batch_idx]
                                for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                                    vocab_ids = curr_cs_topk_token_idx.type(torch.LongTensor)
                                    decoded_word = self.tokenizer.batch_decode(vocab_ids)
                                    decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)])
                                    cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob.item()/(span_len+1)
                                    #pdb.set_trace()
                        
                        cs_topk_indices = new_cs_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len
                        mono_topk_indices = new_mono_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len
                        #pdb.set_trace()

                        mono_topk_log_prob = next_mono_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
                        cs_topk_log_prob = next_cs_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
                        #pdb.set_trace()

                                     
                    # rank all preds
                    for batch_preds in mono_batch_rank_preds:
                        #pdb.set_trace()
                        sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
                        selected_words = sorted_batch_preds[:ranking_topk]
                        mono_rank_preds.append(selected_words)
                        #pdb.set_trace()
                    
                    for batch_preds in cs_batch_rank_preds:
                        #pdb.set_trace()
                        sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
                        selected_words = sorted_batch_preds[:ranking_topk]
                        cs_rank_preds.append(selected_words)
                        #pdb.set_trace()
                
                else:
                    mono_outputs = self.model(**mono_inputs, output_hidden_states=True)
                    cs_outputs = self.model(**cs_inputs, output_hidden_states=True)

                    mono_hidden_states = mono_outputs.hidden_states[1:]
                    cs_hidden_states = cs_outputs.hidden_states[1:]


                    mono_masked_indices = torch.nonzero(mono_inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=False)
                    mono_masked_index = dict()
                    mono_masked_index_list = []
                    mono_masked_rows, mono_masked_cols = [], []
                    for pos in mono_masked_indices:
                        row_pos = pos[0].item()
                        col_pos = pos[1].item()
                        if row_pos not in mono_masked_index:
                            mono_masked_index[row_pos] = []
                        mono_masked_index[row_pos].append(col_pos)
                    for key in sorted(mono_masked_index.keys()):
                        mono_masked_rows.append(key) 
                        mono_masked_cols.append(min(mono_masked_index[key]))
                    ####pdb.set_trace()
                    

                    cs_masked_indices = torch.nonzero(cs_inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=False)
                    cs_masked_index = dict()
                    cs_masked_index_list = []
                    cs_masked_rows, cs_masked_cols = [], []
                    for pos in cs_masked_indices:
                        row_pos = pos[0].item()
                        col_pos = pos[1].item()
                        if row_pos not in cs_masked_index:
                            cs_masked_index[row_pos] = []
                        cs_masked_index[row_pos].append(col_pos)
                    for key in sorted(cs_masked_index.keys()):
                        cs_masked_rows.append(key) 
                        cs_masked_cols.append(min(cs_masked_index[key]))
                    ####pdb.set_trace()
                    
                    # all these have batch_size*1
                    mono_start_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
                    cs_start_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1
                    mono_masked_rows = np.array(mono_masked_rows)[..., np.newaxis] 
                    cs_masked_rows = np.array(cs_masked_rows)[..., np.newaxis]
                    ####pdb.set_trace()

                    mono_batch_rank_preds_per_layer = dict()
                    cs_batch_rank_preds_per_layer = dict()

                    batch_sz = len(batch)
                    batch_indices = torch.arange(batch_sz).unsqueeze(-1)
                    ####pdb.set_trace()

                    
                    for layer in selected_layers:
                        mono_batch_rank_preds_per_layer[layer] = []
                        cs_batch_rank_preds_per_layer[layer] = []
                        for _ in range(batch_sz):
                            mono_dict = dict()
                            cs_dict = dict()
                            mono_batch_rank_preds_per_layer[layer].append(mono_dict)
                            cs_batch_rank_preds_per_layer[layer].append(cs_dict)

                    for layer in  selected_layers:
                        assert layer < len(mono_hidden_states)

                        mono_logits = self.model.lm_head(mono_hidden_states[layer])
                        cs_logits = self.model.lm_head(cs_hidden_states[layer])
                        ####pdb.set_trace()
                        
                        mono_log_probs = torch.log(mono_logits.softmax(dim=-1)) # batch*seq*vocab
                        cs_log_probs = torch.log(cs_logits.softmax(dim=-1))
                        ####pdb.set_trace()
                        
                        mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # indices return indices in vocabulary
                        cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True)
                        ####pdb.set_trace()


                        mono_current_pos = np.array(mono_masked_cols.copy())[..., np.newaxis] # batch_size*1
                        cs_current_pos = np.array(cs_masked_cols.copy())[..., np.newaxis] # batch_size*1
                        ####pdb.set_trace()
                    
        

                        mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_start_pos, :], mono_topk_indices[mono_masked_rows, mono_start_pos, :] # bach*1*k
                        cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_start_pos, :], cs_topk_indices[cs_masked_rows, cs_start_pos, :]
                        ####pdb.set_trace()
                        
                        
                        for batch_idx in range(batch_sz):
                            mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0]
                            cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0]
                            
                            for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                                #####pdb.set_trace()
                                decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                                decoded_word = f"{decoded_word}<{curr_mono_topk_token_idx}>"
                                mono_batch_rank_preds_per_layer[layer][batch_idx][decoded_word] = curr_mono_topk_log_prob.item()
                                ####pdb.set_trace()
                            
                            for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                                decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                                decoded_word = f"{decoded_word}<{curr_cs_topk_token_idx}>"
                                cs_batch_rank_preds_per_layer[layer][batch_idx][decoded_word] = curr_cs_topk_log_prob.item()
                                #####pdb.set_trace()
                            ####pdb.set_trace()

                        
                        mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1))
                        mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor)
                        

                        cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1)) # k*batch_size*1
                        cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor) # k*batch_size*1
                        ####pdb.set_trace()

                        for span_len in range(1, max_obj_token_len):
                            mono_span_pos_rows = []
                            mono_span_pos_cols = []

                            cs_span_pos_rows = []
                            cs_span_pos_cols = []

                            selected_mono_topk_indices = []
                            selected_cs_topk_indices = []

                            mono_span_pos = np.zeros((batch_sz, span_len))
                            cs_span_pos = np.zeros((batch_sz, span_len))

                            # set span indices
                            for batch_idx in range(batch_sz):
                                if obj_token_lengths[batch_idx] >= (span_len+1):
                                    mono_current_pos[batch_idx][0] += 1
                                    mono_span_pos_rows.append([batch_idx])
                                    mono_span_pos_cols.append(np.arange(mono_start_pos[batch_idx][0], mono_current_pos[batch_idx][0])) # batch_size*span
                                    selected_mono_topk_indices.append(mono_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0))

                                    cs_current_pos[batch_idx][0] += 1
                                    cs_span_pos_rows.append([batch_idx])
                                    cs_span_pos_cols.append(np.arange(cs_start_pos[batch_idx][0], cs_current_pos[batch_idx][0])) # batch_size*span
                                    selected_cs_topk_indices.append(cs_topk_indices.permute((1,0,2))[batch_idx].unsqueeze(0))
                                ####pdb.set_trace()
                            
                            selected_cs_topk_indices = torch.cat(selected_cs_topk_indices, axis=0).permute((1,0,2))
                            selected_mono_topk_indices = torch.cat(selected_mono_topk_indices, axis=0).permute((1,0,2))
                            ####pdb.set_trace()
                            

                            all_mono_joint_proba = []
                            all_cs_joint_proba = []

                            for cand_rank in range(len(mono_topk_log_prob)):
                                
                                mono_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in mono_inputs.items()}
                                mono_inputs_copy['input_ids'][mono_span_pos_rows, mono_span_pos_cols] = selected_mono_topk_indices[cand_rank].to('cuda') # batch*seq_length*
                                
                                cs_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in cs_inputs.items()}
                                cs_inputs_copy['input_ids'][cs_span_pos_rows, cs_span_pos_cols] = selected_cs_topk_indices[cand_rank].to('cuda') # batch*seq_length*
                                ####pdb.set_trace()

                                mono_outputs = self.model(**mono_inputs_copy, output_hidden_states=True).hidden_states[1:][layer]
                                mono_cand_proba = self.model.lm_head(mono_outputs).softmax(dim=-1) # batch*seq_length*vocab
                                mono_cand_proba = torch.log(mono_cand_proba)
                                ####pdb.set_trace()
                                for key, tensor in mono_inputs.items():
                                    tensor.detach().cpu()
          
                                cs_outputs =  self.model(**cs_inputs_copy, output_hidden_states=True).hidden_states[layer]
                                cs_cand_proba = self.model.lm_head(cs_outputs).softmax(dim=-1) # batch*seq_length*vocab
                                cs_cand_proba = torch.log(cs_cand_proba)
                                ####pdb.set_trace()
                                for key, tensor in cs_inputs.items():
                                    tensor.detach().cpu()


                                mono_cand_proba = mono_cand_proba[batch_indices, mono_current_pos, :].squeeze(1) # batch*vocab
                                cs_cand_proba = cs_cand_proba[batch_indices, cs_current_pos, :].squeeze(1) #batch*vocab
                                ####pdb.set_trace()

                                mono_prev_proba = mono_topk_log_prob[cand_rank] # batch*len
                                cs_prev_proba = cs_topk_log_prob[cand_rank] # batch*1
                                ####pdb.set_trace()

                                mono_joint_proba = mono_prev_proba + mono_cand_proba # batch*vocab
                                mono_joint_proba = mono_joint_proba.unsqueeze(1)
                                all_mono_joint_proba.append(mono_joint_proba)
                                cs_joint_proba = cs_prev_proba + cs_cand_proba # batch*vocab
                                cs_joint_proba = cs_joint_proba.unsqueeze(1)
                                all_cs_joint_proba.append(cs_joint_proba)
                                ####pdb.set_trace()
                            
                            all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # batch*k*vocab
                            all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # batch*k*vocab
                            ####pdb.set_trace()

                            vocab_size = all_cs_joint_proba.shape[-1]
                            ####pdb.set_trace()

                            all_mono_joint_proba = all_mono_joint_proba.view(all_mono_joint_proba.shape[0], -1) #batch*(k*vocab)
                            all_cs_joint_proba = all_cs_joint_proba.view(all_cs_joint_proba.shape[0], -1) #batch*(k*vocab)
                            ####pdb.set_trace()

                            next_mono_topk_log_prob, next_mono_topk_indices = all_mono_joint_proba.topk(beam_topk, sorted=True) #batch*k
                            prefix_indices_mono, vocab_indices_mono = next_mono_topk_indices//vocab_size, next_mono_topk_indices%vocab_size
                            prefix_indices_mono = prefix_indices_mono.cpu()
                            ####pdb.set_trace()


                            next_cs_topk_log_prob, next_cs_topk_indices = all_cs_joint_proba.topk(beam_topk, sorted=True)
                            prefix_indices_cs, vocab_indices_cs = next_cs_topk_indices//vocab_size, next_cs_topk_indices%vocab_size
                            prefix_indices_cs = prefix_indices_cs.cpu()
                            ####pdb.set_trace()

                            new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
                            new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1))
                            ####pdb.set_trace()

                            mono_topk_indices = mono_topk_indices.permute((1,0,2)) #batch*k*1
                            cs_topk_indices = cs_topk_indices.permute((1,0,2)) #batch*k*1
                            ####pdb.set_trace()

                            for batch_idx in range(batch_sz):
                                new_mono_indices[batch_idx, :, :-1] = mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]]
                                new_mono_indices[batch_idx, :, -1] = vocab_indices_mono[batch_idx]

                                new_cs_indices[batch_idx, :, :-1] = cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]]
                                new_cs_indices[batch_idx, :, -1] =  vocab_indices_cs[batch_idx]
                                ####pdb.set_trace()

                                if obj_token_lengths[batch_idx] >= (span_len+1):
                                    mono_topk_indices_instance = new_mono_indices[batch_idx]
                                    mono_topk_log_prob_instance = next_mono_topk_log_prob[batch_idx]
                                    for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                                        vocab_indices = curr_mono_topk_token_idx.type(torch.LongTensor)
                                        decoded_word = self.tokenizer.batch_decode(vocab_indices)
                                        decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_indices)])
                                        mono_batch_rank_preds_per_layer[layer][batch_idx][decoded_word] = curr_mono_topk_log_prob.item()/(span_len+1)
                                        ####pdb.set_trace()
                            
                                    cs_topk_indices_instance = new_cs_indices[batch_idx]
                                    cs_topk_log_prob_instance = next_cs_topk_log_prob[batch_idx]
                                    for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                                        vocab_indices = curr_cs_topk_token_idx.type(torch.LongTensor)
                                        decoded_word = self.tokenizer.batch_decode(vocab_indices)
                                        decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_indices)])
                                        cs_batch_rank_preds_per_layer[layer][batch_idx][decoded_word] = curr_cs_topk_log_prob.item()/(span_len+1)
                                        ####pdb.set_trace()
                            
                            cs_topk_indices = new_cs_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len
                            mono_topk_indices = new_mono_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len
                            ####pdb.set_trace()

                            mono_topk_log_prob = next_mono_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
                            cs_topk_log_prob = next_cs_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
                            ####pdb.set_trace()

                        ####pdb.set_trace()      
                        # rank all preds
                        for batch_preds in mono_batch_rank_preds_per_layer[layer]:
                            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
                            selected_words = sorted_batch_preds[:ranking_topk]
                            mono_rank_preds_per_layer[layer].append(selected_words)
                            ####pdb.set_trace()
                        
                        for batch_preds in cs_batch_rank_preds_per_layer[layer]:
                            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
                            selected_words = sorted_batch_preds[:ranking_topk]
                            cs_rank_preds_per_layer[layer].append(selected_words)
                            ####pdb.set_trace()
                        pdb.set_trace()
        
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
                    
                    for layer in tqdm(selected_layers):
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
                    #####pdb.set_trace()
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
                    #####pdb.set_trace()
                    input = self.tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True).to("cuda")

                    outputs = self.model(**input)
                    logits = outputs.logits
                    predicted = logits.argmax(dim=-1).detach().cpu().numpy()
                    #####pdb.set_trace()
                    if self.task_type == 'nli_qa':
                        #####pdb.set_trace()
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
       #####pdb.set_trace()
       return encoder_outputs[0], encoder_outputs[1][1:], encoder_outputs[2]

    def _tokenize_obj(self, obj_labels):
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
        
    def _mask_sentences(self, prompts):
        new_prompts = []
        for prompt in prompts:
            new_mask = "<extra_id_0>"
            new_prompt = prompt.replace('[Y]', new_mask)
            new_prompts.append(new_prompt)
        return new_prompts
                
    def inference_cloze_task(self, instances, batch_size=16, selected_layers=[], beam_topk=5, ranking_topk=5):
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
                all_obj_tokens, all_label_tokens, obj_token_lengths, label_token_lengths, all_attn_masks = self._tokenize_obj(obj_labels)
                mono_prompts = self._mask_sentences(mono_prompts)
                cs_prompts = self._mask_sentences(cs_prompts)
                
                mono_inputs = self.tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
                cs_inputs = self.tokenizer(cs_prompts, padding=True, truncation=True, return_tensors='pt').to('cuda')
               
                all_label_tokens = torch.Tensor(all_label_tokens).to('cuda').long()
                all_attn_masks = torch.Tensor(all_attn_masks).to('cuda').long()
                mono_input_ids = mono_inputs['input_ids']
                cs_input_ids = cs_inputs['input_ids']

                if len(selected_layers) == 0 or -1 in selected_layers:
                    mono_outputs = self.model(input_ids=mono_input_ids, labels=all_label_tokens)
                    mono_outputs = mono_outputs.logits
                    mono_masked_indices = []
                    mono_masked_start_idx = []
                    mono_masked_current_idx = []
                    mono_masked_batch_indices = []

                    for idx, mono_output in enumerate(mono_outputs):
                        mono_masked_indices.append(list(range(mono_output.size()[1]))[1:-2-(mono_output.size()[1]-label_token_lengths[idx])])
                        mono_masked_start_idx.append(min(mono_masked_indices[idx]))
                        mono_masked_current_idx.append(min(mono_masked_indices[idx]))
                        mono_masked_batch_indices.append(idx)
                        assert len(mono_masked_indices[idx]) == obj_token_lengths[idx]

                    
                    cs_outputs = self.model(input_ids=cs_input_ids, labels=all_label_tokens)
                    cs_outputs = cs_outputs.logits
                    cs_masked_indices = [] # batch_size*l(l: length_mask & varies
                    cs_masked_start_idx = []
                    cs_masked_current_idx = []
                    cs_masked_batch_indices = []
                    
                    max_obj_token_len = max(obj_token_lengths)
                    for idx, cs_output in enumerate(cs_outputs):
                        cs_masked_indices.append(list(range(cs_output.size()[1]))[1:-2-(cs_output.size()[1]-label_token_lengths[idx])])
                        cs_masked_start_idx.append(min(cs_masked_indices[idx]))
                        cs_masked_current_idx.append(min(cs_masked_indices[idx]))
                        cs_masked_batch_indices.append(idx)
                        assert len(cs_masked_indices[idx]) == obj_token_lengths[idx]

                    mono_log_probs = torch.log(mono_outputs.softmax(dim=-1))
                    cs_log_probs = torch.log(cs_outputs.softmax(dim=-1))

                    mono_masked_rows = np.array(mono_masked_batch_indices)[..., np.newaxis] 
                    mono_masked_current_cols = np.array(mono_masked_start_idx)[..., np.newaxis]
                    cs_masked_rows = np.array(cs_masked_batch_indices)[..., np.newaxis]
                    cs_masked_current_cols = np.array(cs_masked_current_idx)[..., np.newaxis]

                    batch_sz = len(batch)

                    
                    mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # indices return indices in vocabulary batch_size x seq_length x topk
                    cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True) # batch_size x seq_length x topk

                    mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_masked_current_cols, :], mono_topk_indices[mono_masked_rows, mono_masked_current_cols, :] # bach*1*k
                    cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_masked_current_cols, :], cs_topk_indices[cs_masked_rows, cs_masked_current_cols, :]
 

                    mono_batch_rank_preds = []
                    cs_batch_rank_preds = []
                    for _ in range(batch_sz):
                        mono_dict = dict()
                        cs_dict = dict()
                        mono_batch_rank_preds.append(mono_dict)
                        cs_batch_rank_preds.append(cs_dict)

                    for batch_idx in range(batch_sz):
                        mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0]
                        cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0]
                        
                        for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                            decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                            decoded_word = f"{decoded_word}<{curr_mono_topk_token_idx}>"
                            mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()
                        
                        for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                            decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                            decoded_word = f"{decoded_word}<{curr_cs_topk_token_idx}>"
                            cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob
                    
                    mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1))
                    cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1))

                    mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor)
                    cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor)       

                    
                    for span_len in range(1, max_obj_token_len):
                        mono_span_pos_rows = []
                        mono_span_pos_cols = []

                        cs_span_pos_rows = []
                        cs_span_pos_cols = []

                        selected_mono_topk_indices = []
                        selected_cs_topk_indices = []

                        mono_span_pos = np.zeros((batch_sz, span_len))
                        cs_span_pos = np.zeros((batch_sz, span_len))
                        
                        
                        # set span indices
                        for batch_idx in range(batch_sz):
                            if obj_token_lengths[batch_idx] >= (span_len+1):
                                mono_masked_current_cols[batch_idx][0] += 1
                                cs_masked_current_cols[batch_idx][0] += 1


                        all_mono_joint_proba = []
                        all_cs_joint_proba = []
                        for cand_rank in range(len(mono_topk_log_prob)):    
                            mono_cand_proba = mono_log_probs[mono_masked_rows, mono_masked_current_cols, :].squeeze(1) #batch*1*vocab
                            cs_cand_proba = cs_log_probs[mono_masked_rows, mono_masked_current_cols, :].squeeze(1)

                            mono_prev_proba = mono_topk_log_prob[cand_rank] # batch*1
                            cs_prev_proba = cs_topk_log_prob[cand_rank] # batch*1


                            mono_joint_proba = mono_prev_proba + mono_cand_proba # batch*vocab
                            mono_joint_proba = mono_joint_proba.unsqueeze(1)
                            all_mono_joint_proba.append(mono_joint_proba)
                            cs_joint_proba = cs_prev_proba + cs_cand_proba # batch*vocab
                            cs_joint_proba = cs_joint_proba.unsqueeze(1)
                            all_cs_joint_proba.append(cs_joint_proba)

                        
                        all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # batch*k*vocab
                        all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # batch*k*vocab

                        
                        vocab_size = all_cs_joint_proba.shape[-1]

                        all_mono_joint_proba = all_mono_joint_proba.view(all_mono_joint_proba.shape[0], -1) #batch*(k*vocab)
                        all_cs_joint_proba = all_cs_joint_proba.view(all_cs_joint_proba.shape[0], -1) #batch*(k*vocab)

                        next_mono_topk_log_prob, next_mono_topk_indices = all_mono_joint_proba.topk(beam_topk, sorted=True) #batch*k
                        prefix_indices_mono, vocab_indices_mono = next_mono_topk_indices//vocab_size, next_mono_topk_indices%vocab_size
                        prefix_indices_mono = prefix_indices_mono.cpu()


                        next_cs_topk_log_prob, next_cs_topk_indices = all_cs_joint_proba.topk(beam_topk, sorted=True)
                        prefix_indices_cs, vocab_indices_cs = next_cs_topk_indices//vocab_size, next_cs_topk_indices%vocab_size
                        prefix_indices_cs = prefix_indices_cs.cpu()

                        new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
                        new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1))

                        mono_topk_indices = mono_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
                        cs_topk_indices = cs_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
                        

                        for batch_idx in range(batch_sz):
                            #####pdb.set_trace()
                            new_mono_indices[batch_idx, :, :-1] = mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]]
                            new_mono_indices[batch_idx, :, -1] = vocab_indices_mono[batch_idx]
                            
                            new_cs_indices[batch_idx, :, :-1] = cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]]

                            new_cs_indices[batch_idx, :, -1] =  vocab_indices_cs[batch_idx]

                            if obj_token_lengths[batch_idx] >= (span_len+1):
                                mono_topk_indices_instance = new_mono_indices[batch_idx]
                                mono_topk_log_prob_instance = next_mono_topk_log_prob[batch_idx]
                                for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                                    vocab_ids = curr_mono_topk_token_idx.type(torch.LongTensor)
                                    decoded_word =  self.tokenizer.batch_decode(vocab_ids)
                                    decoded_word = [f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)]
                                    mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()/(span_len+1)
                        
                                cs_topk_indices_instance = new_cs_indices[batch_idx]
                                cs_topk_log_prob_instance = next_cs_topk_log_prob[batch_idx]
                                for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                                    vocab_ids = curr_cs_topk_token_idx.type(torch.LongTensor)
                                    decoded_word =  self.tokenizer.batch_decode(vocab_ids)
                                    decoded_word = [f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)]
                                    cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob.item()/(span_len+1)

                                
                        
    
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

                        mono_masked_indices = []
                        mono_masked_start_idx = []
                        mono_masked_current_idx = []
                        mono_masked_batch_indices = []

                        for idx, mono_output in enumerate(mono_outputs):
                            mono_masked_indices.append(list(range(mono_output.size()[1]))[1:-2-(mono_output.size()[1]-label_token_lengths[idx])])
                            mono_masked_start_idx.append(min(mono_masked_indices[idx]))
                            mono_masked_current_idx.append(min(mono_masked_indices[idx]))
                            mono_masked_batch_indices.append(idx)
                            assert len(mono_masked_indices[idx]) == obj_token_lengths[idx]

                    
                        cs_masked_indices = [] # batch_size*l(l: length_mask & varies
                        cs_masked_start_idx = []
                        cs_masked_current_idx = []
                        cs_masked_batch_indices = []
                    
                        max_obj_token_len = max(obj_token_lengths)
                        for idx, cs_output in enumerate(cs_outputs):
                            cs_masked_indices.append(list(range(cs_output.size()[1]))[1:-2-(cs_output.size()[1]-label_token_lengths[idx])])
                            cs_masked_start_idx.append(min(cs_masked_indices[idx]))
                            cs_masked_current_idx.append(min(cs_masked_indices[idx]))
                            cs_masked_batch_indices.append(idx)
                            assert len(cs_masked_indices[idx]) == obj_token_lengths[idx]

                        mono_log_probs = torch.log(mono_outputs.softmax(dim=-1))
                        cs_log_probs = torch.log(cs_outputs.softmax(dim=-1))

                        mono_masked_rows = np.array(mono_masked_batch_indices)[..., np.newaxis] 
                        mono_masked_current_cols = np.array(mono_masked_start_idx)[..., np.newaxis]
                        cs_masked_rows = np.array(cs_masked_batch_indices)[..., np.newaxis]
                        cs_masked_current_cols = np.array(cs_masked_current_idx)[..., np.newaxis]

                        batch_sz = len(batch)

                    
                        mono_topk_log_prob, mono_topk_indices = mono_log_probs.topk(beam_topk, sorted=True) # indices return indices in vocabulary batch_size x seq_length x topk
                        cs_topk_log_prob, cs_topk_indices = cs_log_probs.topk(beam_topk, sorted=True) # batch_size x seq_length x topk


                        mono_topk_log_prob, mono_topk_indices = mono_topk_log_prob[mono_masked_rows, mono_masked_current_cols, :], mono_topk_indices[mono_masked_rows, mono_masked_current_cols, :] # bach*1*k
                        cs_topk_log_prob, cs_topk_indices = cs_topk_log_prob[cs_masked_rows, cs_masked_current_cols, :], cs_topk_indices[cs_masked_rows, cs_masked_current_cols, :]



                        mono_batch_rank_preds = []
                        cs_batch_rank_preds = []
                        for _ in range(batch_sz):
                            mono_dict = dict()
                            cs_dict = dict()
                            mono_batch_rank_preds.append(mono_dict)
                            cs_batch_rank_preds.append(cs_dict)

                        for batch_idx in range(batch_sz):
                            mono_topk_log_prob_instance, mono_topk_indices_instance = mono_topk_log_prob[batch_idx][0], mono_topk_indices[batch_idx][0]
                            cs_topk_log_prob_instance, cs_topk_indices_instance = cs_topk_log_prob[batch_idx][0], cs_topk_indices[batch_idx][0]
                            
                            for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                                decoded_word = self.tokenizer.decode(curr_mono_topk_token_idx)
                                ######pdb.set_trace()
                                mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob
                            
                            for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                                decoded_word = self.tokenizer.decode(curr_cs_topk_token_idx)
                                cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob
                    
                        mono_topk_log_prob = mono_topk_log_prob.permute((2,0,1))
                        cs_topk_log_prob = cs_topk_log_prob.permute((2,0,1))


                        mono_topk_indices = mono_topk_indices.permute((2,0,1)).type(torch.LongTensor)
                        cs_topk_indices = cs_topk_indices.permute((2,0,1)).type(torch.LongTensor)
                    
                        for span_len in range(1, max_obj_token_len):
                            mono_span_pos_rows = []
                            mono_span_pos_cols = []

                            cs_span_pos_rows = []
                            cs_span_pos_cols = []


                            mono_span_pos = np.zeros((batch_sz, span_len))
                            cs_span_pos = np.zeros((batch_sz, span_len))
                            
                            
                            # set span indices
                            for batch_idx in range(batch_sz):
                                if obj_token_lengths[batch_idx] >= (span_len+1):
                                    mono_masked_current_cols[batch_idx][0] += 1
                                    cs_masked_current_cols[batch_idx][0] += 1

                            

                            all_mono_joint_proba = []
                            all_cs_joint_proba = []
                            for cand_rank in range(len(mono_topk_log_prob)):    
                                mono_cand_proba = mono_log_probs[mono_masked_rows, mono_masked_current_cols, :].squeeze(1) #batch*1*vocab
                                cs_cand_proba = cs_log_probs[mono_masked_rows, mono_masked_current_cols, :].squeeze(1)

                                mono_prev_proba = mono_topk_log_prob[cand_rank] # batch*1
                                cs_prev_proba = cs_topk_log_prob[cand_rank] # batch*1


                                mono_joint_proba = mono_prev_proba + mono_cand_proba # batch*vocab
                                mono_joint_proba = mono_joint_proba.unsqueeze(1)
                                all_mono_joint_proba.append(mono_joint_proba)
                                cs_joint_proba = cs_prev_proba + cs_cand_proba # batch*vocab
                                cs_joint_proba = cs_joint_proba.unsqueeze(1)
                                all_cs_joint_proba.append(cs_joint_proba)
                            
                            all_mono_joint_proba = torch.cat(all_mono_joint_proba, dim=1) # batch*k*vocab
                            all_cs_joint_proba = torch.cat(all_cs_joint_proba, dim=1) # batch*k*vocab
                            
                            vocab_size = all_cs_joint_proba.shape[-1]

                            all_mono_joint_proba = all_mono_joint_proba.view(all_mono_joint_proba.shape[0], -1) #batch*(k*vocab)
                            all_cs_joint_proba = all_cs_joint_proba.view(all_cs_joint_proba.shape[0], -1) #batch*(k*vocab)

                            next_mono_topk_log_prob, next_mono_topk_indices = all_mono_joint_proba.topk(beam_topk, sorted=True) #batch*k
                            prefix_indices_mono, vocab_indices_mono = next_mono_topk_indices//vocab_size, next_mono_topk_indices%vocab_size
                            prefix_indices_mono = prefix_indices_mono.cpu()


                            next_cs_topk_log_prob, next_cs_topk_indices = all_cs_joint_proba.topk(beam_topk, sorted=True)
                            prefix_indices_cs, vocab_indices_cs = next_cs_topk_indices//vocab_size, next_cs_topk_indices%vocab_size
                            prefix_indices_cs = prefix_indices_cs.cpu()

                            new_mono_indices = torch.zeros((batch_sz, len(mono_topk_log_prob), span_len+1)) # batch*k*len
                            new_cs_indices = torch.zeros((batch_sz, len(cs_topk_log_prob), span_len+1))

                            mono_topk_indices = mono_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1
                            cs_topk_indices= cs_topk_indices.permute((1,0,2)).to('cpu') #batch*k*1

                            for batch_idx in range(batch_sz):
                                ######pdb.set_trace()
                                new_mono_indices[batch_idx, :, :-1] = mono_topk_indices[batch_idx][prefix_indices_mono[batch_idx]]
                                new_mono_indices[batch_idx, :, -1] = vocab_indices_mono[batch_idx]

                                new_cs_indices[batch_idx, :, :-1] = cs_topk_indices[batch_idx][prefix_indices_cs[batch_idx]]
                                new_cs_indices[batch_idx, :, -1] =  vocab_indices_cs[batch_idx]

                                if obj_token_lengths[batch_idx] >= (span_len+1):
                                    mono_topk_indices_instance = new_mono_indices[batch_idx]
                                    mono_topk_log_prob_instance = next_mono_topk_log_prob[batch_idx]
                                    for curr_mono_topk_log_prob, curr_mono_topk_token_idx in zip(mono_topk_log_prob_instance, mono_topk_indices_instance):
                                        vocab_ids = curr_mono_topk_token_idx.type(torch.LongTensor)
                                        decoded_word =  self.tokenizer.batch_decode(vocab_ids)
                                        decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)])
                                        #####pdb.set_trace()
                                        mono_batch_rank_preds[batch_idx][decoded_word] = curr_mono_topk_log_prob.item()/(span_len+1)
                            
                                    cs_topk_indices_instance = new_cs_indices[batch_idx]
                                    cs_topk_log_prob_instance = next_cs_topk_log_prob[batch_idx]
                                    for curr_cs_topk_log_prob, curr_cs_topk_token_idx in zip(cs_topk_log_prob_instance, cs_topk_indices_instance):
                                        vocab_ids = curr_cs_topk_token_idx.type(torch.LongTensor)
                                        decoded_word =  self.tokenizer.batch_decode(vocab_ids)
                                        decoded_word = " ".join([f"{word}<{vocab_id}>" for word, vocab_id in zip(decoded_word, vocab_ids)])
                                        #####pdb.set_trace()
                                        cs_batch_rank_preds[batch_idx][decoded_word] = curr_cs_topk_log_prob.item()/(span_len+1)
                            
                            cs_topk_indices = new_cs_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len
                            mono_topk_indices = new_mono_indices.permute((1,0,2)).type(torch.LongTensor) #k*batch*len

                            mono_topk_log_prob = next_mono_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
                            cs_topk_log_prob = next_cs_topk_log_prob.permute((1,0)).unsqueeze(-1) # k*batch*1
                        
                        # rank all preds
                        for batch_preds in mono_batch_rank_preds:
                            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
                            selected_words = sorted_batch_preds[:ranking_topk]
                            mono_rank_preds_per_layer[layer].append(selected_words)
                        pdb.set_trace()
                        
                        for batch_preds in cs_batch_rank_preds:
                            sorted_batch_preds = sorted(batch_preds, key=batch_preds.get, reverse=True)
                            selected_words = sorted_batch_preds[:ranking_topk]
                            cs_rank_preds_per_layer[layer].append(selected_words)
                        pdb.set_trace()
                        
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
                        #####pdb.set_trace()
                        decoder_outputs = self.model.decoder(
                                input_ids=decoder_input_ids,
                                attention_mask=target_attn_mask,
                                encoder_hidden_states=enc_hidden_states[layer],
                                encoder_attention_mask=source_input['attention_mask']
                            )
                        sequence_output = decoder_outputs[0]
                        lm_logits = self.model.lm_head(sequence_output) # batch_size x seq_length x voc_size
                        #####pdb.set_trace()
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
                #####pdb.set_trace()
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
        #####pdb.set_trace()

        for layer_idx in source_hidden_states_per_layer.keys():
            source_hidden_states_per_layer[layer_idx] = torch.cat(source_hidden_states_per_layer[layer_idx], dim=0)
        
        for layer_idx in target_hidden_states_per_layer.keys():
            target_hidden_states_per_layer[layer_idx] = torch.cat(target_hidden_states_per_layer[layer_idx], dim=0)
        #####pdb.set_trace()

        return source_hidden_states_per_layer, target_hidden_states_per_layer # layer_size, num_instances, dim

    def decode_on_particular_hidden_states(self, queries_dl, selected_layers):
        self.model.eval()
        source_preds = dict();target_preds = dict()
        all_enc_hidden_states, attentions, tgt_batches = [], [], []
        with torch.no_grad():
            for batch in tqdm(queries_dl):
                #####pdb.set_trace()
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



                #####pdb.set_trace()
        return source_preds, target_preds


    def decode_on_particular_hidden_state(self, enc_outputs, attentions, tgt_batches, layer_idx):
        self.model.eval()

        all_source_decoder_outputs = []
        with torch.no_grad():
            for enc_hidden_states, attns, tgt_batch in tzip(enc_outputs, attentions, tgt_batches):
                #####pdb.set_trace()
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