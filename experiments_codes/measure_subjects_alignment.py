import os
from argparse import ArgumentParser
import pickle
from utils import initialize_wrapped_model_and_tokenizer, load_mlama
from tqdm import tqdm
import json
import torch

import math
import numpy as np

class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        X = X.to(self.device)
        Y = Y.to(self.device)
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

def main(args):
    task_type = 'cloze'
    wrapped_model, tokenizer = initialize_wrapped_model_and_tokenizer(args.model_name, task_type)
    model = wrapped_model.model
    if args.model_type == 'encoder-decoder':
        model = model.encoder
    mlama_instances = load_mlama(args.matrix_lang, args.embedded_lang,tokenizer)
    model.eval()

    mono_layerwise_representations = dict()
    cs_layerwise_representations = dict()





    for layer in args.selected_layers:
        mono_layerwise_representations[layer] = []
        cs_layerwise_representations[layer] = []

    accum_batch_size = 0
    cka = CudaCKA('cuda')
    layerwise_similarity = dict()
    with torch.no_grad():
        for batch_pos in tqdm(range(0, len(mlama_instances), args.batch_size)):

            batch = mlama_instances[batch_pos:min(batch_pos+args.batch_size, len(mlama_instances))]
            mono_prompts = [instance['template'].replace('[X]', instance['subj_label_same_lang']) for instance in batch]
            cs_prompts = [instance['template'].replace('[X]', instance['subj_label_cross_lang']) for instance in batch]

            mono_inputs = tokenizer(mono_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
            cs_inputs = tokenizer(cs_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)

            mono_attn_masks = mono_inputs['attention_mask'].unsqueeze(-1).detach().cpu()
            cs_attn_masks = cs_inputs['attention_mask'].unsqueeze(-1).detach().cpu()
            
            mono_out_hidden_states = model(**mono_inputs, output_hidden_states=True).hidden_states[1:]
            cs_out_hidden_states = model(**cs_inputs, output_hidden_states=True).hidden_states[1:]



            for layer in args.selected_layers:
                assert layer < len(mono_out_hidden_states)
                if args.model_type != 'decoder':
                    mono_sent_embeddings = (mono_out_hidden_states[layer].detach().cpu()*mono_attn_masks).sum(dim=1)/mono_attn_masks.sum(dim=1)
                    cs_sent_embeddings = (cs_out_hidden_states[layer].detach().cpu()*cs_attn_masks).sum(dim=1)/cs_attn_masks.sum(dim=1)
                else:
                    mono_sent_embeddings = mono_out_hidden_states[layer][:,-1,:]
                    cs_sent_embeddings = cs_out_hidden_states[layer][:,-1,:]
                

                rep_sim = cka.kernel_CKA(mono_sent_embeddings.float(), cs_sent_embeddings.float()).detach().cpu().item()
                if layer not in layerwise_similarity:
                    layerwise_similarity[layer] = []
                layerwise_similarity[layer].append(rep_sim)


    for layer in tqdm(args.selected_layers):
        valid_values = [val for val in layerwise_similarity[layer] if not np.isnan(val)]
        layerwise_similarity[layer] = {
            'mean':sum(valid_values)/len(valid_values),
            'error':1.96*(np.std(valid_values, ddof=1)/np.sqrt(len(valid_values)))
        }

    out_filepath = f"{args.output_prefix}_matrix-{args.matrix_lang}-embedded-{args.embedded_lang}-layerwise-representation-sim.json"
    
    
    with open(out_filepath, 'w') as f:
        json.dump(layerwise_similarity, f, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--selected_layers', type=int, nargs='+', default=[], help='Which layer(s) do you want to analyze')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--matrix_lang', type=str)
    parser.add_argument('--embedded_lang', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--output_prefix', type=str)

    args = parser.parse_args()

    main(args)
