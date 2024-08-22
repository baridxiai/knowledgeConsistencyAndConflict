# Name: Barid
import os
import wandb
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import datasets
import torch
import numpy as np
from mLama import mLama_util
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import VAELens

XLM_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
XLM_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").to("cuda")

VAELens_model = VAELens.VAELens(XLM_model.config, XLM_model.lm_head)
df = pd.read_parquet("./mlama53.parquet", engine="fastparquet")
m_lama = datasets.Dataset.from_pandas(df)
m_lama_organized = []
n = 0
temp = []
for data in m_lama:
    n += 1
    predicate_id, key, langauge, org, sub_label, obj_label = (
        mLama_util.m_lama_cross_robust_parser(data)
    )
    if langauge in ["zh"]:
        v = [org, sub_label, obj_label, langauge, key, predicate_id]
        m_lama_organized.append(v)
print(n)
batch_size = 8
epoch = 0
adm_opt = torch.optim.Adam(VAELens_model.parameters(), lr=0.0001)
wandb.init(
    # Set the project where this run will be logged
    project="VAELens",
    # Track hyperparameters and run metadata
)
NLL = torch.nn.NLLLoss(ignore_index=XLM_tok.pad_token_id, reduction="sum")


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == "logistic":
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == "linear":
        return min(1, step / x0)


def loss_fn(logits, label, mean, logv, anneal_function, step, k, x0):

    label = label.view(-1).to("cuda")
    logp = logits.view(-1, logits.size(2)).to("cuda")

    # Negative Log Likelihood
    NLL_loss = NLL(logp, label)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp()).to("cuda")
    KL_weight = kl_anneal_function(anneal_function, step, k, x0).to("cuda")

    return NLL_loss, KL_loss, KL_weight


while epoch < 10:
    epoch += 1
    m_lama_organized = random.shuffle(m_lama_organized)
    for i in range(len(m_lama_organized)):
        step = step +  1
        batch = m_lama_organized[i * batch_size : (i + 1) * batch_size]
        if len(batch) == 0:
            continue
        obj = [b[2] for b in batch]
        obj = XLM_tok(
            obj, padding=True,  return_tensors="pt"
        )
        label = torch.nn.functional.pad(obj["input_ids"],[0,1],"constant",XLM_tok.pad_token_id)[:,1:].to("cuda")
        output_rep = XLM_model.get_input_embeddings()(obj["input_ids"].to("cuda"))
        batch, positions, _ = mLama_util.query_answer_parser(batch, XLM_tok)
        rep = mLama_util.get_rep(batch, XLM_model, XLM_tok)

        layer_rep = mLama_util.get_position_rep(rep, positions)
        seq_rep = VAELens.seq_rep(layer_rep)
        logits, mean, logv, z = VAELens_model(seq_rep,output_rep)
        NLL_loss, KL_loss, KL_weight = loss_fn(
            logits, label, mean, logv, "linear", step, 1, 1000
        )
        loss = (NLL_loss + KL_weight * KL_loss) / batch_size
        adm_opt.zero_grad()
        loss.backward()
        adm_opt.step()
        if step %100 == 99:
            print(
                "Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                % (
                    loss.item(),
                    NLL_loss.item() / batch_size,
                    KL_loss.item() / batch_size,
                    KL_weight,
                )
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "NLL_loss": NLL_loss.item() / batch_size,
                    "KL_loss": KL_loss.item() / batch_size,
                    "KL_weight": KL_weight,
                }
            )
        if step % 1000 == 999:
            torch.save({"VAELens_model": VAELens_model}, "./VAELens_model")
