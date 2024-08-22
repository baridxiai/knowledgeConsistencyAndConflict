import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence


def seq_rep(layer_rep):
    for k, v in enumerate(layer_rep):
        v = torch.Tensor(v)
        if k == 0:
            re_seq_rep = torch.unsqueeze(v, 1)
        else:
            re_seq_rep = torch.concatenate([re_seq_rep, torch.unsqueeze(v, 1)], 1)
    return re_seq_rep


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class VAELens(nn.Module):
    def __init__(self, config, output_embed_matrix, rnn_type="lstm"):

        super().__init__()
        self.rnn_type = rnn_type

        self.embedding_dropout = nn.Dropout(p=0.1)

        # if rnn_type == 'rnn':
        #     rnn = nn.RNN
        # elif rnn_type == 'gru':
        #     rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        # else:
        #     raise ValueError()
        embedding_size = config.hidden_size
        hidden_size = config.hidden_size
        self.eos = torch.nn.Embedding(1, hidden_size).to("cuda")
        self.encoder_rnn = torch.nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        ).to("cuda")
        self.decoder_rnn = torch.nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        ).to("cuda")
        self.latent_size = config.hidden_size // 4

        self.hidden2mean = nn.Linear(hidden_size, self.latent_size).to("cuda")
        self.hidden2logv = nn.Linear(hidden_size, self.latent_size).to("cuda")
        self.latent2hidden = nn.Linear(self.latent_size, hidden_size).to("cuda")
        self.outputs2vocab = output_embed_matrix
        self.outputs2vocab.requires_grad = False

    def forward(self, input_rep,output_rep):

        batch_size = input_rep.size(0)

        _, h_c = self.encoder_rnn(input_rep.to("cuda"))
        hidden = h_c[0].to("cuda")

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv).to("cuda")

        z = torch.randn([batch_size, self.latent_size]).to("cuda")
        z = z * std.to("cuda") + mean.to("cuda")
        # DECODER
        hidden = self.latent2hidden(z.to("cuda")).to("cuda")

        # decoder input

        # decoder forward pass
        outputs, _ = self.decoder_rnn(
            output_rep.to("cuda"),
            (hidden, hidden),
        )

        # process outputs
        logits = self.outputs2vocab(outputs.to("cuda"))

        return logits, mean, logv, z

    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(
            0, batch_size, out=self.tensor()
        ).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = (
            self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        )

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(
                    torch.Tensor(batch_size).fill_(self.sos_idx).long()
                )

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(
                generations, input_sequence, sequence_running, t
            )

            # update gloabl running sequence
            sequence_mask[sequence_running] = input_sequence != self.eos_idx
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(
                    0, len(running_seqs), out=self.tensor()
                ).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode="greedy"):

        if mode == "greedy":
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:, t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
