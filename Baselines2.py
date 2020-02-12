""" BiAtt flow, LSTM """
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import Const

# Normal attention
def dotprod_attention(q, k, v, attn_mask=None):
	"""
	:param : (batch, seq_len, seq_len)
	:return: (batch, seq_len, seq_len)
	"""
	attn = torch.matmul(q, k.transpose(1, 2))
	if attn_mask is not None:
		attn.data.masked_fill_(attn_mask, -1e10)

	attn = F.softmax(attn, dim=-1)
	output = torch.matmul(attn, v)
	return output, attn


class CNNencoder(nn.Module):
	def __init__(self, d_emb, d_filter, d_out, filter_list):
		super(CNNencoder, self).__init__()
		self.convs = nn.ModuleList(
			nn.Conv2d(1, d_filter, (filter_size, d_emb))
			for filter_size in filter_list
		)

		self.trans1 = nn.Sequential(
			nn.Linear(d_filter * len(filter_list), d_out),
			nn.Tanh()
		)

	def forward(self, sent):

		x = sent.unsqueeze(1)       # batch x 1 x seq x d_word_vec
		convs_results = [F.relu(conv(x)) for conv in self.convs]
		maxpl_results = [F.max_pool1d(conv.squeeze(-1), conv.size(2)).squeeze(2) for conv in convs_results]

		output = torch.cat(maxpl_results, dim=1)
		output = self.trans1(output)

		return output


# CNN
class cLSTM(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(cLSTM, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3,4,5])
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_lstm = nn.LSTM(100, 100, num_layers=1, bidirectional=False)
		self.dropout_mid = nn.Dropout(0.3)

		# classifier
		self.d_lin_2 = 100
		self.classifier = nn.Linear(self.d_lin_2, self.num_classes)


	def forward(self, sents, lengths):
		"""
		:param sents: batch x seq_len x d_word_vec
		:param lengths: numpy array 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)
		w_embed = self.embeddings(sents)

		# utt encoder
		s_utt = self.utt_cnn(w_embed)      # batch x 100
		s_utt = self.dropout_in(s_utt)

		s_cont  = self.cont_lstm(s_utt.unsqueeze(1))[0].squeeze(1)
		s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, 0



# CMN
class CMN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(CMN, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops       # default 1
		self.wind_1 = args.wind1    # default 20

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3,4,5])
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=False)
		self.hidden_state = self.init_hidden(1)                                 # do this step by step

		# memory bank
		if self.hops > 1:
			self.mem_gru = nn.GRU(100, 100, num_layers=1, bidirectional=False)
		self.dropout_mid = nn.Dropout(0.3)

		# classifier
		self.d_lin_2 = 100
		self.classifier = nn.Linear(self.d_lin_2, self.num_classes)

	def init_hidden(self, batch_size):
		# variable of size [num_layers*num_directions, b_sz, hidden_sz]
		return Variable(torch.zeros(1, batch_size, 100))

	def forward(self, sents, lengths):
		"""
		:param sents: batch x seq_len x d_word_vec
		:param lengths: numpy array 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)
		w_embed = self.embeddings(sents)

		# utt encoder
		s_utt = self.utt_cnn(w_embed)      # batch x 100
		s_utt = self.dropout_in(s_utt)

		s_gru = []                         # all hidden states
		s_mem = []                         # K memory
		s_out = []                         # all output
		cont_inp = s_utt.unsqueeze(1)      # batch x 1 x 100
		hidden = self.hidden_state.to(sents.device)
		s_out.append(cont_inp[0:1])                 # the first utterance has no history
		for i in range(1, sents.size()[0]):
			u = cont_inp[i - 1:i]                   # last utterance
			cont_out, hidden = self.cont_gru(u, hidden)
			s_gru.append(hidden)
			if i < self.wind_1 + 1:
				s_mem.append(hidden)
			else:
				s_mem = s_mem[1:] + [hidden]

			query = cont_inp[i:i + 1]                                           # 1 x 1 x 100
			mem_bank = torch.cat(s_mem, dim=0).transpose(0, 1).contiguous()      # 1 x batch x 100
			for hop in range(self.hops):
				mem_bank = self.dropout_mid(mem_bank)                           # dropout
				attn_out, attn_weight = dotprod_attention(query, mem_bank, mem_bank)
				query = F.tanh(query + attn_out)

				if self.hops > 1:
					mem_inp = mem_bank.transpose(0, 1).contiguous()
					mem_update = self.mem_gru(mem_inp)[0]                       # batch x 1 x 100
					mem_bank = mem_update.transpose(0, 1).contiguous()
			s_out.append(query)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, 0


