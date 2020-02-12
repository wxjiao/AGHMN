""" Dynamic Memory Network 2019-05-09, work """
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.init as init
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


class AttnGRUCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AttnGRUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.Wr = nn.Linear(input_size, hidden_size)
		self.Ur = nn.Linear(hidden_size, hidden_size)
		self.W = nn.Linear(input_size, hidden_size)
		self.U = nn.Linear(hidden_size, hidden_size)

		init.xavier_normal_(self.Wr.state_dict()['weight'])
		init.xavier_normal_(self.Ur.state_dict()['weight'])
		init.xavier_normal_(self.W.state_dict()['weight'])
		init.xavier_normal_(self.U.state_dict()['weight'])

	def forward(self, c, hi_1, g):

		r_i = F.sigmoid(self.Wr(c) + self.Ur(hi_1))
		h_tilda = F.tanh(self.W(c) + r_i*self.U(hi_1))
		hi = g*h_tilda + (1 - g)*hi_1

		return hi


class AttnRNNCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AttnRNNCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.W = nn.Linear(input_size, hidden_size)
		self.U = nn.Linear(hidden_size, hidden_size)

		init.xavier_normal_(self.W.state_dict()['weight'])
		init.xavier_normal_(self.U.state_dict()['weight'])

	def forward(self, c, hi_1, g):

		h_tilda = F.tanh(self.W(c) + self.U(hi_1))
		hi = g*h_tilda + (1 - g)*hi_1

		return hi


class AttGRU(nn.Module):
	def __init__(self, d_model, rnn_type='gru', bidirectional=False):
		super(AttGRU, self).__init__()
		self.d_model = d_model
		self.bidirectional = bidirectional
		self.rnn = AttnGRUCell(100, 100) if rnn_type in ['gru'] else AttnRNNCell(100, 100)
		if self.bidirectional:
			self.rnn_bwd = AttnGRUCell(100, 100) if rnn_type in ['gru'] else AttnRNNCell(100, 100)

	def forward(self, query, context, init_hidden, attn_mask=None):
		"""
		:param q: batch x 1 x d_h
		:param c: batch x seq_len x d_h
		:param h: 1 x batch x d_h
		:return:
		"""
		attn = torch.matmul(query, context.transpose(1, 2))
		if attn_mask is not None:
			attn.data.masked_fill_(attn_mask, -1e10)
		scores = F.softmax(attn, dim=-1)                             # batch x 1 x seq_len

		# AttGRU summary
		hidden = init_hidden                                         # 1 x batch x d_h
		if self.bidirectional:
			hidden, hidden_bwd = init_hidden.chunk(2, 0)             # 2 x batch x d_h
		inp = context.transpose(0, 1).contiguous()                   # seq_len x batch x d_h
		gates = scores.transpose(1, 2).transpose(0, 1).contiguous()  # seq_len x batch x 1
		seq_len = context.size()[1]
		for i in range(seq_len):
			hidden = self.rnn(inp[i:i + 1], hidden, gates[i:i + 1])
			if self.bidirectional:
				hidden_bwd = self.rnn_bwd(inp[seq_len - i - 1:seq_len - i], hidden_bwd, gates[seq_len - i - 1:seq_len - i])

		output = hidden.transpose(0, 1).contiguous()  # batch x 1 x d_h
		if self.bidirectional:
			output = torch.cat([hidden, hidden_bwd], dim=-1).transpose(0, 1).contiguous()  # batch x 1 x d_h*2

		return output, scores


def get_attn_pad_mask(seq_q, seq_k):
	assert seq_q.dim() == 2 and seq_k.dim() == 2

	pad_attn_mask = torch.matmul(seq_q.unsqueeze(2).float(), seq_k.unsqueeze(1).float())
	pad_attn_mask = pad_attn_mask.eq(Const.PAD)  # b_size x 1 x len_k
	#print(pad_attn_mask)

	return pad_attn_mask.cuda(seq_k.device)


# CNN utterance reader
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


# Handle variable lengths
class LSTMencoder(nn.Module):
	def __init__(self, d_emb, d_out, num_layers):
		super(LSTMencoder, self).__init__()
		# default encoder 2 layers
		self.gru = nn.LSTM(input_size=d_emb, hidden_size=d_out,
						   bidirectional=True, num_layers=num_layers, dropout=0.3)

	def forward(self, sent, sent_lens):
		"""
		:param sent: torch tensor, batch_size x seq_len x d_emb
		:param sent_lens: numpy tensor, batch_size x 1
		:return:
		"""
		device = sent.device
		# seq_len x batch_size x d_rnn_in
		sent_embs = sent.transpose(0, 1)

		# sort by length
		s_lens, idx_sort = np.sort(sent_lens)[::-1], np.argsort(-sent_lens)
		idx_unsort = np.argsort(idx_sort)

		idx_sort = torch.from_numpy(idx_sort).cuda(device)
		s_embs = sent_embs.index_select(1, Variable(idx_sort))

		# padding
		sent_packed = pack_padded_sequence(s_embs, s_lens)
		sent_output = self.gru(sent_packed)[0]
		sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

		# unsort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))

		# batch x seq_len x 2*d_out
		output = sent_output.transpose(0, 1)

		return output


# BiF + Uni-AttGRU
class BiF_AGRU_LSTM(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(BiF_AGRU_LSTM, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops  # default 1
		self.wind_1 = args.wind1  # default 20

		# utt encoder
		# utt encoder
		self.utt_gru = LSTMencoder(args.d_word_vec, args.d_h1, num_layers=1)
		self.d_lin_1 = args.d_h1 * 2
		self.lin_1 = nn.Linear(self.d_lin_1, 100)
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=True)
		self.dropout_mid = nn.Dropout(0.3)

		# AttGRU
		self.AttGRU = nn.ModuleList(
			AttGRU(d_model=100) for hop in range(self.hops)
		)

		# classifier
		self.classifier = nn.Linear(100, self.num_classes)

	def init_hidden(self, num_directs, num_layers, batch_size, d_model):
		return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

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
		w_gru = self.utt_gru(w_embed, lengths)      # batch x 2*d_h1
		maxpl = torch.max(w_gru, dim=1)[0]
		s_utt = F.tanh(self.lin_1(maxpl))                 # batch x d_h1
		s_utt = self.dropout_in(s_utt)

		s_out = []  # all output
		cont_inp = s_utt.unsqueeze(1)    # batch x 1 x 100
		s_out.append(cont_inp[:1])       # the first utterance, no memory
		attn_weights = []

		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				#print(m_pad.size())
				batches.append(m_pad)
				mask = [0] * pad + [1] * (self.wind_1 - pad)
				#print(mask)
				masks.append(mask)
			batches_tensor = torch.cat(batches, dim=1)                          # K x (batch - 1) x 100
			masks_tensor = torch.tensor(masks).long().to(sents.device)          # (batch -1) x K
			query_mask = torch.ones(masks_tensor.size()[0], 1).long().to(sents.device)      # (batch - 1) x 1
			attn_mask = get_attn_pad_mask(query_mask, masks_tensor)                         # (batch - 1) x 1 x K

			# memory
			mem_out = self.cont_gru(batches_tensor)[0]                    # K x (batch - 1) x 200
			mem_fwd, mem_bwd = mem_out.chunk(2, -1)
			mem_bank = (batches_tensor + mem_fwd + mem_bwd).transpose(0, 1).contiguous()          # (batch - 1) x K x 100
			mem_bank = self.dropout_mid(mem_bank)

			# query
			query = cont_inp[1:]                                                # (batch - 1) x 1 x 100

			# multi-hops
			eps_mem = query
			for hop in range(self.hops):
				attn_hid = self.init_hidden(1, 1, masks_tensor.size()[0], 100).to(sents.device)
				attn_out, attn_weight = self.AttGRU[hop](eps_mem, mem_bank, attn_hid, attn_mask)
				attn_weights.append(attn_weight.squeeze(1))
				eps_mem = eps_mem + attn_out
				eps_mem = self.dropout_mid(eps_mem)
			s_out.append(eps_mem)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		#s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, attn_weights


# UniF + Bi-AttGRU
class UniF_BiAGRU_LSTM(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(UniF_BiAGRU_LSTM, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops  # default 1
		self.wind_1 = args.wind1  # default 20

		# utt encoder
		self.utt_gru = LSTMencoder(args.d_word_vec, args.d_h1, num_layers=1)
		self.d_lin_1 = args.d_h1 * 2
		self.lin_1 = nn.Linear(self.d_lin_1, 100)
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=False)
		self.dropout_mid = nn.Dropout(0.3)

		# AttGRU
		self.AttGRU = nn.ModuleList(
			AttGRU(d_model=100, bidirectional=True) for hop in range(self.hops)
		)

		# classifier
		self.classifier = nn.Linear(100, self.num_classes)

	def init_hidden(self, num_directs, num_layers, batch_size, d_model):
		return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

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
		w_gru = self.utt_gru(w_embed, lengths)              # batch x 2*d_h1
		maxpl = torch.max(w_gru, dim=1)[0]
		s_utt = F.tanh(self.lin_1(maxpl))                   # batch x d_h1
		s_utt = self.dropout_in(s_utt)

		s_out = []  # all output
		cont_inp = s_utt.unsqueeze(1)    # batch x 1 x 100
		s_out.append(cont_inp[:1])       # the first utterance, no memory
		attn_weights = []

		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				#print(m_pad.size())
				batches.append(m_pad)
				mask = [0] * pad + [1] * (self.wind_1 - pad)
				#print(mask)
				masks.append(mask)
			batches_tensor = torch.cat(batches, dim=1)                          # K x (batch - 1) x 100
			masks_tensor = torch.tensor(masks).long().to(sents.device)          # (batch -1) x K
			query_mask = torch.ones(masks_tensor.size()[0], 1).long().to(sents.device)      # (batch - 1) x 1
			attn_mask = get_attn_pad_mask(query_mask, masks_tensor)                         # (batch - 1) x 1 x K

			# memory
			mem_out = self.cont_gru(batches_tensor)[0]        # K x (batch - 1) x 200
			mem_bank = (batches_tensor + mem_out).transpose(0, 1).contiguous()         # (batch - 1) x K x 100
			mem_bank = self.dropout_mid(mem_bank)

			# query
			query = cont_inp[1:]                                                # (batch - 1) x 1 x 100

			# multi-hops
			eps_mem = query
			for hop in range(self.hops):
				attn_hid = self.init_hidden(2, 1, masks_tensor.size()[0], 100).to(sents.device)
				attn_out, attn_weight = self.AttGRU[hop](eps_mem, mem_bank, attn_hid, attn_mask)
				attn_weights.append(attn_weight.squeeze(1))
				attn_out_fwd, attn_out_bwd = attn_out.chunk(2, -1)
				eps_mem = eps_mem + attn_out_fwd + attn_out_bwd
				eps_mem = self.dropout_mid(eps_mem)
			s_out.append(eps_mem)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		#s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, attn_weights


# BiF + Uni-AttGRU
class BiF_AGRU_CNN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(BiF_AGRU_CNN, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops  # default 1
		self.wind_1 = args.wind1  # default 20

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3,4,5])
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=True)
		self.dropout_mid = nn.Dropout(0.3)

		# AttGRU
		self.AttGRU = nn.ModuleList(
			AttGRU(d_model=100) for hop in range(self.hops)
		)

		# classifier
		self.classifier = nn.Linear(100, self.num_classes)

	def init_hidden(self, num_directs, num_layers, batch_size, d_model):
		return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

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

		s_out = []  # all output
		cont_inp = s_utt.unsqueeze(1)    # batch x 1 x 100
		s_out.append(cont_inp[:1])       # the first utterance, no memory
		attn_weights = []

		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				#print(m_pad.size())
				batches.append(m_pad)
				mask = [0] * pad + [1] * (self.wind_1 - pad)
				#print(mask)
				masks.append(mask)
			batches_tensor = torch.cat(batches, dim=1)                          # K x (batch - 1) x 100
			masks_tensor = torch.tensor(masks).long().to(sents.device)          # (batch -1) x K
			query_mask = torch.ones(masks_tensor.size()[0], 1).long().to(sents.device)      # (batch - 1) x 1
			attn_mask = get_attn_pad_mask(query_mask, masks_tensor)                         # (batch - 1) x 1 x K

			# memory
			mem_out = self.cont_gru(batches_tensor)[0]                    # K x (batch - 1) x 200
			mem_fwd, mem_bwd = mem_out.chunk(2, -1)
			mem_bank = (batches_tensor + mem_fwd + mem_bwd).transpose(0, 1).contiguous()          # (batch - 1) x K x 100
			mem_bank = self.dropout_mid(mem_bank)

			# query
			query = cont_inp[1:]                                                # (batch - 1) x 1 x 100

			# multi-hops
			eps_mem = query
			for hop in range(self.hops):
				attn_hid = self.init_hidden(1, 1, masks_tensor.size()[0], 100).to(sents.device)
				attn_out, attn_weight = self.AttGRU[hop](eps_mem, mem_bank, attn_hid, attn_mask)
				attn_weights.append(attn_weight.squeeze(1))
				eps_mem = eps_mem + attn_out
				eps_mem = self.dropout_mid(eps_mem)
			s_out.append(eps_mem)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		#s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, attn_weights


# UniF + Bi-AttGRU
class UniF_BiAGRU_CNN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(UniF_BiAGRU_CNN, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops  # default 1
		self.wind_1 = args.wind1  # default 20

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3,4,5])
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=False)
		self.dropout_mid = nn.Dropout(0.3)

		# AttGRU
		self.AttGRU = nn.ModuleList(
			AttGRU(d_model=100, bidirectional=True) for hop in range(self.hops)
		)

		# classifier
		self.classifier = nn.Linear(100, self.num_classes)

	def init_hidden(self, num_directs, num_layers, batch_size, d_model):
		return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

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

		s_out = []  # all output
		cont_inp = s_utt.unsqueeze(1)    # batch x 1 x 100
		s_out.append(cont_inp[:1])       # the first utterance, no memory
		attn_weights = []

		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				#print(m_pad.size())
				batches.append(m_pad)
				mask = [0] * pad + [1] * (self.wind_1 - pad)
				#print(mask)
				masks.append(mask)
			batches_tensor = torch.cat(batches, dim=1)                          # K x (batch - 1) x 100
			masks_tensor = torch.tensor(masks).long().to(sents.device)          # (batch -1) x K
			query_mask = torch.ones(masks_tensor.size()[0], 1).long().to(sents.device)      # (batch - 1) x 1
			attn_mask = get_attn_pad_mask(query_mask, masks_tensor)                         # (batch - 1) x 1 x K

			# memory
			mem_out = self.cont_gru(batches_tensor)[0]        # K x (batch - 1) x 200
			mem_bank = (batches_tensor + mem_out).transpose(0, 1).contiguous()         # (batch - 1) x K x 100
			mem_bank = self.dropout_mid(mem_bank)

			# query
			query = cont_inp[1:]                                                # (batch - 1) x 1 x 100

			# multi-hops
			eps_mem = query
			for hop in range(self.hops):
				attn_hid = self.init_hidden(2, 1, masks_tensor.size()[0], 100).to(sents.device)
				attn_out, attn_weight = self.AttGRU[hop](eps_mem, mem_bank, attn_hid, attn_mask)
				attn_weights.append(attn_weight.squeeze(1))
				attn_out_fwd, attn_out_bwd = attn_out.chunk(2, -1)
				eps_mem = eps_mem + attn_out_fwd + attn_out_bwd
				eps_mem = self.dropout_mid(eps_mem)
			s_out.append(eps_mem)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		#s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, attn_weights



# UniF + Att + CNN
class UniF_Att_CNN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(UniF_Att_CNN, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops  # default 1
		self.wind_1 = args.wind1  # default 20

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3, 4, 5])
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=False)
		self.lin_1 = nn.Linear(200, 100)
		self.dropout_mid = nn.Dropout(0.3)

		# classifier
		self.classifier = nn.Linear(100, self.num_classes)

	def init_hidden(self, num_directs, num_layers, batch_size, d_model):
		return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

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

		s_out = []  # all output
		cont_inp = s_utt.unsqueeze(1)  # batch x 1 x 100
		s_out.append(cont_inp[:1])  # the first utterance, no memory
		attn_weights = []

		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				# print(m_pad.size())
				batches.append(m_pad)
				mask = [0] * pad + [1] * (self.wind_1 - pad)
				# print(mask)
				masks.append(mask)
			batches_tensor = torch.cat(batches, dim=1)  # K x (batch - 1) x 100
			masks_tensor = torch.tensor(masks).long().to(sents.device)  # (batch -1) x K
			query_mask = torch.ones(masks_tensor.size()[0], 1).long().to(sents.device)  # (batch - 1) x 1
			attn_mask = get_attn_pad_mask(query_mask, masks_tensor)  # (batch - 1) x 1 x K

			# memory
			mem_out = self.cont_gru(batches_tensor)[0]  # K x (batch - 1) x 200
			mem_bank = (batches_tensor + mem_out).transpose(0, 1).contiguous()  # (batch - 1) x K x 100
			mem_bank = self.dropout_mid(mem_bank)

			# query
			query = cont_inp[1:]  # (batch - 1) x 1 x 100

			# multi-hops
			eps_mem = query
			for hop in range(self.hops):
				attn_out, attn_weight = dotprod_attention(eps_mem, mem_bank, mem_bank, attn_mask)
				attn_weights.append(attn_weight.squeeze(1))
				eps_mem = eps_mem + attn_out
				eps_mem = self.dropout_mid(eps_mem)
			s_out.append(eps_mem)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		# s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, attn_weights


# BiF + Att + CNN
class BiF_Att_CNN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(BiF_Att_CNN, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding
		self.gpu = args.gpu
		# sliding window
		self.hops = args.hops  # default 1
		self.wind_1 = args.wind1  # default 20

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3, 4, 5])
		self.dropout_in = nn.Dropout(0.3)

		# context encoder
		self.cont_gru = nn.GRU(100, 100, num_layers=1, bidirectional=True)
		self.dropout_mid = nn.Dropout(0.3)

		# classifier
		self.classifier = nn.Linear(100, self.num_classes)

	def init_hidden(self, num_directs, num_layers, batch_size, d_model):
		return Variable(torch.zeros(num_directs * num_layers, batch_size, d_model))

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

		s_out = []  # all output
		cont_inp = s_utt.unsqueeze(1)  # batch x 1 x 100
		s_out.append(cont_inp[:1])  # the first utterance, no memory
		attn_weights = []

		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				# print(m_pad.size())
				batches.append(m_pad)
				mask = [0] * pad + [1] * (self.wind_1 - pad)
				# print(mask)
				masks.append(mask)
			batches_tensor = torch.cat(batches, dim=1)  # K x (batch - 1) x 100
			masks_tensor = torch.tensor(masks).long().to(sents.device)  # (batch -1) x K
			query_mask = torch.ones(masks_tensor.size()[0], 1).long().to(sents.device)  # (batch - 1) x 1
			attn_mask = get_attn_pad_mask(query_mask, masks_tensor)  # (batch - 1) x 1 x K

			# memory
			mem_out = self.cont_gru(batches_tensor)[0]  # K x (batch - 1) x 200
			mem_fwd, mem_bwd = mem_out.chunk(2, -1)
			mem_bank = (batches_tensor + mem_fwd + mem_bwd).transpose(0, 1).contiguous()  # (batch - 1) x K x 100
			mem_bank = self.dropout_mid(mem_bank)

			# query
			query = cont_inp[1:]  # (batch - 1) x 1 x 100

			# multi-hops
			eps_mem = query
			for hop in range(self.hops):
				attn_out, attn_weight = dotprod_attention(eps_mem, mem_bank, mem_bank, attn_mask)
				attn_weights.append(attn_weight.squeeze(1))
				eps_mem = eps_mem + attn_out
				eps_mem = self.dropout_mid(eps_mem)
			s_out.append(eps_mem)

		s_cont = torch.cat(s_out, dim=0).squeeze(1)
		# s_cont = self.dropout_mid(s_cont)

		s_output = self.classifier(s_cont)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, attn_weights