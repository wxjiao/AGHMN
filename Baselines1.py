""" BiAtt flow, LSTM """
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import Const


# Handle variable lengths
class GRUencoder(nn.Module):
	def __init__(self, d_emb, d_out, num_layers):
		super(GRUencoder, self).__init__()
		# default encoder 2 layers
		self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
		                  bidirectional=True, num_layers=num_layers, dropout=0.3)

	def forward(self, sent, sent_lens):
		"""
		:param sent: torch tensor, batch_size x seq_len x d_emb
		:param sent_lens: numpy tensor, batch_size x 1
		:return:
		"""
		device = sent.device
		# seq_len x batch_size x d_rnn_in
		sent_embs = sent.transpose(0,1)

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
		output = sent_output.transpose(0,1)

		return output


# BiLSTM
class BiGRU(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(BiGRU, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding

		# utt encoder
		self.utt_gru = GRUencoder(args.d_word_vec, args.d_h1, num_layers=1)
		self.d_lin_1 = args.d_h1 * 2
		self.lin_1 = nn.Sequential(
			nn.Linear(self.d_lin_1, 100),
			nn.Tanh()
		)
		self.dropout_in = nn.Dropout(0.3)

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
		w_gru = self.utt_gru(w_embed, lengths)      # batch x 2*d_h1
		maxpl = torch.max(w_gru, dim=1)[0]
		s_utt = self.lin_1(maxpl)                   # batch x d_h1
		s_utt = self.dropout_in(s_utt)

		s_output = self.classifier(s_utt)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, 0


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
class CNN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(CNN, self).__init__()
		self.num_classes = emodict.n_words
		self.embeddings = embedding

		# utt encoder
		self.utt_cnn = CNNencoder(args.d_word_vec, 64, 100, [3,4,5])
		self.dropout_in = nn.Dropout(0.3)

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

		s_output = self.classifier(s_utt)
		pred_s = F.log_softmax(s_output, dim=1)

		return pred_s, 0