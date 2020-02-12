"""
Train on Emotion dataset
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Utils
import math
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def emotrain(model, data_loader, tr_emodict, emodict, args, focus_emo):
	"""
	:data_loader input the whole field
	"""
	# start time
	time_st = time.time()
	decay_rate = args.decay

	# dataloaders
	train_loader = data_loader['train']
	dev_loader = data_loader['dev']
	feats, labels = train_loader['feat'], train_loader['label']

	lr = args.lr
	model_opt = optim.Adam(model.parameters(), lr=lr)

	# weight for loss
	weight_rate = 0.0
	if args.dataset in ['MELD']:
		weight_rate = 0.5
	weights = torch.from_numpy(loss_weight(tr_emodict, emodict, focus_emo, rate=weight_rate)).float()
	print("Dataset {} Weight rate {} \nEmotion rates {} \nLoss weights {}\n".format(
		args.dataset, weight_rate, emodict.word2count, weights))

	model.train()

	over_fitting = 0
	cur_best = -1e10
	glob_steps = 0
	report_loss = 0
	for epoch in range(1, args.epochs + 1):
		feats, labels = Utils.shuffle_lists(feats, labels)
		print("===========Epoch==============")
		print("-{}-{}".format(epoch, Utils.timeSince(time_st)))
		for bz in range(len(labels)):
			# tensorize a dialog list
			feat, lens = Utils.ToTensor(feats[bz], is_len=True)
			label = Utils.ToTensor(labels[bz])
			feat = Variable(feat)
			label = Variable(label)

			if args.gpu != None:
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				model.cuda(device)
				feat = feat.cuda(device)
				label = label.cuda(device)
				weights = weights.cuda(device)

			log_probs = model(feat, lens)[0]
			#print(log_prob, label)
			loss = comput_class_loss(log_probs, label, weights)
			loss.backward()
			report_loss += loss.item()
			glob_steps += 1

			# gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.maxnorm)
			model_opt.step()
			model_opt.zero_grad()

			if glob_steps % args.report_loss == 0:
				print("Steps: {} Loss: {} LR: {}".format(glob_steps, report_loss/args.report_loss, model_opt.param_groups[0]['lr']))
				report_loss = 0

		# validate
		Recall, Precision, F1, Avgs, Val_loss = emoeval(model=model, data_loader=dev_loader, tr_emodict=tr_emodict, emodict=emodict, args=args, focus_emo=focus_emo)
		print("Validate: val_loss {}\n re {}\n pr {}\n F1 {}\n Av {}\n".format(Val_loss, Recall, Precision, F1, Avgs))

		last_best = Avgs[0]  # WF1
		if last_best >= cur_best:
			Utils.model_saver(model, args.save_dir, args.type, args.dataset)
			cur_best = last_best
			over_fitting = 0
		else:
			over_fitting += 1
			model_opt.param_groups[0]['lr'] *= decay_rate

		# if over_fitting >= args.patience:
		# 	break



def comput_class_loss(log_prob, target, weights):
	""" classification loss """
	loss = F.nll_loss(log_prob, target.view(target.size(0)), weight=weights, reduction='sum')
	loss /= target.size(0)

	return loss


def relabel(label, emodict, focus_dict):
	""" calculate the overall label """
	labels = [[sum(label[label==i].size()) if emodict.index2word[i] in focus_dict else 0 for i in range(emodict.n_words)]]
	labels = np.array(labels, dtype=np.float)
	if np.sum(labels) != 0:
		labels /= np.sum(labels)
	labelc = torch.tensor(labels).float()

	return labelc.to(label.device)


def comput_over_loss(log_prob, target):
	""" classification loss """
	loss = target.view(log_prob.size()) * log_prob     # no weight so use mean
	loss = -torch.sum(loss) / log_prob.size()[1]

	return loss


def loss_weight(tr_ladict, ladict, focus_dict, rate=1.0):
	""" loss weights """
	min_emo = float(min([tr_ladict.word2count[w] for w in focus_dict]))
	weight = [math.pow(min_emo / tr_ladict.word2count[k], rate) if k in focus_dict
	          else 0 for k,v in ladict.word2count.items()]
	weight = np.array(weight)
	weight /= np.sum(weight)

	return weight


def emoeval(model, data_loader, tr_emodict, emodict, args, focus_emo):
	""" data_loader only input 'dev' """
	model.eval()

	# weight for loss
	weight_rate = 0
	weights = torch.from_numpy(loss_weight(tr_emodict, emodict, focus_emo, rate=weight_rate)).float()

	TP = np.zeros([emodict.n_words], dtype=np.long) # recall
	TP_FN = np.zeros([emodict.n_words], dtype=np.long) # gold
	TP_FP = np.zeros([emodict.n_words], dtype=np.long)
	focus_idx = [emodict.word2index[emo] for emo in focus_emo]

	feats, labels = data_loader['feat'], data_loader['label']
	val_loss = 0
	for bz in range(len(labels)):
		feat, lens = Utils.ToTensor(feats[bz], is_len=True)
		label = Utils.ToTensor(labels[bz])
		feat = Variable(feat)
		label = Variable(label)

		if args.gpu != None:
			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
			device = torch.device("cuda: 0")
			model.cuda(device)
			feat = feat.cuda(device)
			label = label.cuda(device)
			weights = weights.cuda(device)

		log_probs = model(feat, lens)[0]
		# val loss
		loss = comput_class_loss(log_probs, label, weights)
		val_loss += loss.item()

		# accuracy
		emo_predidx = torch.argmax(log_probs, dim=1)
		emo_true = label.view(label.size(0))

		for lb in range(emo_true.size(0)):
			idx = emo_true[lb].item()
			TP_FN[idx] += 1
			if idx in focus_idx:
				if emo_true[lb] == emo_predidx[lb]:
					TP[idx] += 1
				if emo_predidx[lb] in focus_idx:
					TP_FP[emo_predidx[lb]] += 1

	f_TP = [TP[emodict.word2index[w]] for w in focus_emo]
	f_TP_FN = [TP_FN[emodict.word2index[w]] for w in focus_emo]
	f_TP_FP = [TP_FP[emodict.word2index[w]] for w in focus_emo]
	Recall = [np.round(tp/tp_fn*100, 2) if tp_fn>0 else 0 for tp,tp_fn in zip(f_TP,f_TP_FN)]
	Precision = [np.round(tp/tp_fp*100, 2) if tp_fp>0 else 0 for tp,tp_fp in zip(f_TP,f_TP_FP)]
	F1 = [np.round(2*r*p/(r+p),2) if r+p>0 else 0 for r,p in zip(Recall,Precision)]

	wRecall = sum([r * w / sum(f_TP_FN) for r,w in zip(Recall, f_TP_FN)])
	uRecall = sum(Recall) / len(Recall)
	wPrecision = sum([p * w / sum(f_TP_FN) for p,w in zip(Precision, f_TP_FN)])
	uPrecision = sum(Precision) / len(Precision)
	wF1 = sum([f1 * w / sum(f_TP_FN) for f1,w in zip(F1, f_TP_FN)])
	uF1 = sum(F1) / len(F1)

	Avgs = [np.round(wRecall,2), np.round(uRecall,2), np.round(wPrecision,2), np.round(uPrecision,2), np.round(wF1,2), np.round(uF1,2)]

	model.train()

	return Recall, Precision, F1, Avgs, val_loss/len(labels)


# 1-31-36, 14-27-
def visualization(model, data_loader, args, bz=1, uz=30, figname='Attn.png'):
	model.eval()

	feats, labels = data_loader['feat'], data_loader['label']
	#print(labels)

	feat, lens = Utils.ToTensor(feats[bz-1], is_len=True)
	feat = Variable(feat)
	label = Utils.ToTensor(labels[bz-1])

	if args.gpu != None:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
		device = torch.device("cuda: 0")
		model.cuda(device)
		feat = feat.cuda(device)

	log_probs, attn_weights = model(feat, lens)             #[hop1, hop2, ..]
	preds = torch.argmax(log_probs, dim=1).cpu().detach().numpy()

	attn_uz = [attn_weights[hop][uz-2:uz-1] for hop in range(len(attn_weights))]
	fig, axes = plt.subplots(nrows=len(attn_uz), ncols=1)
	for hop in range(len(attn_uz)):
		ax = axes.flat[hop] if len(attn_uz) > 1 else axes
		cax = ax.matshow(attn_uz[hop].cpu().detach().numpy(), cmap='YlOrRd')          #'YlOrRd', 'bwr'

	if len(attn_uz) > 1:
		fig.colorbar(cax, ax=axes.ravel().tolist())
	else:
		fig.colorbar(cax)
	#plt.savefig(figname)

	# save .mat file
	attn_data = {}
	attn_data['test_conv'] = bz
	attn_data['conv_len'] = len(label)
	#print(len(label))
	attn_data['test_utt'] = uz
	attn_data['labels'] = label.cpu().detach().numpy()
	attn_data['preds'] = preds
	for hop in range(len(attn_uz)):
		attn_data['attn_hop_'+str(hop)] = attn_uz[hop].cpu().detach().numpy()
	sio.savemat('attn_conv{}_utt{}.mat'.format(bz, uz), attn_data)

	return 1


def Casestudy(model, data_loader, emodict, args, path='Data'):
	model.eval()

	feats, labels = data_loader['feat'], data_loader['label']
	label_preds = []
	for bz in range(len(labels)):
		feat, lens = Utils.ToTensor(feats[bz], is_len=True)
		label = Utils.ToTensor(labels[bz])
		feat = Variable(feat)
		label = Variable(label)

		if args.gpu != None:
			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
			device = torch.device("cuda: 0")
			model.cuda(device)
			feat = feat.cuda(device)
			label = label.cuda(device)

		log_probs = model(feat, lens)[0]

		emo_pred = torch.argmax(log_probs, dim=1)
		emo_true = label.view(label.size(0))

		label_pred = []
		for lb in range(emo_true.size(0)):
			true_idx = emo_true[lb].item()
			pred_idx = emo_pred[lb].item()
			label_pred.append((emodict.index2word[true_idx], emodict.index2word[pred_idx]))
		label_preds.append(label_pred)

		Utils.saveToJson(path + '_Case.json', label_preds)

	return 1


