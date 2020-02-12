"""
main function for wordCNN or +charCNN training
"""
import os
import argparse
import Utils
import Const
from Preprocess import Dictionary # import the object for pickle loading
from Baselines1 import *
from Baselines2 import *
from Modules_part1 import *
from Modules_part2 import UniF_BiAGRU_CNN,UniF_BiAGRU_LSTM,BiF_AGRU_CNN,BiF_AGRU_LSTM,UniF_Att_CNN,BiF_Att_CNN
from EmoTrain import emotrain, emoeval, visualization, Casestudy
from datetime import datetime


def main():
	'''Main function'''

	parser = argparse.ArgumentParser()

	# learning
	parser.add_argument('-lr', type=float, default=5e-4) # 2.5e-4 for Friends and EmotionPush, 1e-4 for IEMOCAP
	parser.add_argument('-decay', type=float, default=0.95)
	parser.add_argument('-maxnorm', type=float, default=5)
	parser.add_argument('-epochs', type=int, default=60)
	parser.add_argument('-batch_size', type=int, default=1)
	parser.add_argument('-patience', type=int, default=10,
	                    help='patience for early stopping')
	parser.add_argument('-save_dir', type=str, default="snapshot",
	                    help='where to save the models')
	# data
	parser.add_argument('-dataset', type=str, default='Friends',
	                    help='dataset')
	parser.add_argument('-data_path', type=str, required = True,
	                    help='data path')
	parser.add_argument('-vocab_path', type=str, required=True,
	                    help='global vocabulary path')
	parser.add_argument('-emodict_path', type=str, required=True,
	                    help='emotion label dict path')
	parser.add_argument('-tr_emodict_path', type=str, default=None,
	                    help='training set emodict path')
	parser.add_argument('-max_seq_len', type=int, default=60,
	                    help='the sequence length')
	# model
	parser.add_argument('-type', type=str, default='gru',
	                    help='the model type: gru')
	parser.add_argument('-d_word_vec', type=int, default=300,
	                    help='the word embeddings size')
	parser.add_argument('-d_h1', type=int, default=100,
	                    help='the hidden size of rnn1')
	parser.add_argument('-d_h2', type=int, default=100,
	                    help='the hidden size of rnn2')
	parser.add_argument('-hops', type=int, default=1,
	                    help='the number of hops')
	parser.add_argument('-d_fc', type=int, default=100,
	                    help='the size of fc')
	parser.add_argument('-wind1', type=int, default=40,
	                    help='the word-level context window')
	parser.add_argument('-gpu', type=str, default=None,
	                    help='gpu: default 0')
	parser.add_argument('-embedding', type=str, default=None,
	                    help='filename of embedding pickle')
	parser.add_argument('-report_loss', type=int, default=720,
	                    help='how many steps to report loss')

	args = parser.parse_args()
	print(args, '\n')

	# load vocabs
	print("Loading vocabulary...")
	worddict = Utils.loadFrPickle(args.vocab_path)
	print("Loading emotion label dict...")
	emodict = Utils.loadFrPickle(args.emodict_path)
	print("Loading review tr_emodict...")
	tr_emodict = Utils.loadFrPickle(args.tr_emodict_path)

	# load field
	print("Loading field...")
	field = Utils.loadFrPickle(args.data_path)
	test_loader = field['test']

	# word embedding
	print("Initializing word embeddings...")
	embedding = nn.Embedding(worddict.n_words, args.d_word_vec, padding_idx=Const.PAD)
	if args.d_word_vec == 300:
		if args.embedding != None and os.path.isfile(args.embedding):
			np_embedding = Utils.loadFrPickle(args.embedding)
		else:
			np_embedding = Utils.load_pretrain(args.d_word_vec, worddict, type='word2vec')
			Utils.saveToPickle(args.dataset + '_embedding.pt', np_embedding)
		embedding.weight.data.copy_(torch.from_numpy(np_embedding))
	embedding.weight.requires_grad = False

	# model type
	model = BiGRU(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiGRU']:
		print("Model type confirmed: BiGRU")
		model = BiGRU(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['CNN']:
		print("Model type confirmed: CNN")
		model = CNN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['cLSTM']:
		print("Model type confirmed: cLSTM")
		model = cLSTM(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['CMN']:
		print("Model type confirmed: CMN")
		model = CMN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)

	if args.type in ['UniF_Att']:
		print("Model type confirmed: UniF_Att")
		model = UniF_Att(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiF_Att']:
		print("Model type confirmed: BiF_Att")
		model = BiF_Att(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['UniF_AGRU']:
		print("Model type confirmed: UniF_AGRU")
		model = UniF_AGRU(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['UniF_BiAGRU']:
		print("Model type confirmed: UniF_BiAGRU")
		model = UniF_BiAGRU(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiF_AGRU']:
		print("Model type confirmed: BiF_AGRU")
		model = BiF_AGRU(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiF_BiAGRU']:
		print("Model type confirmed: BiF_BiAGRU")
		model = BiF_BiAGRU(emodict=emodict, worddict=worddict, embedding=embedding, args=args)

	# test utterance reader
	if args.type in ['UniF_BiAGRU_CNN']:
		print("Model type confirmed: UniF_BiAGRU_CNN")
		model = UniF_BiAGRU_CNN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiF_AGRU_CNN']:
		print("Model type confirmed: BiF_AGRU_CNN")
		model = BiF_AGRU_CNN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['UniF_BiAGRU_LSTM']:
		print("Model type confirmed: UniF_BiAGRU_LSTM")
		model = UniF_BiAGRU_LSTM(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiF_AGRU_LSTM']:
		print("Model type confirmed: UniF_Att_CNN")
		model = UniF_Att_CNN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['UniF_Att_CNN']:
		print("Model type confirmed: UniF_BiAGRU_LSTM")
		model = UniF_Att_CNN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)
	if args.type in ['BiF_Att_CNN']:
		print("Model type confirmed: BiF_Att_CNN")
		model = BiF_Att_CNN(emodict=emodict, worddict=worddict, embedding=embedding, args=args)


	# Choose focused emotions
	focus_emo = Const.five_emo
	args.decay = 0.95
	if args.dataset == 'IEMOCAP6':
		focus_emo = Const.six_iem
		args.decay = 0.95
	if args.dataset == 'MELD':
		focus_emo = Const.sev_meld
		args.decay = 0.95
	print("Focused emotion labels {}".format(focus_emo))

	emotrain(model=model,
	         data_loader=field,
	         tr_emodict=tr_emodict,
	         emodict=emodict,
	         args=args,
	         focus_emo=focus_emo)

	# test
	print("Load best models for testing!")

	model = Utils.model_loader(args.save_dir, args.type, args.dataset)
	Recall, Precision, F1, Avgs, Val_loss = emoeval(model=model,
	                                      data_loader=test_loader,
	                                      tr_emodict=tr_emodict,
	                                      emodict=emodict,
	                                      args=args,
	                                      focus_emo=focus_emo)
	print("Test: val_loss {}\n re {}\n pr {}\n F1 {}\n Av {}\n".format(Val_loss, Recall, Precision, F1, Avgs))

	# record the test results
	record_file = '{}/{}_{}.txt'.format(args.save_dir, args.type, args.dataset)
	if os.path.isfile(record_file):
		f_rec = open(record_file, "a")
	else:
		f_rec = open(record_file, "w")
	f_rec.write("{} - {}-{} - {}:\t\n \tre {}\n \tpr {}\n \tF1 {}\n \tAv {}\n\n".format(datetime.now(), args.hops, args.wind1, args.lr, Recall, Precision, F1, Avgs))
	f_rec.close()

	# Casestudy(model, test_loader, emodict, args, path=args.type)

	# # visualize
	# visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=27)
	# visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=28)
	# visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=29)
	#visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=30)
	# visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=31)
	# visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=32)
	# visualization(model=model, data_loader=test_loader, args=args, bz=12, uz=33)

	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=3)
	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=4)
	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=5)
	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=6)
	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=7)
	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=8)
	# visualization(model=model, data_loader=test_loader, args=args, bz=97, uz=9)


if __name__ == '__main__':
	main()
