#!bin/bash
# Var assignment
LR=5e-4
GPU=0
TP=UniF_Att
du=100
dc=100
w1=10
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5 6 7 8 9 0
do
echo --- $Enc - $Dec $iter ---
#python EmoMain.py -epochs 60 -lr $LR -gpu $GPU -type $TP -hops 1 -wind1 $w1 -d_h1 $du -d_h2 $dc -report_loss 96 -data_path IEMOCAP6_data.pt -vocab_path IEMOCAP6_vocab.pt -emodict_path IEMOCAP6_emodict.pt -tr_emodict_path IEMOCAP6_tr_emodict.pt -dataset IEMOCAP6 -embedding IEMOCAP6_embedding.pt
python EmoMain.py -epochs 20 -lr $LR -gpu $GPU -type $TP -hops 1 -wind1 $w1 -d_h1 $du -d_h2 $dc -report_loss 1038 -data_path MELD_data.pt -vocab_path MELD_vocab.pt -emodict_path MELD_emodict.pt -tr_emodict_path MELD_tr_emodict.pt -dataset MELD -embedding MELD_embedding.pt
done