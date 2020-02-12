""" Constants for sequences """

# word pad and unk
PAD = 0
UNK = 1
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

# focused classes, emotion
five_emo = ['joy', 'sadness', 'neutral', 'anger', 'surprise']

sev_meld = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
"""
original:
{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
[4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]

now: [1.0, 15.0, 4.0, 3.0, 15.0, 6.0, 15.0]
"""
six_iem = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
sev_emory = ['joyful', 'sad', 'neutral', 'mad', 'peaceful', 'powerful', 'scared']
five_emory = ['joyful', 'sad', 'neutral', 'mad', 'scared']
two_mosi = ['positive', 'negative']