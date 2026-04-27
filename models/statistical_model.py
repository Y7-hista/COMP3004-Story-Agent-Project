import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize

string.punctuation = string.punctuation +"'"+"-"+"'"+"-" # adds some additional characters to the punctuation
string.punctuation = string.punctuation.replace(".", "")
file = open("dataset.txt" , encoding = "utf8").read()
file_nl_removed = ""


for line in file:
    line_nl_removed = line.replace("\n", " ") # removes newline characters
    file_nl_removed += line_nl_removed # adds filtered line to the list
    file_p = "".join([char for char in file_nl_removed if char not in string.punctuation])
    # joins all the lines in the list in a single string

from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize
unigram=[]
bigram=[]
trigram=[]
tokenized_text = []

sents = nltk.sent_tokenize(file_p)

for sentence in sents:
    sentence = sentence.lower()
    sequence = word_tokenize(sentence)
    for word in sequence:
        if word == ".":
            sequence.remove(word)
        else:
            unigram.append(word)
            tokenized_text.append(sequence)
            bigram.extend(list(ngrams(sequence, 2) ) )
            #unigram, bigram, trigram models are created
            trigram.extend(list(ngrams(sequence, 3)))

            freq_uni = nltk.FreqDist(unigram)
            freq_bi = nltk.FreqDist(bigram)
            freq_tri = nltk.FreqDist(trigram)

            print("5 most common unigrams:" + str(freq_uni.most_common(5)))
            print("5 most common bigrams: " + str(freq_bi.most_common(5)))
            print("5 most common trigrams: " + str(freq_tri.most_common(5)))