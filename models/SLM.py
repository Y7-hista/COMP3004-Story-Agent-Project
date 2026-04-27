# Statistical language models

import nltk
import random
import string

from nltk import sent_tokenize
from nltk import word_tokenize

from nltk.util import ngrams
from nltk.probability import FreqDist

from nltk.lm import MLE
from nltk.lm import Laplace
from nltk.lm import KneserNeyInterpolated

from nltk.lm.preprocessing import padded_everygram_pipeline
from models.ngram_model import NGramModel
import pickle
import os

# Run for the first time:
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class StatisticalLanguageModel:
    def __init__(self, n = 4, smoothing = "laplace"):
        self.n = n
        self.smoothing = smoothing
        self.model = None
        # self.model = NGramModel(n = n, smoothing=smoothing)
        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        self.fourgrams = []
        self.tokenized_text = []

    def preprocess(self, text):
        print("Preprocessing")
        punctuation = string.punctuation + "'" + "-" + "'" + "-"
        punctuation = punctuation.replace(".", "")
        text = text.lower()

        for p in punctuation:
            if p not in ".?!":
                text = text.replace(p, " ")
        sentences = sent_tokenize(text)
        processed = []

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            tokens = [
                t for t in tokens
                    if t not in [".", "!", "?"]
            ]
            if len(tokens) > 0:
                processed.append(tokens)
        return processed
    
    def build_grams(self):
        for seq in self.tokenized_text:
            self.unigrams.extend(seq)
            self.bigrams.extend(list(ngrams(seq, 2)))
            self.trigrams.extend(list(ngrams(seq, 3)))
            self.fourgrams.extend(list(ngrams(seq, 4)))
        freq_uni = nltk.FreqDist(self.unigrams)
        freq_bi = nltk.FreqDist(self.bigrams)
        freq_tri = nltk.FreqDist(self.trigrams)
        freq_four = nltk.FreqDist(self.fourgrams)
        print("5 most common unigrams:" + str(freq_uni.most_common(5)))
        print("5 most common bigrams: " + str(freq_bi.most_common(5)))
        print("5 most common trigrams: " + str(freq_tri.most_common(5)))
        print("5 most common fourgrams: " + str(freq_four.most_common(5)))
    
    def build_lm(self):
        print("Building Language Models.")
        # train_data = self.tokenized_text
        # vocab = list(set(train_data))

        # self.model.train(train_data)
        train_data, vocab = padded_everygram_pipeline(self.n, self.tokenized_text)

        if self.smoothing == "mle":
            self.model = MLE(self.n)
        elif self.smoothing == "laplace":
            self.model = Laplace(self.n)
        # elif self.smoothing == "kneserney":
        #     self.model = KneserNeyInterpolated(self.n, discount=0.1)
        else:
            raise ValueError("unknown smoothing")
        
        self.model.fit(train_data, vocab)
        print(f"{self.smoothing} model built")

    def train(self, text, model_path = "saved_models/SLM_model.pkl"):
        if os.path.exists(model_path):
            print("Loading saved model...")
            # loaded = StatisticalLanguageModel.load_model(model_path)
            # self.__dict__.update(loaded.__dict__)
            self.model = StatisticalLanguageModel.load_model(model_path)
            return
        
        self.tokenized_text = self.preprocess(text)
        self.build_grams()
        self.build_lm()

        self.save_model(model_path)
        print("Model saved.")

    # LM Function
    def sentence_probability(self, sentences):
        tokens = word_tokenize(sentences.lower())
        probability = 1
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - (self.n - 1) : i])
            word = tokens[i]
            p = self.model.score(word, context)
            probability *= p
        return probability

    # Test model perplexity
    def perplexity(self, test_sentences):
        test=[word_tokenize(test_sentences.lower())]
        test_grams, _= padded_everygram_pipeline(self.n, test)
        test_grams = list(test_grams)[0]

        return self.model.perplexity(test_grams)

    # next word prediction
    def next_word(self, context):
        context = tuple(word_tokenize(context.lower())[-(self.n - 1) : ])

        vocab = list(self.model.vocab)
        scored=[]

        for w in vocab:
            p = self.model.score(w, list(context))
            scored.append((w, p))

        scored = sorted(scored, key = lambda x:x[1], reverse = True)

        return scored[:10]

    def thematic_seed(self, keywords):
        """
        Search promots input in the dataset
        """
        candidates=[]
        keyword_set = set(
            k.lower()
            for k in keywords
        )

        for sentence in self.tokenized_text:
            overlap = len(keyword_set & set(sentence))

            if overlap > 0:
                candidates.append((overlap, sentence))

        if not candidates:
            return ["once","upon","a","time"]

        # overlap越高越好
        candidates.sort(key = lambda x : x[0], reverse = True)

        top = candidates[ : 20]
        chosen = random.choice(top)[1]

        return chosen[:min(8, len(chosen))]

    # Generating Language
    def generate(self, keywords, num_sentences = 6, max_sentence_len = 20):
        # 初始prompt
        context=list(keywords)
        story=[]

        for i in range(num_sentences):
            sentence=[]
            while len(sentence)<max_sentence_len:
                # keyword injection
                if random.random()<0.15 and len(keywords)>0:
                    generated=[random.choice(keywords)]
                else:
                    try:
                        generated=self.model.generate(num_words=4, text_seed=context)
                    except:
                        break

                    if not isinstance(generated, list):
                        generated=[generated]

                for tok in generated:
                    if tok in ["<s>","</s>"]:
                        break
                    sentence.append(tok)
                    context.append(tok)
                    context=context[-(self.n - 1):]

                    if len(sentence)>=max_sentence_len:
                        break

                if len(sentence)>6 and random.random()<0.35:
                    break

            if sentence:
                text=" ".join(sentence)
                if text[-1] not in ".!?":
                    text+="."

                story.append(text.capitalize())
        # context = list(keywords)
        # story = []

        # for s in range(num_sentences):
        #     sentence = []
        #     for i in range(max_sentence_len):
        #         if (random.random() < 0.18 and len(keywords) > 0):
        #             next_token = random.choice(keywords)
        #         else:
        #             next_token = self.model.generate(num_words = 1, text_seed = context)

        #             # nltk often returns list
        #             if isinstance(next_token, list):
        #                 next_token = next_token[0]

        #         if next_token in ["<s>", "</s>"]:
        #             break

        #         sentence.append(next_token)
        #         context.append(next_token)
        #         context = context[-(self.n - 1):]

        #         # probabilistic sentence stop
        #         if (i > 8 and random.random() < 0.15):
        #             break

        #     if len(sentence) > 0:
        #         if sentence[-1] not in [".", "!", "?"]:
        #             sentence.append(".")

        #         sentence = " ".join(sentence)
        #         story.append(sentence.capitalize())

        return " ".join(story)

    # compare models
    def compare_smoothing(self, sentence):
        for m in ["mle", "laplace"]:
            tmp = StatisticalLanguageModel(n = self.n, smoothing = m)
            flat=" ".join(
                [
                 " ".join(x)
                 for x in self.tokenized_text
                ]
            )
            tmp.train(flat)

            print(m, tmp.perplexity(sentence))

    def save_model(self,path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)
