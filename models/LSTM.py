import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string, random, os
from models.model_utils import save_model, load_model
from models.topic_planner import TopicPlanner

class LSTMNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = 2

        # LSTM layer
        self.lstm = nn.LSTM(input_size = 128, hidden_size = hidden_dim, num_layers = self.num_layers, dropout = 0.3, batch_first = True)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input_seq, hidden_state):
        x = self.embedding(input_seq.long())
        output, hidden_state = self.lstm(x, hidden_state)
        output = self.fc(output)

        return output, hidden_state
    
    def init_hidden(self, batch_size):
        # The init hidden method now initialises a tuple containing both the hidden state (h0) and the cell state (c0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)
    
class LSTMModel:
    def __init__(self, sequence_length = 12, hidden_dim = 256, planner = None):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.planner = planner
        self.model = None
        self.vocab = []
        self.word_to_index = {}
        self.index_to_word = {}
        self.training_sequences = []

    def preprocess(self, text):
        print("LSTM preprocessing")
        text = text.lower()
        punctuation = string.punctuation + "'" + "-" + "'" + "-"

        for p in punctuation:
            if p not in ".?!":
                text = text.replace(p, " ")

        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, text):
        words = text.lower().split()    
        self.vocab = sorted(list(set(words)))
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        print("(LSTM) Vocabulary size:", len(self.vocab))

    def prepare_training_data(self, text, sequence_length = 2):
        words = text.lower().split()
        X_seq = []
        y_seq = []

        for i in range(len(words) - sequence_length):
            input_sequence = words[i:i + sequence_length]
            target_word = words[i + sequence_length]
            X_seq.append(input_sequence)
            y_seq.append(target_word)
        self.training_sequences = (X_seq, y_seq)
        print("(LSTM) Sequences:", len(X_seq))

    def words_to_indices(self, sequence):
        indices=[]

        for word in sequence:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
            else:
                indices.append(random.randint(0, len(self.vocab) - 1))

        return indices

    def build_LSTM(self):
        print("Build LSTM Model")
        self.model = LSTMNLM(len(self.vocab), len(self.vocab), self.hidden_dim)

    def train(self, text, epochs=10, lr=0.001, batch_size=64, model_path="saved_models/lstm.pkl"
    ):
        self.build_vocabulary(text)
        words=["<bos>"] + text.lower().split() + ["<eos>"]
        X=[]
        Y=[]

        for i in range(len(words) - self.sequence_length - 1):
            X.append(words[i:i + self.sequence_length])
            Y.append(words[i + 1:i + self.sequence_length + 1])

        if os.path.exists(model_path):
            print("Loading saved model...")
            self.model=load_model(model_path)
            return

        self.build_LSTM()

        X_idx=[self.words_to_indices(x) for x in X]
        Y_idx=[self.words_to_indices(y) for y in Y]
        X_train=torch.tensor(X_idx, dtype=torch.long)

        Y_train=torch.tensor(Y_idx, dtype=torch.long)
        optimizer=optim.Adam(self.model.parameters(), lr=lr)
        criterion=nn.CrossEntropyLoss()
        print("Training LSTM")
        best_loss=999
        patience=2
        bad_epochs=0

        for epoch in range(epochs):
            # milder scheduled sampling
            teacher_force_prob=max(0.95 - epoch * 0.015, 0.70)
            ids=list(range(X_train.size(0)))
            random.shuffle(ids)
            total_loss=0

            # train subset enough for TinyStories
            for i in ids[:5000]:
                x=X_train[i].unsqueeze(0)
                y=Y_train[i].unsqueeze(0)
                hidden=self.model.init_hidden(1)
                optimizer.zero_grad()
                inp=x[:,0].unsqueeze(1)
                outputs=[]

                for t in range(self.sequence_length):
                    out, hidden=self.model(inp, hidden)

                    outputs.append(out)
                    use_truth=(random.random() < teacher_force_prob)

                    if use_truth:
                        nxt=x[:,t].unsqueeze(1)
                    else:
                        nxt=(out[:,-1,:].argmax(-1).unsqueeze(1))
                    inp=nxt

                output=torch.cat(outputs, dim=1)
                loss=criterion(output.reshape(-1, len(self.vocab)), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss=total_loss / 5000
            print(f"(LSTM) Epoch {epoch+1}: loss {avg_loss:.4f}" )

            # EARLY STOP
            if avg_loss < best_loss:
                best_loss=avg_loss
                bad_epochs=0
            else:
                bad_epochs+=1

            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

        save_model(self.model, model_path)




    def generate_sentence_lstm(self, context_indices, target, neighbors, unused, max_len = 18):
        hidden = self.model.init_hidden(1)
        sentence = []
        with torch.no_grad():
            for i in range(max_len):
                x= torch.tensor([context_indices], dtype = torch.long)
                output, hidden = self.model(x, hidden)

                logits = output[:, -1, :].squeeze()

                # repetition penalty (not rule template)
                for w in sentence[-4:]:
                    if w in self.word_to_index:
                        logits[self.word_to_index[w]] -= 2.5

                # topic steering
                if target in self.word_to_index:
                    logits[self.word_to_index[target]] += 1.8

                if target in neighbors:
                    for n in neighbors[target][:3]:
                        if n in self.word_to_index:
                            logits[self.word_to_index[n]] += 0.7

                # encourage uncovered prompt use
                for kw in unused:
                    if kw in self.word_to_index:
                        logits[self.word_to_index[kw]] += 0.5

                # entropy-controlled sampling
                temperature=0.75
                probs=torch.softmax(logits / temperature, dim=-1)
                topk=25
                vals, ids=torch.topk(probs, topk)
                vals /= vals.sum()

                idx=ids[torch.multinomial(vals, 1)].item()
                word=self.index_to_word[idx]
                sentence.append(word)

                if word in unused:
                    unused.remove(word)

                context_indices=(context_indices[1:] + [idx])

                if (len(sentence) >= 8 and random.random() < 0.25):
                    break

        return (" ".join(sentence).capitalize() + ".", context_indices)


    def generate_story(self, keywords, max_words = 150):
        plan=self.planner.build_topic_plan(keywords)
        seed=plan["seed"]

        while len(seed)<self.sequence_length:
            seed.append(random.choice(self.vocab))

        context=self.words_to_indices(seed[:self.sequence_length])
        hidden=self.model.init_hidden(1)
        words=[]
        sentence_count=0

        with torch.no_grad():
            for step in range(max_words):
                x=torch.tensor([context], dtype=torch.long)
                out,hidden=self.model(x, hidden)
                logits=out[:,-1,:].squeeze()

                # repetition penalty
                for recent in words[-5:]:
                    if recent in self.word_to_index:
                        logits[self.word_to_index[recent]] -= 2.5

                # topic boost
                for kw in keywords:
                    if kw in self.word_to_index:
                        logits[self.word_to_index[kw]] += 1.5


                # lower temp
                probs=torch.softmax(logits / 0.75, dim = -1)

                # TOP-K ONLY
                vals,ids=torch.topk(probs, 8)
                vals=vals/vals.sum()
                idx=ids[torch.multinomial(vals, 1)].item()
                word=self.index_to_word[idx]

                if (word == "<eos>" and len(words) > 20):
                    break

                words.append(word)
                context=(context[1:] + [idx])

                if word in ["home", "happy", "together", "smiled", "end"] and len(words) > 40:
                    # sentence_count+=1
                    break

                if sentence_count>=6:
                    break

        txt=" ".join(words)
        txt=txt.replace("then then then", "then")
        txt=txt.replace("always always", "always")
        txt=txt.replace("something something", "something")

        if not txt.endswith("."):
            txt+="."

        return txt.capitalize()

