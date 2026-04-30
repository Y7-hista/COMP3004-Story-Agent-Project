import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import os
from models.model_utils import save_model, load_model


class ElmanNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 128, hidden_dim = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = 2
        # simple RNN layer (Elman)
        self.rnn = nn.RNN(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True, nonlinearity = "tanh", dropout=0.3, num_layers = self.num_layers)
        # Output layer to predict vocabulary probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input_seq, hidden_state):
        x = self.embedding(input_seq.long())
        output, hidden_state = self.rnn(x, hidden_state)
        output = self.fc(output)
        # output = self.softmax(output)
        return output, hidden_state

    def init_hidden(self, batch_size):
        # Initialize hidden state (h_0)
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
    
class RNNModel:
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
        print("RNN Preprocessing")
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
        print("(RNN) Vocabulary size:", len(self.vocab))

    def prepare_trainning_data_rnn(self, text, sequence_length = 2):
        # Sequence length for RNN
        words = text.lower().split()
        X_seq = []
        y_seq = []

        for i in range(len(words) - self.sequence_length):
            input_sequence = words[i: i + self.sequence_length]
            target_word = words[i + self.sequence_length]
            X_seq.append(input_sequence)
            y_seq.append(target_word)
        self.training_sequences = (X_seq, y_seq)
        print("Trainning sequences: ", len(X_seq))
    
    def words_to_indices(self, sequence):
        return [self.word_to_index[word] for word in sequence]
    
    def build_RNN(self):
        print("Building RNN")
        self.model = ElmanNLM(len(self.vocab), embedding_dim = 128, hidden_dim = self.hidden_dim)

    def train(self, text, epochs = 30, lr = 0.001, model_path = "saved_models/rnn.pkl"):
        tokens = self.preprocess(text)
        self.build_vocabulary(text)
        self.prepare_trainning_data_rnn(text, sequence_length = self.sequence_length )
        if os.path.exists(model_path):
            print("Loading saved model...")
            self.model = load_model(model_path)
            return
        # tokens = self.preprocess(text)
        # self.build_vocabulary(text)
        # self.prepare_trainning_data_rnn(text, sequence_length = self.sequence_length )

        self.build_RNN()

        X_text_seq, y_text_seq = self.training_sequences
        X_indices_seq = [self.words_to_indices(seq) for seq in X_text_seq]
        y_indices_seq = [self.word_to_index[word] for word in y_text_seq]
     
        X_train_tensor = torch.tensor(X_indices_seq, dtype = torch.long)
        y_train_tensor = torch.tensor(y_indices_seq, dtype = torch.long)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr)
        print("Training RNN")

        for epoch in range(epochs):
            total_loss = 0

            indices = random.sample(range(X_train_tensor.size(0)), min(5000, X_train_tensor.size(0)))
            for i in indices: # Test
            # for i in range(X_train_tensor.size(0)):
                # Initialise hidden state for each sequence with batch size 1
                hidden = self.model.init_hidden(1)
                optimizer.zero_grad()
                # Shape (1, seq_len, vocab_size) - batch_size = 1
                sequence_in = X_train_tensor[i].unsqueeze(0) 
                target = y_train_tensor[i].unsqueeze(0)

                # Forward pass
                output, hidden = self.model(sequence_in, hidden)
                # Loss function calculation: predict the last word in the sequence
                loss = loss_function(output[:, -1, :], target) # Output at last time step, target is next word index
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / min(5000, X_train_tensor.size(0))
            print(f"(RNN) Epoch {epoch+1}/{epochs} " f"| Avg Loss={avg_loss:.4f}")
        
        print("Training Complete")
        save_model(self.model, model_path)


    def generate_sentence_rnn(self, context_indices, target, neighbors, unused, max_len=16):
        hidden = self.model.init_hidden(1)
        sentence=[]
        with torch.no_grad():
            for t in range(max_len):
                x=torch.tensor([context_indices], dtype=torch.long)
                output, hidden = self.model(x, hidden)
                logits=output[:,-1,:].squeeze()

                # lower temperature
                temperature=0.85
                probs=torch.softmax(logits/temperature, dim=-1)

                # topic bias
                if target in self.word_to_index:
                    probs[self.word_to_index[target]] *= 3.0

                if target in neighbors:
                    for n in neighbors[target]:
                        if n in self.word_to_index:
                            probs[self.word_to_index[n]] *= 1.4

                # repetition penalty
                for w in sentence[-3:]:
                    if w in self.word_to_index:
                        probs[self.word_to_index[w]] *= 0.03

                probs=probs / probs.sum()


                # top-k safer than top-p here
                topk=8
                vals, ids=torch.topk(probs, topk)

                vals=vals / vals.sum()

                idx=ids[torch.multinomial(vals, 1)].item()
                next_word=self.index_to_word[idx]

                # sentence stopping heuristic
                if (len(sentence) > 8 and next_word in ["and", "but", "the", "a", "of", "to"] is False and random.random()<0.18
                ):
                    break

                sentence.append(next_word)
                context_indices=(context_indices[1:] + [idx])

                if next_word in unused:
                    unused.remove(next_word)

        txt=" ".join(sentence)
        txt=txt.capitalize()+"."

        return txt, context_indices


    def sample_next_word(self, logits, target, neighbors, unused, recent_words):
        probs=torch.softmax(logits, dim=-1)

        # temperature
        temperature=1.25
        probs=torch.softmax(logits/temperature, dim=-1)

        # topic bias
        if target in self.word_to_index:
            idx=self.word_to_index[target]
            probs[idx]*=2.2

        if target in neighbors:
            for n in neighbors[target]:
                if n in self.word_to_index:
                    probs[
                    self.word_to_index[n]
                    ]*=1.3

        for kw in unused:
            if kw in self.word_to_index:
                probs[self.word_to_index[kw]]*=1.2

        # anti repeat
        for w in recent_words[-4:]:
            if w in self.word_to_index:
                probs[self.word_to_index[w]]*=0.05

        probs=probs/probs.sum()

        # nucleus sampling
        sorted_probs,sorted_ids=(torch.sort(probs, descending=True))
        cum=torch.cumsum(sorted_probs, dim=0)

        mask=(cum<=0.95)
        mask[0]=True

        sorted_probs=sorted_probs[mask]
        sorted_ids=sorted_ids[mask]
        sorted_probs=(sorted_probs/ sorted_probs.sum())
        sampled=torch.multinomial(sorted_probs, 1).item()

        return sorted_ids[sampled].item()


    def generate_story(self, keywords, num_sentences = 8):
        if self.planner is None:
            raise ValueError("planner missing")

        # use TopicPlanner
        plan=(self.planner.build_topic_plan(keywords))

        seed_tokens=plan["seed"]
        topic_plan=plan["plan"]
        neighbors=plan["neighbors"]

        while len(seed_tokens)<self.sequence_length:
            seed_tokens.append(random.choice(self.vocab))

        context=self.words_to_indices(seed_tokens[:self.sequence_length])
        unused=set(k.lower() for k in keywords)
        story=[]

        for s in range(num_sentences):
            target=(topic_plan[s] if s < len(topic_plan) else random.choice(keywords))

            sent, context=(self.generate_sentence_rnn(context, target, neighbors, unused))

            story.append(sent)

        return " ".join(story)
    

