import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import os
import pickle

class ElmanNLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim = 128):
        super().__init__()
        # self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # simple RNN layer (Elman)
        self.rnn = nn.RNN(input_size = vocab_size, hidden_size = hidden_dim, batch_first = True, nonlinearity = "tanh")
        # Output layer to predict vocabulary probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input_seq, hidden_state):
        output, hidden_state = self.rnn(input_seq.float(), hidden_state)
        output = self.fc(output)
        output = self.softmax(output)

        return output, hidden_state

    def init_hidden(self, batch_size):
        # Initialize hidden state (h_0)
        return torch.zeros(1, batch_size, self.hidden_dim)
    
class RNNModel:
    def __init__(self, sequence_length = 4, hidden_dim = 128):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.model = None
        self.vocab = []
        self.word_to_index = {}
        self.index_to_word = {}
        self.training_sequences = []

    def preprocess(self, text):
        print("Preprocessing")
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
        print("Vocabulary size:", len(self.vocab))

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
        indices=[]

        for word in sequence:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
            else:
                indices.append(random.randint(0, len(self.vocab) - 1))

        return indices
        # return [self.word_to_index[word] for word in sequence]
    
    def one_hot_encode_sequence(self, indices, vocab_size):
        # Efficient one-hot encoding
        return torch.eye(vocab_size)[torch.tensor(indices)]

    def build_RNN(self):
        print("Building RNN")
        self.model = ElmanNLM(len(self.vocab), self.hidden_dim)

    def train(self, text, epochs = 30, lr = 0.001, model_path = "saved_models/rnn.pkl"):
        if os.path.exists(model_path):
            print("Loading saved model...")
            self.model = self.load_model(model_path)
            return
        tokens = self.preprocess(text)
        self.build_vocabulary(text)
        # self.build_
        self.prepare_trainning_data_rnn(text, sequence_length = self.sequence_length )

        self.build_RNN()

        X_text_seq, y_text_seq = self.training_sequences
        X_indices_seq = [self.words_to_indices(seq) for seq in X_text_seq]
        y_indices_seq = [self.word_to_index[word] for word in y_text_seq]
     
        X_encode_seq = [self.one_hot_encode_sequence(indices, len(self.vocab)) for indices in X_indices_seq]
        # y_targets = torch.tensor(y_indices_seq)

        X_train_tensor = torch.stack(X_encode_seq)
        y_train_tensor=torch.tensor(y_indices_seq)
        # y_train_tensor = y_targets

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr)
        print("Training RNN")

        for epoch in range(epochs):
            total_loss = 0
            for i in range(X_train_tensor.size(0)):
            # for i in range(min(1000, X_train_tensor.size(0))): Test
                # Initialise hidden state for each sequence with batch size 1
                hidden = self.model.init_hidden(1)
                self.model.zero_grad() # Clear gradients
                # Shape (1, seq_len, vocab_size) - batch_size = 1
                sequence_in = X_train_tensor[i].unsqueeze(0) 
                target = y_train_tensor[i].unsqueeze(0)

                # Forward pass
                output, hidden = self.model(sequence_in, hidden)
                # Loss function calculation: predict the last word in the sequence
                loss = loss_function(output[:, -1, :], target) # Output at last time step, target is next word index

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / min(1000, X_train_tensor.size(0))
            print(f"Epoch {epoch+1}/{epochs} " f"| Avg Loss={avg_loss:.4f}")
        
        print("Training Complete")
        self.save_model(model_path)


    def next_word(self, context):
        words = context.lower().split()
        words = words[-self.sequence_length:]
        while len(words) < self.sequence_length:
            words.insert(0, random.choice(self.vocab))

        indices = [self.word_to_index[w] for w in words]
        x = (self.one_hot_encode_sequence(indices, len(self.vocab)).unsqueeze(0))
        hidden = self.model.init_hidden(1)
        
        with torch.no_grad():
            output, _ = self.model(x, hidden)
        
        probs = torch.exp(output[:, -1, :]).squeeze()
        top = torch.topk(probs, 10)
        words = []
        for p, i in zip(top.values, top.indices):
            words.append((self.index_to_word[i.item()], p.item()))
        return words
    
    def generate_text_rnn(self, seed_sequence, length, vocab_size, keywords=None):
        # self.model.eval()
        self.model.eval() 
        generated_text = []
        seed_words = seed_sequence.split()

        current_sequence_indices = (self.words_to_indices(seed_words))
        hidden = self.model.init_hidden(1)

        with torch.no_grad():
            for step in range(length):
                input_tensor = (self.one_hot_encode_sequence(current_sequence_indices, vocab_size).unsqueeze(0))
                output, hidden = self.model(input_tensor, hidden)

                probs = torch.exp(output[:, -1, :]).squeeze()

                # thematic keyword injection
                if (keywords is not None and len(keywords) > 0 and random.random() < 0.18):
                    inject = random.choice(keywords)

                    if inject in self.word_to_index:
                        predicted_index = (self.word_to_index[inject])
                    else:
                        predicted_index = (torch.multinomial(probs, 1).item())
                else:
                    predicted_index = (torch.multinomial(probs, 1).item())

                next_word = (self.index_to_word[predicted_index])
                generated_text.append(next_word)

                current_sequence_indices = (current_sequence_indices[1:] + [predicted_index])

        return " ".join(generated_text)

    def thematic_seed(self, keywords):
        """
        retrieve thematic seed from training data
        """
        X_seq, _ = self.training_sequences
        candidates = []

        keyword_set = set(k.lower() for k in keywords)

        for seq in X_seq:
            overlap = len(keyword_set & set(seq))
            if overlap > 0:
                candidates.append((overlap, seq))

        if not candidates:
            return random.choice(X_seq)

        candidates.sort(key = lambda x:x[0], reverse=True)
        top = candidates[:20]
        chosen = random.choice(top)[1]

        return chosen

    def generate_story(self, keywords, max_words = 100):
        seed_tokens = self.thematic_seed(keywords)
        while len(seed_tokens) < self.sequence_length:
            seed_tokens.append(random.choice(self.vocab))

        seed_tokens=seed_tokens[:self.sequence_length]
        seed = " ".join(seed_tokens)

        print("Generated story (RNN thematic):")

        story = self.generate_text_rnn(seed, max_words, len(self.vocab), keywords=keywords)
        story_lower = story.lower()
        missing = []

        for k in keywords:
            if k.lower() not in story_lower:
                missing.append(k)

        if len(missing) > 0:
            story += (" In the end they found " + ", ".join(missing) + ".")

        return story
        

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

