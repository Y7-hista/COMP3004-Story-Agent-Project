import random
import re
from models.SLM import StatisticalLanguageModel
from models.RNN import RNNModel
import nltk

class StoryAgent:
    def __init__(self,max_stories=150):
        print("Loading dataset...")
        self.stories = []
        current = []

        with open("data/TinyStories-train.txt", encoding = "utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if line == "<|endoftext|>":
                    if current:
                        story = " ".join(current)
                        self.stories.append(story)
                        current = []
                        count += 1
                        if count >= max_stories:
                            break
                else:
                    current.append(line)

        text = " ".join(self.stories)
        # Train all models
        self.models = {}

        # Bigram
        print("Training Bigram")
        bigram = StatisticalLanguageModel(n = 2, smoothing="laplace")
        bigram.train(text, "saved_models/bigram.pkl")
        self.models["Bigram"] = bigram

        # Trigram
        print("Training Trigram")
        trigram = StatisticalLanguageModel(n = 3, smoothing = "laplace")
        trigram.train(text, "saved_models/trigram.pkl")
        self.models["Trigram"] = trigram

        # RNN
        print("Training RNN")
        rnn = RNNModel()
        rnn.train(text, epochs = 30)
        self.models["RNN"] = rnn

        # configs = {
        #     "Bigram":
        #     dict(n = 2, smoothing = "laplace", path = "saved_models/bigram.pkl"),
        #     "Trigram":
        #     dict(n = 3, smoothing = "laplace", path = "saved_models/trigram.pkl"),
        #     # "kneserney":
        #     # dict(n = 3, smoothing = "kneserney", path = "saved_models/kneser.pkl")
        # }

        # for name, cfg in configs.items():
        #     print(f"Training {name}")

        #     lm = StatisticalLanguageModel(n = cfg["n"], smoothing = cfg["smoothing"])

        #     lm.train(text, cfg["path"])

        #     self.models[name] = lm

    def retrieve_seed(self, keywords):
        keyword_set = set(k.lower() for k in keywords)
        scored = []
        for story in self.stories:
            words = set(re.findall(r"\w+", story.lower()))
            overlap = len(keyword_set & words)

            if overlap > 0:
                scored.append((overlap, story))

        if not scored:
            seed_story = random.choice(self.stories)
        else:
            scored.sort(key = lambda x:x[0], reverse = True)

            top = [x[1] for x in scored[:10]]
            seed_story = random.choice(top)

        sents = nltk.sent_tokenize(seed_story)
        seed_words = []

        for s in sents[:2]:
            seed_words += nltk.word_tokenize(s.lower())

        return (keywords + seed_words[:20])

    def generate(self, keywords, model_name = "trigram"):
        """
        Retrieval-Augmented Story Generation
        prompt control the theme of story
        """
        if model_name not in self.models:
            model_name = "trigram"
        # RNN separate generation
        if model_name=="RNN":
            return self.models["RNN"].generate_story(keywords)
        # N-gram generation
        
        seed = self.retrieve_seed(keywords)
        generated = self.models[model_name].generate(seed, num_sentences = 6)

        # keyword coverage repair
        story_text = generated.lower()
        missing = []

        for k in keywords:
            if k.lower() not in story_text:
                missing.append(k)

        if missing:
            generated += (" In the end they found " + ", ".join(missing) + ".")

        return generated
    
    def compare_models(self, keywords, runs = 5):
        results = {}
        for model_name in ["Bigram", "Trigram", "RNN"]:
            stories=[]
            for _ in range(runs):
                s=self.generate(keywords, model_name)
                stories.append(s)

            results[model_name]=stories
            # results[m] = self.generate(keywords, m)
        return results