import random
from models.ngram_model import NGramModel
from data_utils import tokenize


class StoryAgent:

    def __init__(self, max_stories=500):

        print("Loading TXT dataset...")

        tokens = []
        story_count = 0
        current_story = ""

        with open("data/TinyStories-train.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line == "<|endoftext|>":

                    if current_story:
                        tokens += ["<s>"] + tokenize(current_story.lower()) + ["</s>"]
                        story_count += 1
                        current_story = ""

                    if story_count >= max_stories:
                        break

                else:
                    current_story += " " + line

        print(f"Loaded {story_count} stories")

        # ===== 正确训练 =====
        self.trigram = NGramModel(n=3)
        self.trigram.train(tokens)

        self.bigram = NGramModel(n=2)
        self.bigram.train(tokens)

        self.unigram = NGramModel(n=1)
        self.unigram.train(tokens)

        self.vocab = list(set(tokens))

        print("Training finished.")

    def generate(self, keywords, max_len=100):

        keywords = [k.lower() for k in keywords]
        result = ["<s>"]

        for _ in range(max_len):

            next_word = None

            # ===== trigram =====
            if len(result) >= 2:
                next_word = self.trigram.predict(result)

            # ===== bigram =====
            if next_word is None and len(result) >= 1:
                next_word = self.bigram.predict(result)

            # ===== unigram =====
            if next_word is None:
                next_word = random.choice(self.vocab)

            # 避免 start token
            if next_word == "<s>":
                continue

            # 句子结束
            if next_word == "</s>":
                break

            result.append(next_word)

        # ===== 插入关键词（更合理）=====
        for kw in keywords:
            insert_pos = random.randint(1, len(result)-1)
            result.insert(insert_pos, kw)

        final_words = [w for w in result if w not in {"<s>", "</s>"}]

        return " ".join(final_words)