import re
import math
from collections import Counter
from itertools import combinations
import spacy

nlp = spacy.load("en_core_web_sm")

class StoryEvaluator:

    def __init__(self):
        pass

    # basic helpers
    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())


    def ngrams(self, words, n):
        if len(words) < n:
            return []

        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


    # 1 keyword coverage
    def keyword_coverage(self, story, keywords):
        story=story.lower()
        hit=0

        for k in keywords:
            if k.lower() in story:
                hit+=1

        return hit/max(1,len(keywords))


    # 2 keyword dispersion
    # avoid "cat sun castle"
    # glued together cheating
    def keyword_dispersion(self, story, keywords):
        words=self.tokenize(story)
        positions=[]

        for k in keywords:
            pos=[i for i,w in enumerate(words) if w==k.lower()]

            if len(pos)>0:
                positions.append(pos[0])

        if len(positions)<2:
            return 0

        spread=max(positions)-min(positions)

        return spread/max(1,len(words))


    # 3 Distinct-1
    def distinct1(self,story):
        words=self.tokenize(story)

        if len(words)==0:
            return 0

        return len(set(words)) / len(words)


    # 4 Distinct-2
    # stronger diversity metric
    def distinct2(self,story):
        words=self.tokenize(story)
        grams=self.ngrams(words, 2)

        if len(grams)==0:
            return 0

        return len(set(grams)) / len(grams)


    # 5 sentence length
    # keep but auxiliary
    def avg_sentence_length(self, story):
        sents=[s.strip() for s in re.split(r"[.!?]", story) if s.strip()]

        if len(sents)==0:
            return 0

        total_words=sum(len(self.tokenize(s)) for s in sents)

        return total_words/len(sents)


    # 6 repetition
    # trigram loop repetition
    # much better than adjacent repeat
    def repetition_rate(self, story):
        words=self.tokenize(story)
        grams=self.ngrams(words, 3)

        if len(grams)==0:
            return 0

        repeats=(len(grams) - len(set(grams)))

        return repeats/len(grams)

    # 7 lexical entropy
    # randomness / collapse balance
    def lexical_entropy(self, story):
        words=self.tokenize(story)
        if len(words)==0:
            return 0

        counts=Counter(words)
        total=len(words)
        entropy=0

        for c in counts.values():
            p=c/total
            entropy -= p*math.log(p+1e-12, 2)

        return entropy


    # 8 self-BLEU lite
    # diversity across generations
    # lower better
    # (simple Jaccard approximation)
    def self_bleu_like(self, stories):
        if len(stories)<2:
            return 0

        overlaps=[]

        for a,b in combinations(stories, 2):
            A=set(self.tokenize(a))
            B=set(self.tokenize(b))
            j=len(A&B)/max(1, len(A|B))
            overlaps.append(j)

        return sum(overlaps)/len(overlaps)

    def syntactic_wellformedness(self, story):
        sentences = [
            s.strip()
            for s in re.split(r"[.!?]",story)
            if s.strip()
        ]

        if not sentences:
            return 0

        valid=0

        for s in sentences:
            doc=nlp(s)
            has_subject=False
            has_verb=False
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    has_subject=True

                if token.pos_=="VERB":
                    has_verb=True

            if has_subject and has_verb:
                valid+=1

        return valid/len(sentences)

    # full evaluation
    def evaluate_runs(self, stories, keywords):
        cover=[]
        dispersion=[]
        d1=[]
        d2=[]
        sent=[]
        rep=[]
        entropy=[]
        syntax_validity=[]

        for s in stories:
            cover.append(self.keyword_coverage(s, keywords))
            dispersion.append(self.keyword_dispersion(s, keywords))
            d1.append(self.distinct1(s))
            d2.append(self.distinct2(s))
            sent.append(self.avg_sentence_length(s))
            rep.append(self.repetition_rate(s))
            entropy.append(self.lexical_entropy(s))
            syntax_validity.append(self.syntactic_wellformedness(s))

        self_bleu=self.self_bleu_like(stories)

        return {
            # prompt relevance
            "keyword_coverage": round(sum(cover)/len(cover), 3),
            "keyword_dispersion": round(sum(dispersion) / len(dispersion), 3),
            # diversity
            "distinct_1": round(sum(d1)/len(d1), 3),
            "distinct_2": round(sum(d2)/len(d2), 3),
            "lexical_entropy": round(sum(entropy) / len(entropy), 3),
            # quality
            "avg_sentence_length": round(sum(sent)/len(sent), 2),
            # degeneration
            "repetition_rate": round(sum(rep)/len(rep), 3),
            # cross-run diversity
            # LOWER better
            "self_bleu_like": round(self_bleu, 3),
            "syntax_validity": round(sum(syntax_validity) / max(1, len(syntax_validity)), 4)
        }