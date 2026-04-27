import re


class StoryEvaluator:
    def keyword_coverage( self, story, keywords):
        story=story.lower()
        hit=0
        for k in keywords:
            if k.lower() in story:
                hit+=1

        return hit/len(keywords)


    def vocab_diversity(self, story):
        words=re.findall(r"\w+", story.lower())
        if len(words)==0:
            return 0

        return len(set(words))/len(words)


    def avg_sentence_length(self, story):
        sents=story.split(".")
        words=story.split()

        if len(sents)<=1:
            return len(words)

        return len(words)/(len(sents)-1)



    def evaluate_runs(self, stories, keywords):
        cover=[]
        div=[]
        sent=[]
        for s in stories:
            cover.append(self.keyword_coverage(s, keywords))
            div.append(self.vocab_diversity(s))
            sent.append(self.avg_sentence_length(s))

        return {
            "keyword_coverage":
                round(sum(cover)/len(cover),3),
            "diversity":
                round(sum(div)/len(div),3),
            "avg_sentence_length":
                round(sum(sent)/len(sent),2)
        }