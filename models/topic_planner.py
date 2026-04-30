import random
import re
from collections import Counter


class TopicPlanner:

    def __init__(self, stories):
        self.stories = stories

    def build_topic_plan(self, keywords, events = 6):
        kw = [k.lower() for k in keywords]
        scored = []

        for story in self.stories:
            words = set(re.findall(r"\w+", story.lower()))
            overlap = len(set(kw)&words)

            if overlap>0:
                scored.append((overlap, story))

        if not scored:
            return {
                "seed":["once", "upon", "a", "time"
                ],
                "plan":["introduction", "journey", "conflict", "climax", "resolution"],
                "neighbors":{}
            }

        scored.sort(key = lambda x:x[0], reverse = True)
        chosen = random.choice(scored[:20])[1]

        # seed
        sents = re.split(r"[.!?]", chosen)
        seed = []

        for s in sents[:2]:
            seed.extend(s.lower().split())

        seed = seed[:10]
        # event plan
        events_plan = []

        for s in sents:
            s = s.strip()

            if len(s.split())>4:
                events_plan.append(s)

            if len(events_plan) >= events:
                break

        # keyword neighbors
        neighbors = {}

        for k in kw:
            context = []

            for story in self.stories:
                words = story.lower().split()

                for i,w in enumerate(words):
                    if w == k:
                        left = max(0, i-4)
                        right = min(len(words), i+5)
                        context.extend(words[left:i] + words[i+1:right])

            freq = Counter(context)
            neighbors[k] = [w for w, _ in freq.most_common(6) if w not in kw]

        topic_chain = []

        for k in kw:
            topic_chain.append(k)

            if neighbors[k]:
                topic_chain.extend(neighbors[k][:2])

        topic_chain +=  ["journey", "conflict", "resolution"]

        return {
           "seed":seed,
           "plan":topic_chain,
           "neighbors":neighbors
        }