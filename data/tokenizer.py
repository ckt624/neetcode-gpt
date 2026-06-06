from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus = list(corpus)
        ans = []
        for _ in range(num_merges):
            counts = Counter((corpus[i], corpus[i + 1]) for i in range(len(corpus) - 1))
            max_word = None

            for word in counts:
                if not max_word or counts[word] > counts[max_word] or counts[word] == counts[max_word] and word < max_word:
                    max_word = word

            new_corpus = []
            i = 0
            while i < len(corpus) - 1:
                if (corpus[i], corpus[i+1]) == max_word:
                    new_corpus.append(''.join(max_word))
                    i += 2
                else:
                    new_corpus.append(corpus[i])
                    i += 1
            corpus = new_corpus
            ans.append(list(max_word))

        return ans
            
