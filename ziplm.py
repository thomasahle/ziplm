import gzip, bz2, lzma

import numpy as np
import scipy.special

import gzip
import io
import numpy as np
import scipy.special

from collections import Counter
import tqdm
import heapq


class SimpleEncoder:
    def __init__(self):
        self.vocab = set()

    def fit(self, text):
        self.vocab = set(text)


class ZipModel:
    def __init__(self, encoder=None, compressor=gzip, conversion=np.log(256)):
        self.bpe = SimpleEncoder() if encoder is None else encoder
        self.training = ""
        self.compressor = compressor
        self.conversion = conversion

    def fit(self, training):
        self.training = training
        self.bpe.fit(training)
        return self

    def logprobs(self, prefix="", temperature=1):
        base_size = len(
            self.compressor.compress("".join([self.training, prefix]).encode())
        )
        code_lengths = np.array(
            [
                (
                    len(
                        self.compressor.compress(
                            "".join([self.training, prefix, v]).encode()
                        )
                    )
                    - base_size
                )
                / len(v)
                for v in self.bpe.vocab
            ]
        )
        return scipy.special.log_softmax(
            -code_lengths * self.conversion * (1 / temperature)
        )

    def sample(self, prefix="", temperature=1):
        scores = self.logprobs(prefix, temperature=temperature)
        vocabulary = list(self.bpe.vocab)
        i = np.random.choice(range(len(vocabulary)), p=np.exp(scores))
        return vocabulary[i]

    def sample_sequence(self, maxlen, prefix="", temperature=1):
        sequence = prefix
        for k in tqdm.tqdm(range(maxlen)):
            result = self.sample(sequence, temperature=temperature)
            yield result
            sequence += result

    def beam_search(self, max_len, beam_width=1, prefix=""):
        # Each item in the beam is a tuple (sequence, score)
        beam = [(prefix, 0)]

        for step in tqdm.tqdm(range(max_len)):
            candidates = []

            # For each sequence in the beam, get possible next tokens
            for sequence, score in beam:
                logprobs = self.logprobs(sequence)
                top_indices = np.argsort(logprobs)[-beam_width:]  # Get top k indices

                for index in top_indices:
                    next_token = list(self.bpe.vocab)[index]
                    new_sequence = sequence + next_token
                    new_score = score + logprobs[index]
                    candidates.append((new_sequence, new_score))

            # Sort all candidates by score
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

            # Keep top beam_width sequences for the next round
            beam = sorted_candidates[:beam_width]

        # Return the best sequence (which is the first one in the beam after sorting)
        return beam[0][0]


def main():
    data = "abcabcabc"
    # data = "this is an example this is an example this is an example"
    data = open("gatsby").read()[-1000:]
    print("Original size:", len(data))

    # LZ77 Encode
    print("BP Encoding...")
    lm = ZipModel(SimpleEncoder())
    # lm = ZipModel(BPEncoder(num_merges=100))
    lm.fit(data)

    print(lm.bpe.vocab)

    print("Doing beam search...")
    max_len = 100
    for size in [1, 10, 100, 1000, 10000]:
        print(lm.beam_search(size, max_len))


if __name__ == "__main__":
    main()
