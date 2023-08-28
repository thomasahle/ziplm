import zlib

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


class BPEncoder:
    def __init__(self, num_merges=10000):
        self.num_merges = num_merges
        self.merges = []
        self.token_to_text = {}

    def fit(self, text):
        for ch in text:
            if ch not in self.vocab:
                self.token_to_text[ch] = ch
        next_token = max(map(ord, self.token_to_text.keys())) + 1

        for i in tqdm.tqdm(range(self.num_merges)):
            cnt = Counter(text[i:i+2] for i in range(len(text)-1))
            if not cnt:
                break
            (pair, c), = cnt.most_common(1)
            if c == 1:
                print("Stopping early because there are no more tokens to merge.")
                break
            text = text.replace(pair, chr(next_token))
            self.merges.append((pair, chr(next_token)))
            word = self.token_to_text[pair[0]] + self.token_to_text[pair[1]]
            self.token_to_text[chr(next_token)] = word
            next_token += 1

    @property
    def vocab(self):
        return set(self.token_to_text.values())

    def encode(self, text):
        for (pair, token) in self.merges:
            text = text.replace(pair, token)
        return text

    def decode(self, text):
        for (pair, token) in self.merges[::-1]:
            text = text.replace(token, pair)
        return text


class ZipModel:
    def __init__(self, encoder=None, conversion=np.log(256)):
        self.bpe = SimpleEncoder() if encoder is None else encoder
        self.training = ""
        self.conversion = conversion

    def fit(self, training):
        self.bpe.fit(training)  # Train the encoder, if we're using one
        self.compressor = zlib.compressobj()
        self.compressor.compress(training.encode())  # "Train" the model
        self.compressor.flush(zlib.Z_SYNC_FLUSH)     # Friendly nudge to reduce the buffer size
        self.base_size = len(self.compressor.copy().flush())  # Size of buffer if we stopped now
        return self

    def measure(self, string):
        # We compy the compressor, so we can close the copy without disturbing the
        # main compressor. We need to close (flush with Z_FINISH) the copy, since that
        # is the only way to get a precise measure of the entropy of the string.
        compressor = self.compressor.copy()
        data = compressor.compress(string.encode())
        data += compressor.flush()
        return len(data) - self.base_size

    def logprobs(self, prefix="", temperature=1):
        code_lengths = np.array([
            self.measure(prefix + v) / len(v)
            for v in self.bpe.vocab
        ])
        return scipy.special.log_softmax(-code_lengths*self.conversion*(1/temperature))

    def sample(self, prefix="", temperature=1):
        scores = self.logprobs(prefix, temperature=temperature)
        vocabulary = list(self.bpe.vocab)
        i = np.random.choice(range(len(vocabulary)), p=np.exp(scores))
        return vocabulary[i]

    def sample_sequence(self, maxlen, prefix="", temperature=1):
        sequence = prefix
        for k in range(maxlen):
            result = self.sample(sequence, temperature=temperature)
            sequence += result
        return sequence

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
    #data = "this is an example this is an example this is an example"
    data = open("gatsby").read().replace("\n", " ")
    print("Original size:", len(data))

    # LZ77 Encode
    print("BP Encoding...")
    #lm = ZipModel(SimpleEncoder())
    lm = ZipModel(BPEncoder(num_merges=1000))
    lm.fit(data)

    print('BPE Vocabulary:', lm.bpe.vocab)
    original_text = "How are you doing?"
    encoded = lm.bpe.encode(original_text)
    decoded = lm.bpe.decode(encoded)
    assert decoded == original_text


    for temp in range(1, 10):
        print(f'Temperature: {temp}, Output:', lm.sample_sequence(20, temperature=temp))
    print()

    # print("Doing beam search...")
    # max_len = 100
    # for size in [1, 10, 100, 1000, 10000]:
    #     print(lm.beam_search(size, max_len))


if __name__ == "__main__":
    main()
