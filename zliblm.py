import random
from collections import Counter, namedtuple
import heapq
from tqdm import tqdm

# Huffman Coding
class Node(namedtuple("Node", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class Leaf(namedtuple("Leaf", ["value"])):
    def walk(self, code, acc):
        code[self.value] = acc or "0"

def huffman_encode(triples):
    h = []
    for value, freq in Counter(triples).items():
        h.append((freq, len(h), Leaf(value)))

    heapq.heapify(h)

    count = len(h)
    while len(h) > 1:
        freq1, _count1, left = heapq.heappop(h)
        freq2, _count2, right = heapq.heappop(h)
        heapq.heappush(h, (freq1 + freq2, count, Node(left, right)))

    code = {}
    if h:
        [(_freq, _count, root)] = h
        root.walk(code, "")
    return code

# LZ77 Encoding
def lz77_encode(s, window_size=200):
    i = 0
    out = []

    with tqdm(total=len(s)) as pbar:
        while i < len(s):
            max_len = 0
            max_offset = 0
            buf_end = min(i + window_size, len(s))

            while buf_end > i:
                offset = max(0, i - window_size)
                substr = s[i:buf_end]

                position = s.rfind(substr, offset, i)
                
                if position != -1 and buf_end - i > max_len:
                    max_len = buf_end - i
                    max_offset = i - position
                buf_end -= 1
            
            if max_len > 0:
                out.append((max_offset, max_len))
                inc = max_len
            else:
                out.append((0, 1, s[i]))
                inc = 1
            pbar.update(inc)
            i += inc
        return out

def sample(cnt, n, prefix=""):
    vocab = list(cnt.keys())
    weights = list(cnt.values())
    for _ in range(n):
        match random.choices(vocab, weights)[0]:
            case (0, 1, ch):
                s = ch
            case (offset, length):
                if offset > len(prefix):
                    continue
                s = prefix[-offset:-offset+length]
            case default:
                print("What?", default)
        yield s
        prefix += s

def decode(tokens, prefix=""):
    for token in tokens:
        match token:
            case (0, 1, ch):
                s = ch
            case (offset, length):
                s = prefix[-offset:-offset+length]
        yield s
        prefix += s

def main():
    data = "abcabcabc"
    #data = "this is an example this is an example this is an example"
    #data = open("text").read()
    print("Original size:", len(data))

    # LZ77 Encode
    lz77_encoded = lz77_encode(data, window_size=200)

    print("Encoded:", lz77_encoded)
    print("Decoded:", "".join(decode(lz77_encoded)))

    print("LZ77 Encoded size:", len(lz77_encoded))

    cnt = Counter(lz77_encoded)
    for triple, count in cnt.most_common(10):
        print(triple, count)

    for ch in sample(cnt, 100, data):
        print(ch, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
