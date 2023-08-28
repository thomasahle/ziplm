# ZipLm with Beam Search, BPE and Progressive Compression

Beam search is a common way of improving the quality of poor-quality language models.
The ZipLM is a "Useless but mildly interesting language model using compressors built-in to Python."
If we try to sample a sentence after training on The Great Gatsby:

```{python}
data = open(gatsby).read()[-1000:]
model = ziplm.ZipModel(SimpleEncoder()).fit(data)
"".join(model.sample_sequence(100))  # I get 'ixlI vsoioul.des-kA;dfagwoI-k;IekxyIaIbu Ddg.u bg,obiouu;xrhistu ewurxasw-v ;suasockmiln.Gyh esbykIw'
```

It's pretty close to random gibberish.
Sure, the ZipLM will succeed in continuing the sequence "abcabcabc..." if you give it a long enough prompt, but it hardly generates legible human text.
In this repository, I've added a simple implementation of Beam search.
Let's see how well it does:

```{python}
data = open(gatsby).read()[-1000:].replace("\n", "")
model = ziplm.ZipModel(SimpleEncoder()).fit(data)
model.beam_search(100, beam_width=1)      # Output: ' ADDsDDsDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD'
model.beam_search(100, beam_width=10)     # Output: " Acbby'b close;DutchuDutchuDutlue lawn, anycbby'bDutlueowawn,Dutc coulwloutchuDutluewlouDutchuDuwmo'"
model.beam_search(100, beam_width=100)    # Output: "Its vt flowlors' t man n dreared into atood nd treerer Gatso his breaggransita nd gre sat tdonce for"
model.beam_search(100, beam_width=1000)   # Output: " Ao his breath k. He had cow of Dais he lawn, st he could hwith d Daisy'ssy'ss omed sought olay with"
```

At least "He had cow of Dais he lawn" sounds more like a real sentence than "Ddg.u bg,obiouu;".
But it's still worse than the simple [Markov chain language models people trained a decade ago](https://kingjamesprogramming.tumblr.com/).
I'd love to try even higher beam widths, but the language model is very slow - even as restricted to "training" on the last 1000 characters of the Gatsby text.
I will solve this problem later, and explore how Byte Pair Encoding helps (or doesn't help), but first, let's understand why Beam search even works.

## Why we need Beam search
To really understand why the ZipLM is so bad without beam search, let's try to look at the actual log-probs of a next output character:
```{python}
model.logprobs("His name was ")
Out[85]:
array([-8.87835804, -3.3331806 , -3.3331806 , -3.3331806 , -3.3331806 ,
       -8.87835804, -3.3331806 , -8.87835804, -3.3331806 , -3.3331806 ,
       -3.3331806 , -3.3331806 , -3.3331806 , -8.87835804, -3.3331806 ,
       -3.3331806 , -3.3331806 , -8.87835804, -3.3331806 , -3.3331806 ,
       -8.87835804, -3.3331806 , -8.87835804, -3.3331806 , -3.3331806 ,
       -3.3331806 , -3.3331806 , -3.3331806 , -3.3331806 , -3.3331806 ,
       -3.3331806 , -3.3331806 , -3.3331806 , -3.3331806 , -3.3331806 ])
```

There are basically just two different values.
What gives?
Well. The ZipLM works by taking the prompt/"training text"; concatenating it with each character from a to z; and compressing each of 26 strings using gzip.
That is, it measures `len(gzip(prompt + "a"))`, `len(gzip(prompt + "b"))`, and so on.
In the case above, the length of `gzip(prompt)` is 565 bytes.
The length of `gzip(prompt + "a")` is 566 bytes, but the length of `gzip(prompt + "b")` is just 565 like the prompt.
Apparently, the compressor was able to incorporate the "b" into some token in the suffix of the prompt.
If we look at the length of `gzip(prompt + ch)` for all characters, `ch`, minus the length of `gzip(prompt)`, we get

```
[1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

Only two different values.
This explains why there are only two different logprops as well.
One way to think about this is that each possible token/character, $x$, has an output probability $p(x|c)$ given the context, $c$.
An ideal compressor would use $\log_2(1/p(x|c))$ bits to encode character $x$.
However, the `gzip` command, as used in the ZipLM, has to output a number of bytes.
So the real-valued $\log_2(1/p(x|c))$ gets "rounded" either up or down.

Why does Beam search help?
With Beam search we are basically asking the question: "Given I'm going to write 100 characters, which _complete string_ will have the smallest compression".
Of course, actually answering this question would require a "complete search" rather than a "beam search", but this is the question we are approximating the answer to as we increase the beam width.
Now we are asking for an approximation to $\log_2(1/p(x_1x_2x_3|c)) = \log_2(1/p(x_1|c)) + \log_2(1/p(x_1x_2|cx_2)) + \log_2(1/p(x_3|cx_2x_2))$, which is a larger number, and so the "rounding" effects are less pronounced.

## Making it faster by delving into gzip itself

The main issue with scaling the Beam search algorithm is that the original ZipLM compresses the entire "training" string every time we generate a new set of logprobs like this: `len(self.compressor.compress("".join([self.training, prefix, v]).encode()))`.
This approach may be fine for the original greedy sampling approach, but with beam search, we are doing a lot more calls to logprob, so this gets slow.
What we'd like instead is a "progressive" encoder, that allows us to compress the training text once, and only pay a constant price per new character measured.
This is a bit more tricky than it might seem at first since the gzip algorithm keeps an internal buffer of the input we give it. So most of the time, as we show it new text, it doesn't output anything. This doesn't mean that the new text has 0 entropy. It just means that gzip is still waiting for more text to decide how to encode it.

I eventually came up with the following solution:

```{python}
    def fit(self, training):
        self.bpe.fit(training)  # Train the encoder, if we're using one
        self.compressor = zlib.compressobj()
        self.compressor.compress(training.encode())  # "Train" the model
        self.compressor.flush(zlib.Z_SYNC_FLUSH)     # Friendly nudge to reduce the buffer size
        self.base_size = len(self.compressor.copy().flush())  # Size of buffer if we stopped now
        return self

    def measure(self, string):
        compressor = self.compressor.copy()          # Copy the internal state and buffer
        data = compressor.compress(string.encode())  # Main compression call
        data += compressor.flush()                   # Squeeze out the rest
        return len(data) - self.base_size            # Subtract size of text already in buffer
```

The idea is to close the compressor after each measurement, (you can't compress new text after `compressor.flush()`,) to be sure I have the most realistic measurement.
To avoid disturbing the main compressor that we took time to "train", I copied it, together with its internal buffer before each measurement.

This works pretty well, and we can now train on the entire corpus, and still do pretty fast beam searches:
```
import ziplm2
data = open("gatsby").read()
model = ziplm2.ZipModel().fit(data)
model.beam_search(100, beam_width=100)  # Output: 'xi drivers in the village never\ntook a fare past the entrance gate without stopping for a minute and'
```

The only problem? This text is just a plain copy of a line from somewhere near the penultimate paragraph.
Why do we overfit so hard? To understand that, we need to dive deep into the inner workings of gzip, or lz77 as the internal algorithm is called.

## The insides of Gzip

Gzip, at its heart, employs the DEFLATE compression algorithm. DEFLATE, in turn, combines the power of two algorithms: LZ77 (Lempel-Ziv 1977) and Huffman coding.
The Huffman coding is simply a way to turn the token probabilities into actual bits we can write to a stream, so the Lz77 part is what we are interested in understanding. 

Example: Assume we are compressing the text ABRACADABRA.
Lz77 converts this into a list of tokens:
```
(0,0,A) (0,0,B) (0,0,R) (-3,1,C) (-2,1,D) (-7,3,A)
```
The format for each token is `(offset, length, next_character)`.
The idea is to repeatedly, greedily find the longest match for the remaining suffix of text to be compressed in the already compressed text.
In the beginning of course there is no match, so we have some 0-length tokens, like (0,0,A).
But by the time we are compressing the suffix `ACADABRA` we can use the `A` we have already seen and write `(-3,1,C)` meaning "go back 3 and take 1 character, then write C".

Once we have converted the text into this tokenized format,  we count the number of each kind of unique token.
This gives us the probability distribution we use for the Huffman encoding.
We see this is an extraordinarily simple scheme, where the assumed probability distribution of a token, given the context, $p(x|c)$, is completely independent of the context!
Of course, there is still some dependency on the character level, but only because the tokens themselves refer to the context.

Now assume we compute the probability $p(x)$, the length $w_x$ and the entropy $e_x = \log_2(1/p(x))$ for each token.
How can we write the longest text using the fewest bits?
Of course by just repeating the token that maximizes $w_x / e_x$.
With a wide enough beam width, this is what beam search will find.
So it's just going to repeat some random substring infinitely.

## Enter Byte embedding

As we have seen, the ZipLM on it's own doesn't work: The probability distribution on a character level is discretized so much that we just get random text.
On the other hand, using beam search - which normally is useful for getting better output from bad language models - will just give us a boring repeated piece of text from the training data.
Does it mean the war is lost, and we must give up and go back to transformers?
Emphatically no!
It simply means we need to use an even smarter form of search.
We need something that still samples from the compressor's probability distribution, rather than just maximizing the tokens per bit. But it should work on a larger scale than single characters.
One way to do this is to find some useful "vocabulary" of subwords, and simply run the sampling algorithm on this instead of the character level.
I wrote a simple Byte pair encoding (as is used by GPT and others).
We can then run sampling at different temperatures and see what happens:

```{python}
lm = ZipModel(BPEncoder(num_merges=1000)).fit(data)
for temp in range(1, 10):
       print(f'Temperature: {temp}, Output:', lm.sample_sequence(20, temperature=temp))
```
Output:
```
Temperature: 1, Output: ong the ?"  "then and the something because s and the there was about the couldn't there was s and the about the there was about the , old sport, old sport, old sport, old sport, old sport
Temperature: 2, Output: serfir," he . It was afterno," she Gatsby's on the Gatsby, , and I another ----"  ", and the a little , and then had been something , old sport, and then . It was
Temperature: 3, Output: s, youGatsbyfirst there Daisy dn't come and the . . . . with the from the , and then thought into the she was afternoon, old sport," said couldn't
Temperature: 4, Output: when 't from the standbegan to thought n't answfor the ought . But she was a little want to , old sportWilson Jordan old sporthad been ."  The
Temperature: 5, Output: ickranning night dn't ly.  "Wolfshiin his into ed and , and I Miss Bakshe was afternoon."  The Gatsby's Miss Bakfrom the turned ," she
```

So okay, it's definitely not completely random, like sampling at the character level.
It still has repetitions at low temps and gets more rambly at larger temps.

## Conclusion

There is clearly still a lot that can be done to make ZipLM a more useful language model.
But one idea I had while writing this is that we could also go the other way:
Why not use the underlying token format of Lz77 when training LLMs?
Surely transformers can easily learn to say `(-3,1,C)` when they want to use characters they've already outputted a while back.
It might even teach them a better understanding of characters, solving some issues often attributed to the Byte Pair Encoding.
Another advantage is the ability to output longer pieces of text at the same price, speeding up inference.
And as a final bonus: You can train your LLM directly on your `.gz` file, without ever unpacking it!
