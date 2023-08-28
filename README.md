# ziplm

Beam search is a common way of improving the quality of poor quality language models.
The ZipLM is a "Useless but mildly interesting language model using compressors built-in to Python."
If we try to sample a sentence after training on The Great Gatsby:

```{python}
data = open(gatsby).read()[-1000:]
model = ziplm.ZipModel(SimpleEncoder()).fit(data)
"".join(model.sample_sequence(100))  # I get 'ixlI vsoioul.des-kA;dfagwoI-k;IekxyIaIbu Ddg.u bg,obiouu;xrhistu ewurxasw-v ;suasockmiln.Gyh esbykIw'
```

It's pretty close to random gibberish.
Sure, the ZipLM will succeed in continuing the sequence "abcabcabc..." if you give it a long enough prompt, but it hardly generates legible human text.
In this repository I've addeda simple implementation of Beam search.
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
I'd love to try even higher beam widths, but the language model is very slow - even as a restricted to "training" on the last 1000 characters of the Gatsby text.
I will solve this problem later, and explore how Byte Pair Encoding helps (or doesn't help), but first let's understand why Beam search even works.

## Why we need Beam search
To really understand why the ZipLM is so bad without beam search, let's try to look at the actual log-probs of a next output character:
```
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
Apparently the compressor was able to incooperate the "b" into some token in the suffix of the prompt.
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
So the real valued $\log_2(1/p(x|c))$ gets "rounded" either up or down.

Why does Beam search help?
With Beam search we are basically asking the question: "Given I'm going to write 100 characters, which _complete string_ will have the smallest compression".
Of course actually answering this question would require a "complete search" rather than a "beam search", but this is the question we are approximating the answer to as we increase the beam width.
Now we are asking for an approximation to $\log_2(1/p(x_1x_2x_3|c)) = \log_2(1/p(x_1|c)) + \log_2(1/p(x_1x_2|cx_2)) + \log_2(1/p(x_3|cx_2x_2))$, which is a larger number, and so the "rounding" effects are less pronounced.

## Making it faster by delving into gzip itself

The main issue with scaling the Beam search algorithm for ZipLM is that we have to recompress 

