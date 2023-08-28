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




