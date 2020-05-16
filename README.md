# Jointly Measuring Diversity and Quality in Text Generation Models
This is the implementation of metrics for measuring Diversity and Quality, which are introduced in [this paper](https://arxiv.org/abs/1904.03971). Besides, some other metrics exist.

For BLEU and Self-BLEU, [this hyperformance implementation](https://github.com/Danial-Alh/FastBLEU) is used.
## Sample Usage
### Multiset distances
Here is an example to compute MS-Jaccard distance. The input of these metrics is a list of tokenized sentences.
```python
from multiset_distances import MultisetDistances

ref1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
ref2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
ref3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
sen1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
sen2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 'in', 'world', 'history']

references = [ref1, ref2, ref3]
sentences = [sen1, sen2]

msd = MultisetDistances(references=references)
msj_distance = msd.get_jaccard_score(sentences=sentences)
```
The value of `msj_distance` is `{3: 0.17, 4: 0.13, 5: 0.09}`, which shows MS-Jaccard for 3-gram, 4-garm and 5-gram, respectively. 

### BERT based distances
Here is an example to compute FBD and EMBD distance. The input of these metrics is a list of strings, and BERT tokenizer is used in the code.


```python
from bert_distances import FBD, EMBD
references = ["that is very good", "it is great"]
sentences1 = ["this is nice", "that is good"]
sentences2 = ["it is bad", "this is very bad"]

fbd = FBD(references=references, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")
fbd_distance_sentences1 = fbd.get_score(sentences=sentences1)
fbd_distance_sentences2 = fbd.get_score(sentences=sentences2)
# fbd_distance_sentences1 = 17.8, fbd_distance_sentences2 = 22.0

embd = EMBD(references=references, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")
embd_distance_sentences1 = embd.get_score(sentences=sentences1)
embd_distance_sentences2 = embd.get_score(sentences=sentences2)
# embd_distance_sentences1 = 10.9, embd_distance_sentences2 = 20.4
```

# Resources
* [Paper](https://arxiv.org/pdf/1904.03971.pdf)
* [Poster](https://iams4n.github.io/posters/NAACL2019_NeuralGen_JointlyMeasuring.pdf)
* [Presentation Video](https://www.youtube.com/watch?v=x0MDJe4Oc4k)
* [Slide](https://docs.google.com/presentation/d/1S-kgqCYNeC9SiIOQ_GshQvVnJYEp1yXARcZAe-a2oDg)

# Citation

Please cite our paper if it helps with your research.

```latex
@misc{montahaei2019jointly,
    title={Jointly Measuring Diversity and Quality in Text Generation Models},
    author={Ehsan Montahaei and Danial Alihosseini and Mahdieh Soleymani Baghshah},
    year={2019},
    eprint={1904.03971},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
