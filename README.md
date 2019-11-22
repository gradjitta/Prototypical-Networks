# Prototypical-Networks
Prototypical Network in Pytorch for Text



*This code repo is experimental and built using CS330 homework material as guide*


# Dataset:
Newsgroup 20 dataset from sklearn.
- Sampled 30 entities per class to build the `mini_newsgroup_data.pkl`
- Used `nlp = spacy.load('en_trf_bertbaseuncased_lg')` to extract embeddings 
- Extracted embeddings in `mini_newsgroup_vectors.pkl`


### TODO

- [x] Working Prototypical-Network for 20 newsgroup dataset
- [ ] Clean up code
- [ ] Add more prototypical networks for text and `example/` folder
- [ ] ....will add more
