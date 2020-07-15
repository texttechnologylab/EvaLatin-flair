from typing import List

import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, BytePairEmbeddings, \
    PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from embeddings import WordToVecFormatEmbeddings

tag_type = 'upos'
column_map = {0: 'id', 1: 'text', 2: 'lemma', 3: 'upos', 4: 'xpos', 5: 'feats',
              6: 'head', 7: 'deprel', 8: 'deps', 9: 'misc'}
corpus: Corpus = ColumnCorpus('resources/data/split-80-20/', column_map,
                              train_file='train.conllu',
                              dev_file='dev.conllu',
                              test_file='/resources/corpora/LT4HALA/EvaLatin/gold_EvaLatin_03-03-2020/pos_classical_gold.conllu',
                              comment_symbol='#')
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    WordToVecFormatEmbeddings('/resources/nlp/embeddings/latin/form/la-2M-300-glove.vec'),
    WordToVecFormatEmbeddings('/resources/nlp/embeddings/latin/form/la-2M-300-fastText.vec'),
    WordToVecFormatEmbeddings('/resources/nlp/embeddings/latin/form/la-2M-300-wang2vec-50it.vec'),
    PooledFlairEmbeddings('/resources/nlp/embeddings/latin/char/lm-historic-latin-forward-v1.0.pt'),
    PooledFlairEmbeddings('/resources/nlp/embeddings/latin/char/lm-historic-latin-backward-v1.0.pt'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.SGD)
trainer.train('resources/taggers/la-pos-no-BPEmb',
              mini_batch_size=64, max_epochs=150, patience=5,
              learning_rate=0.1, min_learning_rate=1e-4,
              embeddings_storage_mode='gpu',
              monitor_train=True, monitor_test=True,
              anneal_with_restarts=True, use_amp=False)
