from pathlib import Path
from typing import List

import flair
import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, BytePairEmbeddings, \
    PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from embeddings import WordToVecFormatEmbeddings

"""
Training script for training the FLAIR models from the EvaLatin challenge.

:author: Manuel Stoeckel
:institute: Text Technology Lab, Goethe University Frankfurt 
"""

tag_type = 'upos'
column_map = {0: 'id', 1: 'text', 2: 'lemma', 3: 'upos', 4: 'xpos', 5: 'feats',
              6: 'head', 7: 'deprel', 8: 'deps', 9: 'misc'}

if __name__ == '__main__':
    corpus: Corpus = ColumnCorpus('resources/data/', column_map,
                                  train_file='train.conllu',
                                  dev_file='dev.conllu',
                                  test_file='/path/to/EvaLatin/gold_data.conllu',
                                  comment_symbol='#')
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    cache_dir = Path(flair.cache_root) / 'embeddings'
    embedding_types: List[TokenEmbeddings] = [
        WordToVecFormatEmbeddings(cache_dir / 'la-2M-300-glove.vec'),
        WordToVecFormatEmbeddings(cache_dir / 'la-2M-300-fastText.vec'),
        WordToVecFormatEmbeddings(cache_dir / 'la-2M-300-wang2vec-50it.vec'),
        BytePairEmbeddings('la', 300, 5000, cache_dir=cache_dir),
        PooledFlairEmbeddings(cache_dir / 'lm-historic-latin-forward-v1.0.pt'),
        PooledFlairEmbeddings(cache_dir / 'lm-historic-latin-backward-v1.0.pt'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.SGD)
    trainer.train('resources/taggers/la-pos-final-full-5K',
                  mini_batch_size=64, max_epochs=150, patience=3,
                  learning_rate=0.1, min_learning_rate=1e-4,
                  embeddings_storage_mode='gpu',
                  monitor_train=True, monitor_test=True,
                  anneal_with_restarts=True, use_amp=False)
