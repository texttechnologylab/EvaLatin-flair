import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from util import load_model, save_model

"""
Training script for continuing training of the FLAIR models from the EvaLatin challenge.

:author: Manuel Stoeckel
:institute: Text Technology Lab, Goethe University Frankfurt 
"""

tag_type = 'upos'
column_map = {0: 'id', 1: 'text', 2: 'lemma', 3: 'upos', 4: 'xpos', 5: 'feats',
              6: 'head', 7: 'deprel', 8: 'deps', 9: 'misc'}

if __name__ == '__main__':
    tagger: SequenceTagger = load_model('/path/to/state_dict.pt', '/path/to/tag_dictionary.pt')
    corpus: Corpus = ColumnCorpus('resources/data/', column_map,
                                  train_file='train.conllu',
                                  dev_file='dev.conllu',
                                  test_file='/path/to/EvaLatin/gold_data.conllu',
                                  comment_symbol='#')

    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.SGD)
    trainer.train('resources/taggers/model_name',
                  mini_batch_size=64, max_epochs=150, patience=3,
                  learning_rate=0.1, min_learning_rate=1e-4,
                  embeddings_storage_mode='gpu',
                  monitor_train=True, monitor_test=True,
                  anneal_with_restarts=True, use_amp=False)

    save_model(tagger)
