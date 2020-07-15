import logging
from pathlib import Path
from typing import List, Union

import flair
import torch
from flair.embeddings import TokenEmbeddings, BytePairEmbeddings, PooledFlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger

from embeddings import WordToVecFormatEmbeddings

"""
Utility script for loading and saving the FLAIR models from the EvaLatin challenge.

:author: Manuel Stoeckel
:institute: Text Technology Lab, Goethe University Frankfurt 
"""

log = logging.getLogger("flair")

tag_type = 'upos'
column_map = {0: 'id', 1: 'text', 2: 'lemma', 3: 'upos', 4: 'xpos', 5: 'feats',
              6: 'head', 7: 'deprel', 8: 'deps', 9: 'misc'}


def load_model(state_dict_path, tag_dictionary_path, cache_path=flair.cache_root) -> SequenceTagger:
    cache_path = Path(cache_path) / 'embeddings'

    log.info("Loading embeddings")
    stacked_embeddings: List[TokenEmbeddings] = [
        WordToVecFormatEmbeddings(cache_path / 'la-2M-300-glove.vec'),
        WordToVecFormatEmbeddings(cache_path / 'la-2M-300-fastText.vec'),
        WordToVecFormatEmbeddings(cache_path / 'la-2M-300-wang2vec-50it.vec'),
        BytePairEmbeddings('la', 300, 5000, cache_dir=cache_path),
        PooledFlairEmbeddings(str(cache_path / 'lm-historic-latin-forward-v1.0.pt')),
        PooledFlairEmbeddings(str(cache_path / 'lm-historic-latin-backward-v1.0.pt'))
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=stacked_embeddings)

    log.info("Loading tag_dictionary")
    tag_dictionary = torch.load(tag_dictionary_path)

    log.info("Instantiating tagger")
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type
    )

    log.info("Loading state_dict")
    tagger.load_state_dict(torch.load(state_dict_path))

    return tagger


def save_model(tagger: SequenceTagger, path: Union[str, Path], base_name):
    path = Path(path)
    log.info("Saving tag_dictionary")
    torch.save(tagger.tag_dictionary, path / (base_name + "tag_dictionary.pt"))

    log.info("Saving state_dict")
    torch.save(tagger.state_dict(), path / (base_name + "tag_dictionary.pt"))
