import logging
import os
import sys
from pathlib import Path

from flair.datasets import DataLoader, ColumnDataset
from flair.models import SequenceTagger

from util import column_map, load_model

"""
Evaluation script the FLAIR models from the EvaLatin challenge.

:author: Manuel Stoeckel
:institute: Text Technology Lab, Goethe University Frankfurt 
"""

log = logging.getLogger("flair")

if __name__ == '__main__':
    try:
        for i in range(1, 4):
            if not os.path.isfile(sys.argv[i]):
                raise FileNotFoundError(f"File '{sys.argv[4]}' (argv[{i}) does not exist")

        if len(sys.argv) > 4:
            if not os.path.isfile(sys.argv[4]):
                raise FileNotFoundError(f"Output file '{sys.argv[4]}' does not exist")
            out_path = Path(sys.argv[3])
        else:
            out_path = None

        log.info("Loading dataset")
        dataset = ColumnDataset(
            Path(sys.argv[3]),
            column_map,
            encoding='utf8',
            comment_symbol='#',
            in_memory=False,
        )

        tagger: SequenceTagger = load_model(sys.argv[1], sys.argv[2])
        tagger.cuda()
        tagger.eval()

        log.info("Evaluating")
        result, _ = tagger.evaluate(dataset, out_path=out_path)
        print("\n".join([f"{m}:\t{s}" for m, s in zip(result.log_header.split(), result.log_line.split())]))
        print(result.detailed_results)
    except:
        print("Usage: python evaluation.py STATE_DICT_FILE TAG_DICTIONARY_FILE CONLL_FILE [OUPUT_FILE]")
        raise
