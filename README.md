# Flair Classifier for the EvaLatin Challenge: POS Tagging of Latin Texts
## Instructions:
1. Download EvaLatin data and separate training files into train and dev splits.
2. Download TTLab embeddings: http://embeddings.texttechnologylab.org/
    - Download all Latin embeddings from the "Historical Latin Corpora Collection"
      and place them in a single folder, preferable '$HOME/.flair/embeddings/'.
    - Place the BPEmb embedding files in a sub-folder, eg. '$HOME/.flair/embeddings/la/'.
    - Folder structure should now be:
        ```
        $HOME/.flair/embeddings/
        ├── la
        │   ├── la.bpe.vs5000.d300.vec
        │   ├── la.bpe.vs5000.model
        │   └── la.bpe.vs5000.vocab
        ├── la-2M-300-fastText.vec
        ├── la-2M-300-glove.vec
        ├── la-2M-300-wang2vec-50it.vec
        ├── lm-historic-latin-backward-v1.0.pt
        └── lm-historic-latin-forward-v1.0.pt
        ```
3. Optional: download the trained model files
4. Start the training script `training.py` or evaluate a downloaded model with `evaluation.py`.
      
## Citation
M. Stoeckel, A. Henlein, W. Hemati, and A. Mehler, “Voting for POS tagging of Latin texts: Using the flair of FLAIR to better Ensemble Classifiers by Example of Latin,” in Proceedings of LT4HALA 2020 – 1st Workshop on Language Technologies for Historical and Ancient Languages, Marseille, France, 2020, pp. 130-135.
```biblatex
@InProceedings{Stoeckel:et:al:2020,
  author    = {Stoeckel, Manuel and Henlein, Alexander and Hemati, Wahed and Mehler, Alexander},
  title     = {{Voting for POS tagging of Latin texts: Using the flair of FLAIR to better Ensemble Classifiers by Example of Latin}},
  booktitle      = {Proceedings of LT4HALA 2020 - 1st Workshop on Language Technologies for Historical and Ancient Languages},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association (ELRA)},
  pages     = {130--135},
  abstract  = {Despite the great importance of the Latin language in the past, there are relatively few resources available today to develop modern NLP tools for this language. Therefore, the EvaLatin Shared Task for Lemmatization and Part-of-Speech (POS) tagging was published in the LT4HALA workshop. In our work, we dealt with the second EvaLatin task, that is, POS tagging. Since most of the available Latin word embeddings were trained on either few or inaccurate data, we trained several embeddings on better data in the first step. Based on these embeddings, we trained several state-of-the-art taggers and used them as input for an ensemble classifier called LSTMVoter. We were able to achieve the best results for both the cross-genre and the cross-time task (90.64\% and 87.00\%) without using additional annotated data (closed modality). In the meantime, we further improved the system and achieved even better results (96.91\% on classical, 90.87\% on cross-genre and 87.35\% on cross-time).},
  url       = {https://www.aclweb.org/anthology/2020.lt4hala-1.21},
  pdf       = {http://www.lrec-conf.org/proceedings/lrec2020/workshops/LT4HALA/pdf/2020.lt4hala-1.21.pdf}
}
``` 