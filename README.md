This repository forks from Tom Crossland's 2021 publication work(https://arxiv.org/abs/2107.00665).

The original contribution to this repository can be found in the folder "masters".

The attempt was to adapt existing models to further performance on the task of both:
    - Named Entity Recognition
    - Relation Extraction
from astrophysics data.

Main contributions were:
    - Fine-tuning multiple BERT models on available astrophysics articles.
        - astroBERT/bert-base-cased
        - fine-tuned on 2,500 texts sentence-split into maximum 256 words, with 16 words overlap. 
    - Building a TFIDF model on 100,000 articles.
    - Named Entity Recognition.
        - BERT model fine-tuned for named entity recognition.
        - Bi-directional LSTMs using Word2Vec embeddings and character encoding (reproduction of previous work).
        - Bi-directional LSTMs using BERT embeddings and word-level character encoding.
            - Numerical flags as additional feature.
            - Boosting rare tokens within each sentence.
    - Relation Extraction.
        - Dense neural network with Bi-LSTM for encoding token windows with Word2Vec embeddings ( reproduction of previous work).
        - Binary classifiers for each class of relation.
            - TFIDF as additional feature.
            - Numerical flags as additional feature.
        - Dense neural network with Bi-LSTM for encoding token windows with BERT embeddings.
            - TFIDF as additional feature.
            - Numerical flags as additional feature.
            - Boosting rare tokens within each sentence.
    - Evaluation of models.