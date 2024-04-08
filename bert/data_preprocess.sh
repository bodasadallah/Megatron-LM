#!/bin/bash

pip install nltk

python tools/preprocess_data.py \
       --input wiki_data/wikiextractor/data/AA/wiki_00 \
       --output-prefix bdata \
       --vocab-file bert/bert-large-uncased-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers 64 \
       --log-interval 10000