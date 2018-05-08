#!/usr/bin/env bash
mkdir -p raw/twitter
wget https://github.com/marsan-ma/chat_corpus/blob/master/twitter_en.txt.gz\?raw=true
mv twitter_en.txt.gz?raw=true twitter_en.txt.gz
gunzip twitter_en.txt.gz twitter_en.txt
mv twitter_en.txt raw/twitter/
