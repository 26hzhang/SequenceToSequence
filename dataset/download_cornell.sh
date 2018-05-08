#!/usr/bin/env bash
mkdir -p raw
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
mv cornell\ movie-dialogs\ corpus cornell
mv cornell raw/
rm cornell_movie_dialogs_corpus.zip
rm -rf __MACOSX