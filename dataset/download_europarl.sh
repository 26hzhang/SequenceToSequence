#!/usr/bin/env bash
mkdir -p raw
# download dataset
wget http://www.statmt.org/europarl/v7/fr-en.tgz
tar -zxf fr-en.tgz
mkdir europarl
mv europarl-v7.fr-en.en europarl/
mv europarl-v7.fr-en.fr europarl/
mv europarl raw/
rm fr-en.tgz
