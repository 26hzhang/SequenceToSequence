#!/usr/bin/env bash
mkdir -p raw
# download dataset
wget https://wit3.fbk.eu/archive/2012-03//texts/en/fr/en-fr.tgz
tar -zxf en-fr.tgz
mv en-fr iwslt2012
mv iwslt2012 raw/
rm en-fr.tgz
# download test dataset (this is for ASR? I'm not sure, but it is useless, only english transcripts, not French version)
# wget https://wit3.fbk.eu/archive/2012-03-test//texts/en/fr/en-fr.tgz
# tar -zxf en-fr.tgz
