#!/usr/bin/env bash
mkdir -p raw
# download dataset
wget https://wit3.fbk.eu/archive/2016-01//texts/en/fr/en-fr.tgz
tar -zxf en-fr.tgz
mv en-fr iwslt2016
mv iwslt2016 raw/
rm en-fr.tgz