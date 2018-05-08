#!/usr/bin/env bash
mkdir -p raw/cmudict
wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b cmudict-0.7b
mv cmudict-0.7b raw/cmudict/