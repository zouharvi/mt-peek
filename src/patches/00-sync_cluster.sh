#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/mt-peek/

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/logs/train_mt_ende_s*_ner_*.log logs/
# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/nikita.tar.gz .
# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/data_out/CCrawl.de-en/fully_random/{r030,r000}/test_out.en .
