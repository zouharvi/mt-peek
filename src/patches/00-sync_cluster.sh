#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/mt-peek/

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/logs/train_mt_s*.log logs/