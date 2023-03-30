#!/usr/bin/env python3

data = [('NOUN', 4.7255), ('.', 1.68954), ('ADV', 0.37352), ('PRT', 0.26072), ('DET', 0.9162), ('ADP', 1.219), ('NUM', 0.54192), ('VERB', 1.46008), ('ADJ', 0.98546), ('PRON', 0.558), ('CONJ', 0.52212), ('X', 0.01686)]

data.sort(key=lambda x: x[1], reverse=True)

for name, token_count in data:
    print(name.capitalize(), "&", "&", f"{token_count:.3f}", "&", "&", "\\\\")