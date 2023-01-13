open('Astrology-ml/text.csv', 'w').close()

with open('Astrology-ml/horoscopes.csv', 'r', encoding="utf8") as fread, open('Astrology-ml/text.csv', 'a', encoding="utf8") as fwrite:
    count = 0
    for line in fread.readlines():
        count += 1
        if count == 1:
            continue
        data, sign, text = line.split(',', maxsplit=2)
        print(text)
        fwrite.write('[SG]')
        fwrite.write(text)
        fwrite.write('[EG]')
        fwrite.write('\n\n')
