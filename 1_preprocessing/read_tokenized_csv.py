with open('b2w_tokenized_none.csv', 'r') as f:
        tokenized_corpus = []
        for l in f:
            line = l.replace('\n','').split(',')
            tokenized_corpus.append(line)
            
tokenized_corpus