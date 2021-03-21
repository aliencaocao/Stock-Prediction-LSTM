# imports
import spacy

# tokenization
nlp = spacy.load('en_core_web_lg')
doc = nlp('I am flying to Manila.')
print([w.text for w in doc])

# lemmatization

doc = nlp('this product integrates both libraries for downloading and applying patches')
for token in doc:
    print(token.text, token.lemma_)

# part of speech tagging
doc = nlp('I have flown to Singapore. I am flying to Manila.')
for token in doc:
    print(token.text, token.pos_, spacy.explain(token.tag_))

print([w.text for w in doc if w.tag_ == 'VBG' or w.tag_ == 'VB'])

for sent in doc.sents:
    print([sent[i] for i in range(len(sent))])

doc = nlp('The Golden Gate Bridge is an iconic landmark in San Fransisco.')
print([w.text for w in doc])

with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[1:4])
with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[7:9])
for token in doc:
    print(token.text, token.lemma_, token.pos_)

# dependency parsing
doc = nlp('I want a green apple.')
for token in doc:
    print(token.text, token.pos_, token.dep_, spacy.explain(token.dep_))

# entity recognition

doc = nlp('The firm earned $1.5 million in 2017. They lost $1.2 million.')
phrase = ''
for token in doc:
    if token.tag_ == '$':
        phrase = token.text
        i = token.i + 1
        while doc[i].tag_ == 'CD':
            phrase += doc[i].text + ' '
            i += 1
        phrase = phrase[:-1]
        print(phrase)

# Word similarity
print(nlp('apple').similarity(nlp('banana')))

doc = nlp('I want a green apple.')
print(doc.similarity(doc[2:5]))
print(nlp('gay').similarity(nlp('lesbian')))
