from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

pt_BR_blob = TextBlob(u'Eu amo essa biblioteca')

print(type(u'Eu amo essa biblioteca'))


en_blob = pt_BR_blob.translate(to='en')

print(en_blob.sentiment)

print(str(en_blob))
print(pt_BR_blob.detect_language())

blob = TextBlob(str(en_blob), analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)
blob = TextBlob(str(pt_BR_blob ), analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)

