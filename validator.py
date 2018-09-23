import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)


matcher.add('HelloWorldRule', None, 
                    [{'LOWER': 'rape'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'of'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'vidya'}],
            )

doc = nlp(u'A protest was staged at the Borella Cemetery Roundabout condemning the rape and murder of  Vidya Sivaloganadan, a schoolgirl in Jaffna. The protesters staged this demonstration demanding justice for Vidya Sivaloganadan.\r\nA list of 16 demands for women and children was also presented during this protest. The protest concluded following a candle light vigil for the victim.')
matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, span.text)

# for ent in doc.ents:
#     print(ent.text, ent.label_)