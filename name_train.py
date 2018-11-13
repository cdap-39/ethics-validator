import random

import spacy

# train_data = [
#     ("Maithripala Sirisena, presidential candidate of the New Democratic Front", [(0, 11, 'PERSON'),(12, 20, 'GPE')]),
#     ("He noted that the murder of Vithya in the Northern Province was a much-spoken about topic in the recent past", [(28, 34, 'PERSON')]),
#     ("Indian Prime Minister Narendra Modi, who is on a two day official visit to Sri Lanka", [(22, 30, 'PERSON'),(31, 35, 'PERSON')]),
#     ("Seya Sadevmi’s killer Saman Jayalath who was found guilty of murder was sentenced to death by Negombo High Court Judge Champa Janaki Rajarathna today.", [(0, 4, 'PERSON'),(5, 12, 'PERSON'),(28, 36, 'PERSON'),(22, 27, 'PERSON'),(19, 25, 'PERSON'),(26, 32, 'PERSON'),(33, 43, 'PERSON')]),
#     ("The case in which Saman Jayalath has been indicted by the Attorney General", [(18, 23, 'PERSON'),(24, 32, 'PERSON')]),
#     ("committing the murder of Seya Sadewmi of Kotadeniyawa was called to hear today before Negombo Provincial High Court Judge Champa Janaki Rajaratne.", [(25, 39, 'PERSON'),(30, 37, 'PERSON'),(122,128,'PERSON'),(129, 135, 'PERSON'),(136, 145, 'PERSON')]),
#     ("Dunesh Priyashantha who was once accused of the murder of Seya Sadewmi who was released on bail yesterday joined a media briefing in Colombo yesterday.", [(0, 6, 'PERSON'),(7, 19, 'PERSON'),(58, 62, 'PERSON'),(63, 70, 'PERSON')]),
# ]

train_data = [
    ("Uber blew through $1 million a week", {'entities':[(0, 4, 'ORG')]}),
    ("Android Pay expands to Canada", {'entities':[(0, 11, 'PRODUCT'), (23, 30, 'GPE')]}),
    ("Spotify steps up Asia expansion", {'entities':[(0, 8, "ORG"), (17, 21, "LOC")]}),
    ("Google Maps launches location sharing", {'entities':[(0, 11, "PRODUCT")]}),
    ("Google rebrands its business apps", {'entities':[(0, 6, "ORG")]}),
    ("look what i found on google! 😂", {'entities':[(21, 27, "PRODUCT")]})]


# TRAIN_DATA = [
#      ("Uber blew through $1 million a week", {'entities': [(0, 4, 'ORG')]}),
#      ("Google rebrands its business apps", {'entities': [(0, 6, "ORG")]})]

nlp = spacy.blank('en')
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(train_data)
    for text, annotations in train_data:
        nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk('./model')

