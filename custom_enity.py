from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


train_data = [
    ("Maithripala Sirisena, presidential candidate of the New Democratic Front", {'entities':[(0, 11, 'PERSON'),(12, 20, 'PERSON')]}),
    ("He noted that the murder of Vithya in the Northern Province was a much-spoken about topic in the recent past", {'entities':[(28, 34, 'PERSON')]}),
    ("Indian Prime Minister Narendra Modi, who is on a two day official visit to Sri Lanka", {'entities':[(22, 30, 'PERSON'),(31, 35, 'PERSON')]}),
    ("Seya Sadevmi’s killer Saman Jayalath who was found guilty of murder was sentenced to death by Negombo High Court Judge Champa Janaki Rajarathna today.", {'entities':[(0, 4, 'PERSON'),(5, 12, 'PERSON'),(28, 36, 'PERSON'),(22, 27, 'PERSON'),(19, 25, 'PERSON'),(26, 32, 'PERSON'),(33, 43, 'PERSON')]}),
    ("The case in which Saman Jayalath has been indicted by the Attorney General", {'entities':[(18, 23, 'PERSON'),(24, 32, 'PERSON')]}),
    ("committing the murder of Seya Sadewmi of Kotadeniyawa was called to hear today before Negombo Provincial High Court Judge Champa Janaki Rajaratne.", {'entities':[(25, 39, 'PERSON'),(30, 37, 'PERSON'),(122,128,'PERSON'),(129, 135, 'PERSON'),(136, 145, 'PERSON')]}),
    ("Dunesh Priyashantha who was once accused of the murder of Seya Sadewmi who was released on bail yesterday joined a media briefing in Colombo yesterday.", {'entities':[(0, 6, 'PERSON'),(7, 19, 'PERSON'),(58, 62, 'PERSON'),(63, 70, 'PERSON')]}),
    ("Dayasiri Jayasekera who was sworn in as Chief Minister of the North Western Provincial Council holds several ministry portfolios as well.", {'entities':[(0, 8, 'PERSON'),(9, 19, 'PERSON')]}),
    ("Dayasiri Jayasekara and Sarath Ekanayaka were sworn in before President Mahinda Rajapakse as chief ministers of the North Western and Central provinces respectively this morning.", {'entities':[(0, 8, 'PERSON'),(9, 19, 'PERSON'),(24, 30, 'PERSON'),(31, 40, 'PERSON'),(72, 79, 'PERSON'),(80, 89, 'PERSON')]}),
    ("The 3-member bench sentenced 7 suspects to death over the murder of 18-year-old schoolgirl Shivaloganathan Vidya in Punkudithivu, Kaytes in Jaffna.", {'entities':[(91, 105, 'PERSON'),(106, 111, 'PERSON')]}),
    ("Health Minister Dr. Rajitha Senaratne says 'legalization of abortion' was just a proposal that was discussed at the recent scientific sessions of the Perinatal Society of Sri Lanka.", {'entities':[(20, 27, 'PERSON'),(28, 37, 'PERSON')]}),
    ("Sub Inspector Sri Gajan has been interdicted due to providing help for the criminals who raped and murdered Shivaloganadan Vidya in Punkuditivu area in Kayts in Jaffna, to escape.", {'entities':[(14, 17, 'PERSON'),(18, 23, 'PERSON'),(108, 122, 'PERSON'),(123, 128, 'PERSON')]}),
    ("Ranjith Aluvihare will head the committee..", {'entities':[(0, 7, 'PERSON'),(8, 17, 'PERSON')]}),
    ("Prison Media spokesman Thushara Upuldeniya stated that the body of the detainee was discovered this morning.", {'entities':[(23, 31, 'PERSON'),(32, 42, 'PERSON')]}),
    ("Seya Sadewmi was found strangled to death on September 13, 2015 while she was reported missing from her home at Kotadeniyawa in Divulapitiya.", {'entities':[(0, 4, 'PERSON'),(5, 12, 'PERSON')]}),
    ("representative of the hospital stated, that 32 year old Dinesh Priyashantha alias Kondaya, arrived at the hospital under the protection of officers of the CID.", {'entities':[(56, 62, 'PERSON'),(63, 75, 'PERSON'),(82, 89, 'PERSON')]}),
    ("Aruna Udaya Shantha Pathirana alias Samayan was hospitalized after receiving injuries in his neck during the shooting.", {'entities':[(0, 5, 'PERSON'),(6, 11, 'PERSON'),(12, 19, 'PERSON'),(20, 29, 'PERSON'),(36, 43, 'PERSON')]}),
    ("The Prime Minister Ranil Wickremesinghe says that the Rajapaksa group including former President Mahinda Rajapaksa should apologize to women", {'entities':[(19, 24, 'PERSON'),(25, 39, 'PERSON'),(54, 63, 'PERSON')]}),
    ("Judge Rohini Walgama issued arrest warrants on the former chairman of the Tangalle Pradeshiya Sabha, Sampath Vidana Pathirana as he failed to appear in Court today", {'entities':[(6, 12, 'PERSON'),(13, 20, 'PERSON'),(101, 108, 'PERSON'),(109, 115, 'PERSON'),(116, 125, 'PERSON')]}),
    ("Vidana Pathirana who was indicted for the murder of a British national and the gang rape of his fiancée at a hotel in Tangalle was on bail.", {'entities':[(0, 6, 'PERSON'),(7, 16, 'PERSON')]}),
    ("Jayantha Jayasuriya says that an additional Solicitor General has been assigned to study the ongoing investigation in connection with the pro-LTTE statement made by former State Minister, Vijayakala Maheswaran.", {'entities':[(0, 8, 'PERSON'),(9, 19, 'PERSON'),(188, 198, 'PERSON'),(199, 209, 'PERSON')]}),
    ("IGP Pujith Jayasundera has handed over investigations into the complaints against State Minister Vijayakala Maheswaran to the Senior Crime DIG.", {'entities':[(4, 10, 'PERSON'),(11, 22, 'PERSON'),(97, 107, 'PERSON'),(108, 118, 'PERSON')]}),
    ("Police say that singer Priyani Jayasinghe has been stabbed to death.", {'entities':[(23, 30, 'PERSON'),(31, 41, 'PERSON')]}),
    ("The victim was Hendavitharana Selin Kumara, alias 'Thel Kumara'", {'entities':[(15, 29, 'PERSON'),(30, 35, 'PERSON'),(36, 42, 'PERSON'),(56, 62, 'PERSON')]}),
    ("Arunesh Thangarajah, 28, a Sri Lankan born person had been knifed to death on south-west London street last morning.", {'entities':[(0, 7, 'PERSON'),(8, 19, 'PERSON')]}),
    ("Former Minister Mahindananda Aluthgamage has been released on bail by Colombo Fort Magistrate Lanka Jayaratne.", {'entities':[(16, 28, 'PERSON'),(29, 40, 'PERSON'),(94, 99, 'PERSON'),(100, 109, 'PERSON')]}),
    ("Former SSP Hemantha Adhikari who is currently under investigation in connection with the murder of journalist Lasantha Wickramatunga has made a 3-hour confidential statement before the Mt. Lavinia Magistrate.", {'entities':[(11, 19, 'PERSON'),(20, 28, 'PERSON'),(111, 119, 'PERSON'),(120, 133, 'PERSON')]}),
    ("The wife of veteran singer Victor Ratnayake, Hashini Nilakshi Amendra has submitted a bail application.", {'entities':[(27, 33, 'PERSON'),(34, 44, 'PERSON'),(46, 53, 'PERSON'),(54, 62, 'PERSON'),(63, 70, 'PERSON')]}),
    ("Police CI Upul Dhammika and former Crime OIC SI Chitrasiri Sugathapala have been granted bail over a case relating to attempting to obtain a bribe of Rs. 8 million from a medical practitioner in the area.", {'entities':[(10, 14, 'PERSON'),(15, 23, 'PERSON'),(48, 58, 'PERSON'),(59, 70, 'PERSON')]}),
    ("Malaka Silva is accused of allegedly assaulting a foreign couple at a nightclub in Bambalapitiya.", {'entities':[(0, 6, 'PERSON'),(7, 12, 'PERSON')]}),



]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='en_core_web_sm', output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print('Losses', losses)

    # test the trained model
    for text, _ in train_data:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in train_data:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)

