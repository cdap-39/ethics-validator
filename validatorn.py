import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)


matcher.add('Disclosing private details of Victim', None,
                    [{'LOWER': 'rape'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'of'}, {'IS_ASCII': True, 'OP': '*'},{'ENT_TYPE': 'PERSON'}],
                    [{'LOWER': 'rape'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'of'}, {'IS_ASCII': True, 'OP': '*'}, {'ENT_TYPE': 'PERSON'}],
                    [{'LOWER': 'murder'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'of'}, {'IS_ASCII': True, 'OP': '*'},{'ENT_TYPE': 'PERSON'}],
                    [{'LOWER': 'murder'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'of'}, {'IS_ASCII': True, 'OP': '*'},{'ENT_TYPE': 'PERSON'}],
                    [{'LOWER': 'victim'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'as'}, {'IS_ASCII': True, 'OP': '*'},{'ENT_TYPE': 'PERSON'}],
                    [{'LOWER': 'victim'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'as'}, {'IS_ASCII': True, 'OP': '*'},{'ENT_TYPE': 'PERSON'}],
                    [{'LOWER': 'raping'}, {'IS_ASCII': True, 'OP': '*'}, {'ENT_TYPE': 'PERSON'}]
            )

#hanging, jumping, poison
matcher.add('Disclosing details dealing with Social Issues', None,
                    [{'LOWER': 'committed'},{'LOWER': 'suicide'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'by'}],
            )

matcher.add('Disclosing details of race, caste, religion, sexual orientation, physical and mental illness or disabilities', None,
                    [{'LOWER': 'gay'},{'LOWER': 'patner'}],
                    [{'LOWER': 'gay'},{'LOWER': 'persons'}],
                    [{'LOWER': 'gay'},{'LOWER': 'males'}],
                    [{'LOWER': 'gay'},{'LOWER': 'youth'}],
                    [{'ENT_TYPE': 'PERSON'},{'IS_ASCII': True, 'OP': '*'},{'LOWER': 'with'}, {'IS_ASCII': True, 'OP': '*'}, {'LOWER': 'disabilities'}]
            )


doc = nlp(u'Several demonstrations and satyagraha campaigns were carried out in various parts of the country today as well in protest of the rape and murder of school student, Vidya Sivaloganathan, in Jaffna.\r\nThe demonstrators demand that the law be strictly enforced over the incident.Students and teachers of Sri Shanmugha College Trincomalee staged a demonstration this morning in protest of the murder of Vidya Sivaloganathan. A protest was held in the Kiliveddi area in Trincomalee urging that the law be strictly be enforced against the suspects of the rape and murder.Protests were also held in the Akkaraipatthu, Alaidi Vembu and Adalachchennai areas of the Ampara District today. The Ceylon Teachers\u2019 Union also organized a protest that was staged in Armour Street, Colombo today demanding justice for the murdered victim.General Secretary of the Ceylon Teachers\u2019 Union, Joseph Stalin stated that an investigative committee must be appointed to conduct extensive investigations, not only to the rape of Vidya, but all cases where students and women were victim to sexual harassment in the post-war period.Rights group, Women for Rights emphasized that it is a regretting fact that harassment against women in the south has now spread to the north.President of Women for Rights, Samanmalee Gunasinghe at a media briefing stated that what was happening in the South has now spread to the North as well and if Vindya\u2019s incident is being used to to incite racism, or bury whatever freedom the people have won, be it in the North or the South, it should not happen.SLFP National Organizer, Susil Premajayantha stated that it is with utmost disgust that this incident is condemned. He further stated that it has not been heard where of nine men gang raping a school student and that this implies the fall of John Amaratunga\u2019s Police. He also noted that the President, the first citizen of this country, when visiting Jaffna to attend another ceremony met with the mother and family of the victim and promised that justice willbe served but the subject minister is compiling a report.He further noted thatbecause law enforcement officials did not act at the correct time and the correct way, the public is outraged.Minister of Mass Media, Shantha Bandara stated that His Excellency the President visited Jaffna in haste just for this matter and made a clear statement that the perpetrators will be punished under a special court. He added that this should happen the president should take up the leadership to immediately put the decision taken into effect before the country forgets the incident and these inhumane suspects be punished in public.')
matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, span.text)

for ent in doc.ents:
    print(ent.text, ent.label_)



def getViolation(newPredict):
    # print(newPredict)  # prints all the data
    print("this predicted values")
    aList = []
    for i in range(len(newPredict)):
        doc = nlp(newPredict[i]["content"])
        for ent in doc.ents:
            print(ent.text, ent.label_)
        matches = matcher(doc)
        violate = []
        status = False
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            newRule = { "rule":string_id, "sentence":span.text}
            if newRule not in violate:
                violate.append(newRule)
            status = True
            print(match_id, string_id, span.text)
        newPredict[i]['media_ethics'] = { "violations":status ,"reason": violate}
        aList.append( newPredict[i])

    print(aList)
    jsonList = json.dumps(aList, separators=(',', ':'))
    return jsonList


#server
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from io import BytesIO

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        try:
            body = json.loads(self.rfile.read(content_length))
            print(body['data'])
            ethics = getViolation(body['data'])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(str(ethics))
        except Exception as e:
            print(str(e))
            self.send_response(401)
            self.end_headers()
            self.wfile.write({'message': str(e)}, "utf-8")

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
print(httpd.server_name+ httpd.server_port)