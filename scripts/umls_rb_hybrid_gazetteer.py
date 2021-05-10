import os
import sys
import string
import collections
from collections import defaultdict
import spacy
from spacy.pipeline import EntityRuler
import csv
import scispacy
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker
import en_core_web_lg as lg
import en_core_sci_lg as scilg
import en_core_sci_sm as scism
from negspacy.negation import Negex
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
import pandas as pd
import re
import time
from datetime import datetime

def clean_text(text):
    new_text = re.sub('[^A-Za-z0-9 /-]+', '', text.lower())
    cl_text = re.sub(r'(?:(?<=\/) | (?=\/))','',new_text)
    #print('{}: {}'.format(cl_text, len(cl_text)))
    return cl_text

def join_words(words):
    
    new_text = words[0]
    special = ['-', '/']
    for i in range(1, len(words)):
        if words[i] in special or words[i-1] in special:
            new_text = new_text + words[i]
        else:
            new_text = new_text + ' ' + words[i]
        
    return new_text

def write_to_csv(_dict, output):
    
    with open(output, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in _dict.items():
            writer.writerow([key, value])

def write_to_csv_pos_neg_final(_dict_positive, _dict_negative, _dict_final, prefix, output):
        
    new_cdc_symptoms = ['PAT_ID', 'NOTE_ID']
    
    for file, sym in _dict_positive.items():
        for symptom, value in sym.items():
            words_pos = symptom.split()
            words_pos.append('p')
            words_pos.insert(0, prefix)
            words_neg = symptom.split()
            words_neg.append('n')
            words_neg.insert(0, prefix)
            words_neutral = symptom.split()
        
            new_pos = '_'.join(words_pos)
            new_neg = '_'.join(words_neg)
            new_neutral = '_'.join(words_neutral)
        
            new_cdc_symptoms.append(new_pos)
            #new_cdc_symptoms.append(new_neutral)
            new_cdc_symptoms.append(new_neg)
      
        break
    
    with open(output, 'w', newline = '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(new_cdc_symptoms)
        #count = 0
        for key, value in _dict_positive.items():
            pat_id, note_id = key.split('_')
            note_id = note_id.replace('.txt', '')
            
            li_men = [pat_id, note_id]
            for key2, value2 in value.items():
                #if count == 0:
                #    print(key2)
                li_men.extend([_dict_positive[key][key2], _dict_negative[key][key2]])
            writer.writerow(li_men)
            #count = count + 1

def init_dict(_dict, notes_for_training, dict_gold_cdc_cui):
    
    for file in notes_for_training:
        #name = file.replace('.source', '')
        name = file.strip()
        for k, v in dict_gold_cdc_cui.items():
            _dict[name][k] = 0


# Uses a dictionary of list
def check_dict(_dict, element):
    
    for k, v in _dict.items():
        if element in v:
            return k
    
    return 'null'


def update_mdict(_dict, file, parent):

    _dict[file][parent] = 1


def update_final_mdict(_dict_final, _dict_positive, _dict_negative):
    
    for key, value in _dict_final.items():
        for key2, value2 in value.items():
            pos = _dict_positive[key][key2]
            neg = _dict_negative[key][key2]
            
            if (pos == 1):
                _dict_final[key][key2] = 1
            if (pos == 0) and (neg == 1):
                _dict_final[key][key2] = -1


def diff(li1, li2):

    li_dif = []
    for x in li1:
        if x not in li2:
            li_dif.append(x)
            
    return li_dif

def load_gaz_cdc(filename):
    
    nlp = spacy.load('en_core_web_lg')
    _dict = defaultdict(list)
    
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            x_new = clean_text(row[0])
            words = [token.lemma_ for token in nlp(x_new.strip())]
            new_str = join_words(words)
            _dict[row[1].lower()].append(new_str)
            
    return _dict

def create_gazetteer(filename):
    
    nlp = spacy.load('en_core_web_lg')
    gaz = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row_0_new = clean_text(row[0])
            row_1_new = clean_text(row[1])
            
            words_0 = [token.lemma_ for token in nlp(row_0_new.strip())]
            new_str_0 = join_words(words_0)
            
            if new_str_0 not in gaz:
                gaz.append(new_str_0)
                
    return gaz

def get_gaz_matches(tokenizer, phrases, texts, span_left, span_right):
    matcher = PhraseMatcher(tokenizer.vocab)
    matcher.add("Phrase", None, *phrases)
    for text in texts:
        doc = tokenizer(text.lower())
        for w in doc:
            _ = doc.vocab[w.text]
        matches = matcher(doc)
        for ent_id, start, end in matches:
            if start - span_left >= 0:
                yield (ent_id, doc[start:end].text, doc[(start - span_left):(end + span_right)].text)
            else:
                yield (ent_id, doc[start:end].text, doc[start:(end + span_right)].text)


def create_matcher(nlp):
    
    patterns = [[{'LEMMA': 'abdominal'}, {'LEMMA': 'bloating'}],
                [{'LEMMA': 'abdominal'}, {'LEMMA': 'bloat'}],
                [{'LEMMA': 'abdominal'}, {'LEMMA': 'cramping'}],
                [{'LEMMA': 'abdominal'}, {'LEMMA': 'cramp'}],
                [{'LEMMA': 'abdominal'}, {'LEMMA': 'pain'}],
                ############## possible variations ############
                [{'LEMMA': 'stomach'}, {'LEMMA': 'pain'}],
                [{'LEMMA': 'stomach'}, {'LEMMA': 'bloating'}],
                [{'LEMMA': 'stomach'}, {'LEMMA': 'bloat'}],
                [{'LEMMA': 'stomach'}, {'LEMMA': 'cramping'}],
                [{'LEMMA': 'stomach'}, {'LEMMA': 'cramp'}],
                [{'LEMMA': 'gastric'}, {'LEMMA': 'bloating'}],
                [{'LEMMA': 'gastric'}, {'LEMMA': 'bloat'}],
                [{'LEMMA': 'gastric'}, {'LEMMA': 'cramping'}],
                [{'LEMMA': 'gastric'}, {'LEMMA': 'cramp'}],
                [{'LEMMA': 'gastric'}, {'LEMMA': 'pain'}],
                ###############################################
                [{'LEMMA': 'ache'}, {'POS': 'CCONJ'}, {'LEMMA': 'pain'}],
                [{'LEMMA': 'achey'}], 
                [{'LEMMA': 'achiness'}],
                [{'LEMMA': 'achy'}],
                [{'LEMMA': 'asthenia'}],
                [{'LEMMA': 'calf'}, {'LEMMA': 'pain'}],
                ############# use case: can't and cannot ############
                [{'LEMMA': 'can'}, {'LEMMA': 'not'}, {'LEMMA': 'breathe'}], 
                [{'LEMMA': 'can'}, {'LEMMA': 'not'}, {'LEMMA': 'smell'}],
                [{'LEMMA': 'can'}, {'LEMMA': 'not'}, {'LEMMA': 'taste'}],
                #####################################################
                [{'LOWER': 'catharsis'}],
                [{'LEMMA': 'cephalic'}, {'LEMMA': 'pain'}],
                [{'LEMMA': 'congested'}],
                [{'LEMMA': 'congestion'}, {'POS': 'CCONJ'}, {'LEMMA': 'runny'}, {'LEMMA': 'nose'}],
                [{'LEMMA': 'cramp'}],
                [{'LEMMA': 'diaphoretic'}],
                [{'LEMMA': 'discombobulation'}],
                [{'LEMMA': 'distaste'}],
                [{'LEMMA': 'doe'}],
                [{'LEMMA': 'dysphagia'}],
                [{'LEMMA': 'elevated'}, {'LEMMA': 'temp'}],
                ############## possible variations. use cases: high, higher ############
                [{'LEMMA': 'high'}, {'LEMMA': 'temp'}],
                [{'LEMMA': 'elevated'}, {'LEMMA': 'temperature'}],
                [{'LEMMA': 'high'}, {'LEMMA': 'temperature'}],
                ########################################################################
                [{'LEMMA': 'exhaustion'}],
                [{'LOWER': 'f'}, {'IS_PUNCT': True}, {'LOWER': 'c'}],
                [{'LEMMA': 'febrile'}],
                [{'LEMMA': 'fever'}, {'POS': 'CCONJ'}, {'LEMMA': 'chill'}],
                [{'LEMMA': 'food'}, {'LEMMA': 'aversion'}],
                [{'LEMMA': 'gag'}],
                [{'LEMMA': 'gi'}, {'LEMMA': 'symptom'}],
                [{'LEMMA': 'headache'}],
                [{'LEMMA': 'hyperthermia'}],
                [{'LEMMA': 'inspiratory'}, {'LEMMA': 'pain'}],
                [{'LEMMA': 'lack'}, {'TAG': 'IN'}, {'LEMMA': 'olfactory'}, {'LEMMA': 'sense'}],
                [{'LEMMA': 'loose'}, {'LEMMA': 'stool'}],
                [{'LEMMA': 'loss'}, {'TAG': 'IN'}, {'LEMMA': 'smell'}],
                [{'LEMMA': 'loss'}, {'TAG': 'IN'}, {'LEMMA': 'taste'}],
                [{'LEMMA': 'low'}, {'LEMMA': 'energy'}],
                [{'LEMMA': 'metallic'}, {'LEMMA': 'taste'}],
                ############## possible variation ############
                [{'LEMMA': 'metal'}, {'LEMMA': 'taste'}],
                ###############################################
                [{'LEMMA': 'mucusy'}],
                [{'LEMMA': 'muscle'}, {'LEMMA': 'pain'}],
                [{'LEMMA': 'myalgia'}],
                [{'LEMMA': 'myodynia'}],
                [{'LOWER': 'n'}, {'IS_PUNCT': True}, {'LOWER': 'v'}],
                [{'LEMMA': 'nausea'}, {'POS': 'CCONJ'}, {'LEMMA': 'vomit'}],
                ############## possible variation ############
                [{'LEMMA': 'nauseous'}],
                ##############################################
                [{'LEMMA': 'neck'}, {'LEMMA': 'pain'}],
                [{'LEMMA': 'new'}, {'LEMMA': 'loss'}, {'POS': 'ADP', 'TAG': 'IN'}, {'LEMMA': 'taste'}, {'POS': 'CCONJ', 'TAG': 'CC'}, {'LEMMA': 'smell'}],
                [{'LEMMA': 'not', 'TAG': 'RB'}, {'LEMMA': 'eat'}],
                [{'LEMMA': 'not', 'TAG': 'RB'}, {'LEMMA': 'hungry'}],
                [{'LEMMA': 'out'}, {'LEMMA': 'of', 'TAG': 'IN'}, {'LOWER': 'it', 'TAG': 'PRP'}],
                [{'LEMMA': 'pain'}, {'LEMMA': 'with', 'POS': 'ADP'}, {'LEMMA': 'inspiration'}],
                [{'LEMMA': 'painful'}, {'LEMMA': 'swallow'}],
                [{'LEMMA': 'pleurisy'}],
                [{'LOWER': 'pnd'}],
                [{'LEMMA': 'poor'}, {'LEMMA': 'appetite'}],
                [{'LOWER': 'purgation'}],
                [{'LEMMA': 'queasiness'}],
                [{'LEMMA': 'queasy'}],
                [{'LEMMA': 'respiratory'}, {'LEMMA': 'difficulty'}],
                [{'LEMMA': 'respiratory'}, {'LEMMA': 'distress'}],
                [{'LEMMA': 'retch'}],
                [{'LEMMA': 'rhinorrhea'}],
                [{'LEMMA': 'rigor'}],
                [{'LEMMA': 'rhinorrhea'}],
                [{'LEMMA': 'runny'}, {'LEMMA': 'nose'}],
                [{'LEMMA': 'scratchy'}, {'LEMMA': 'throat'}],
                [{'LEMMA': 'shake'}],
                [{'LEMMA': 'shakey'}],
                [{'LEMMA': 'shakiness'}],
                [{'LEMMA': 'short'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}],
                [{'LEMMA': 'shortness'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}],
                [{'LEMMA': 'shortness'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}, {'POS': 'CCONJ'}, {'LEMMA': 'difficulty'}, {'LEMMA': 'breathing'}],
                ############## possible variation ############
                [{'LEMMA': 'short'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}, {'POS': 'CCONJ'}, {'LEMMA': 'difficulty'}, {'LEMMA': 'breathing'}],
                ##############################################
                [{'LEMMA': 'sob'}],
                [{'LEMMA': 'sore'}, {'LEMMA': 'throat'}],
                [{'LEMMA': 'soreness'}],
                [{'LEMMA': 'stuffy'}, {'LEMMA': 'nose'}],
                [{'LEMMA': 'sweaty'}],
                ############## possible variation ############
                [{'LEMMA': 'sweat'}],
                [{'LEMMA': 'sweating'}],
                ##############################################
                [{'LEMMA': 'throw'}, {'LEMMA': 'up'}],
                [{'LOWER': 'tired'}], 
                [{'LOWER': 'tiredness'}],
                [{'LOWER': 'tremor'}],
                [{'LOWER': 'trot'}],
                [{'LEMMA': 'trouble'}, {'LEMMA': 'breathing'}],
                [{'LEMMA': 'watery'}, {'LEMMA': 'stool'}],
                [{'LEMMA': 'weariness'}],
                [{'LEMMA': 'wear'}, {'LEMMA': 'out'}]
                ]
    
    matcher = Matcher(nlp.vocab)
    
    count = 0
    for pattern in patterns:
        matcher.add(pattern[0].items()[1], None, pattern)
        count = count + 1
    
    print('patterns added: {}'.format(count))
    
    return matcher

def create_ruler(nlp):
    
    cdc_symptoms = ['cough', 'dyspnea', 'fatigue', 'headaches', 'aches', 'taste smell loss', 'sore throat', 
                    'rhinitis congestion', 'nausea vomiting', 'diarrhea', 'fever']

    patterns = [ {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'abdominal'}, {'LEMMA': 'bloating'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'abdominal'}, {'LEMMA': 'bloat'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'abdominal'}, {'LEMMA': 'cramping'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'abdominal'}, {'LEMMA': 'cramp'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'abdominal'}, {'LEMMA': 'pain'}]},
                 ############## possible variations ############
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'stomach'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'stomach'}, {'LEMMA': 'bloating'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'stomach'}, {'LEMMA': 'bloat'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'stomach'}, {'LEMMA': 'cramping'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'stomach'}, {'LEMMA': 'cramp'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gastric'}, {'LEMMA': 'bloating'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gastric'}, {'LEMMA': 'bloat'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gastric'}, {'LEMMA': 'cramping'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gastric'}, {'LEMMA': 'cramp'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gastric'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'intestinal'}, {'LEMMA': 'bloating'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'intestinal'}, {'LEMMA': 'bloat'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'intestinal'}, {'LEMMA': 'cramping'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'intestinal'}, {'LEMMA': 'cramp'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'intestinal'}, {'LEMMA': 'pain'}]},
                 ###############################################
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'ache'}, {'POS': 'CCONJ'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'achey'}]}, 
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'achiness'}]},
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'achy'}]},
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'asthenia'}]},
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'calf'}, {'LEMMA': 'pain'}]},
                 ############# use case: can't and cannot ############
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'can'}, {'LEMMA': 'not'}, {'LEMMA': 'breathe'}]}, 
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'can'}, {'LEMMA': 'not'}, {'LEMMA': 'smell'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'can'}, {'LEMMA': 'not'}, {'LEMMA': 'taste'}]},
                 #####################################################
                 {'label': cdc_symptoms[9], 'pattern': [{'LOWER': 'catharsis'}]},
                 {'label': cdc_symptoms[3], 'pattern': [{'LEMMA': 'cephalic'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[7], 'pattern': [{'LEMMA': 'congested'}]},
                 {'label': cdc_symptoms[7], 'pattern': [{'LEMMA': 'congestion'}, {'POS': 'CCONJ'}, {'LEMMA': 'runny'}, {'LEMMA': 'nose'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'cramp'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'diaphoretic'}]},
                 {'label': cdc_symptoms[3], 'pattern': [{'LEMMA': 'discombobulation'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'distaste'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'doe'}]},
                 {'label': cdc_symptoms[6], 'pattern': [{'LEMMA': 'dysphagia'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'elevated'}, {'LEMMA': 'temp'}]},
                 ############## possible variations. use cases: high, higher ############
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'high'}, {'LEMMA': 'temp'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'elevated'}, {'LEMMA': 'temperature'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'high'}, {'LEMMA': 'temperature'}]},
                 ########################################################################
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'exhaustion'}]},
                 ##################### possible variations #########################
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'exertion'}]},
                 ###################################################################
                 {'label': cdc_symptoms[10], 'pattern': [{'LOWER': 'f'}, {'IS_PUNCT': True}, {'LOWER': 'c'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'febrile'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'fever'}, {'POS': 'CCONJ'}, {'LEMMA': 'chill'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'food'}, {'LEMMA': 'aversion'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gag'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'gi'}, {'LEMMA': 'symptom'}]},
                 {'label': cdc_symptoms[3], 'pattern': [{'LEMMA': 'headache'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'hyperthermia'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'inspiratory'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'lack'}, {'TAG': 'IN'}, {'LEMMA': 'olfactory'}, {'LEMMA': 'sense'}]},
                 {'label': cdc_symptoms[9], 'pattern': [{'LEMMA': 'loose'}, {'LEMMA': 'stool'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'loss'}, {'TAG': 'IN'}, {'LEMMA': 'smell'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'loss'}, {'TAG': 'IN'}, {'LEMMA': 'taste'}]},
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'low'}, {'LEMMA': 'energy'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'metallic'}, {'LEMMA': 'taste'}]},
                 ############## possible variation ############
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'metal'}, {'LEMMA': 'taste'}]},
                 ###############################################
                 {'label': cdc_symptoms[7], 'pattern': [{'LEMMA': 'mucusy'}]},
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'muscle'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'myalgia'}]},
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'myodynia'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LOWER': 'n'}, {'IS_PUNCT': True}, {'LOWER': 'v'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'nausea'}, {'POS': 'CCONJ'}, {'LEMMA': 'vomit'}]},
                 ############## possible variation ############
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'nauseous'}]},
                 ##############################################
                 {'label': cdc_symptoms[4], 'pattern': [{'LEMMA': 'neck'}, {'LEMMA': 'pain'}]},
                 {'label': cdc_symptoms[5], 'pattern': [{'LEMMA': 'new'}, {'LEMMA': 'loss'}, {'POS': 'ADP', 'TAG': 'IN'}, {'LEMMA': 'taste'}, {'POS': 'CCONJ', 'TAG': 'CC'}, {'LEMMA': 'smell'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'not', 'TAG': 'RB'}, {'LEMMA': 'eat'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'not', 'TAG': 'RB'}, {'LEMMA': 'hungry'}]},
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'out'}, {'LEMMA': 'of', 'TAG': 'IN'}, {'LOWER': 'it', 'TAG': 'PRP'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'pain'}, {'LEMMA': 'with', 'POS': 'ADP'}, {'LEMMA': 'inspiration'}]},
                 {'label': cdc_symptoms[6], 'pattern': [{'LEMMA': 'painful'}, {'LEMMA': 'swallow'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'pleurisy'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LOWER': 'pnd'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'poor'}, {'LEMMA': 'appetite'}]},
                 {'label': cdc_symptoms[9], 'pattern': [{'LOWER': 'purgation'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'queasiness'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'queasy'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'respiratory'}, {'LEMMA': 'difficulty'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'respiratory'}, {'LEMMA': 'distress'}]},
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'retch'}]},
                 {'label': cdc_symptoms[7], 'pattern': [{'LEMMA': 'rhinorrhea'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'rigor'}]},
                 {'label': cdc_symptoms[7], 'pattern': [{'LEMMA': 'runny'}, {'LEMMA': 'nose'}]},
                 {'label': cdc_symptoms[6], 'pattern': [{'LEMMA': 'scratchy'}, {'LEMMA': 'throat'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'shake'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'shakey'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'shakiness'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'short'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'shortness'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'shortness'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}, {'POS': 'CCONJ'}, {'LEMMA': 'difficulty'}, {'LEMMA': 'breathing'}]},
                 ############## possible variation ############
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'short'}, {'POS': 'ADP'}, {'LEMMA': 'breath'}, {'POS': 'CCONJ'}, {'LEMMA': 'difficulty'}, {'LEMMA': 'breathing'}]},
                 ##############################################
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'sob'}]},
                 {'label': cdc_symptoms[6], 'pattern': [{'LEMMA': 'sore'}, {'LEMMA': 'throat'}]},
                 {'label': cdc_symptoms[6], 'pattern': [{'LEMMA': 'soreness'}]},
                 {'label': cdc_symptoms[7], 'pattern': [{'LEMMA': 'stuffy'}, {'LEMMA': 'nose'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'sweaty'}]},
                 ############## possible variation ############
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'sweat'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LEMMA': 'sweating'}]},
                 ##############################################
                 {'label': cdc_symptoms[8], 'pattern': [{'LEMMA': 'throw'}, {'LEMMA': 'up'}]},
                 {'label': cdc_symptoms[2], 'pattern': [{'LOWER': 'tired'}]}, 
                 {'label': cdc_symptoms[2], 'pattern': [{'LOWER': 'tiredness'}]},
                 {'label': cdc_symptoms[10], 'pattern': [{'LOWER': 'tremor'}]},
                 {'label': cdc_symptoms[9], 'pattern': [{'LOWER': 'trot'}]},
                 {'label': cdc_symptoms[1], 'pattern': [{'LEMMA': 'trouble'}, {'LEMMA': 'breathing'}]},
                 {'label': cdc_symptoms[9], 'pattern': [{'LEMMA': 'watery'}, {'LEMMA': 'stool'}]},
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'weariness'}]},
                 {'label': cdc_symptoms[2], 'pattern': [{'LEMMA': 'wear'}, {'LEMMA': 'out'}]}
                ]
    
    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns(patterns)
    
    return ruler

def create_gazetteer_lexicon_count_dictionary(gazetteer):
    
    dict_lexicon_count = defaultdict(list)
    for x in gazetteer:
        dict_lexicon_count[x] = 0
    
    return dict_lexicon_count


def mention_using_gaz(gaz_csv, notes_for_training, doc_folder, dict_gaz, prefix, output):
    
    cdc_symptoms = ['cough', 'dyspnea', 'fatigue', 'headaches', 'aches', 'taste smell loss', 'sore throat', 
                    'rhinitis congestion', 'nausea vomiting', 'diarrhea', 'fever']
    
    nlp = spacy.blank('en')
    nlp.vocab.lex_attr_getters = {}
    gaz = create_gazetteer(gaz_csv)
    patterns = [nlp.make_doc(text) for text in gaz]
    
    nlp_lemma = lg.load()
    
    dict_files_positive = defaultdict(dict)
    init_dict(dict_files_positive, notes_for_training, dict_gaz)
    
    dict_files_negative = defaultdict(dict)
    init_dict(dict_files_negative, notes_for_training, dict_gaz)
    
    dict_files_final = defaultdict(dict)
    init_dict(dict_files_final, notes_for_training, dict_gaz)
    
    span_left = 10
    span_right = 2
    span_left_temporal = 15
    span_right_temporal = 15
    
    nlp_neg = scilg.load()
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    nlp_neg.add_pipe(linker)
    #matcher = create_matcher(nlp_neg)
    #nlp_neg.add_pipe(matcher)
    ruler = create_ruler(nlp_neg)
    nlp_neg.add_pipe(ruler)
    negex = Negex(nlp_neg, language = "en_clinical_sensitive", chunk_prefix = ["without", "no"])
    negex.add_patterns(preceding_negations = ['deny'])
    nlp_neg.add_pipe(negex, last = True)
    
    for file in notes_for_training:
        with open(os.path.join(doc_folder, file), 'r') as f:
            for ent_id, men, text in get_gaz_matches(nlp.tokenizer, patterns, f, span_left, span_right):
                words = [token.lemma_ for token in nlp_lemma(men.lower().strip())]
                sent_words = [token.lemma_ for token in nlp_lemma(text.lower().strip())]
                new_str = join_words(words)
                new_str_sent = join_words(sent_words)
                
                #print(new_str)
                # additional conditions to detect use case like: '... no fever. Patient has sore throat ...'
                split_strings = new_str_sent.split('.')
                for sub in split_strings:
                    threshold = 2
                    if (len(sub.split()) >= threshold):
                        neg = nlp_neg(sub)
                        for e in neg.ents:
                            parent = check_dict(dict_gaz, e.text)
                            
                            if (parent != 'null'):
                                name = file.strip()
                                content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + '\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, parent)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, parent)
                            
                            # for possible variations of symptoms in the lexicon not present in the UMLS
                            if (parent == 'null') and (e.label_ in cdc_symptoms):
                                name = file.strip()
                                content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + '\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, e.label_)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, e.label_)
                                
    
    update_final_mdict(dict_files_final, dict_files_positive, dict_files_negative)
    write_to_csv_pos_neg_final(dict_files_positive, dict_files_negative, dict_files_final, prefix, output)
    
    #print(dict_files)
    return dict_files_final

def read_list_of_notes(notes_csv):

    notes_list = []
    with open(notes_csv, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            notes_list.append(row[0])
    
    #print(notes_list)
    return notes_list

def main():
    
    gaz_csv = sys.argv[1]
    notes_csv = sys.argv[2]
    doc_folder = sys.argv[3]
    output_gaz = sys.argv[4]
    prefix = sys.argv[5]
    
    now = datetime.now()
    timestamp = str(datetime.timestamp(now))
    output_ts = output_gaz + '_' + timestamp + '.csv'
    
    notes_list = read_list_of_notes(notes_csv)
    
    tic = time.perf_counter()
    dict_gaz_cdc = load_gaz_cdc(gaz_csv)
    gaz_men = mention_using_gaz(gaz_csv, notes_list, doc_folder, dict_gaz_cdc, prefix, output_ts)
    toc = time.perf_counter()
    print(f"Finished! Annotation done in {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    main()