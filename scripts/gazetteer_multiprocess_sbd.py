import os
import sys
import string
from string import punctuation
import collections
from collections import defaultdict
import spacy
from spacy.pipeline import EntityRuler
import csv
import scispacy
from negspacy.negation import Negex
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
import pandas as pd
import re
import time
import multiprocessing as mp
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

def string_contains_punctuation(sent):
    
    length = len(sent)
    punc =["'"]
    for i in range(length):
        if (sent[i] in punctuation) and (sent[i] not in punc):
            return sent[0:i], sent[i], sent[(i+1):length]
    
    return '', '', ''
    
def delete_if_exists(filename):

    try:
        os.remove(filename)
    except OSError:
        pass

def write_to_csv(_dict, output):
    
    delete_if_exists(output)
    with open(output, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in _dict.items():
            writer.writerow([key, value])

def write_to_csv_pos_neg_final(_dict_positive, _dict_negative, _dict_final, prefix, output):
    
    delete_if_exists(output)
    
    new_lex_concepts = ['PAT_ID', 'NOTE_ID']
    
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
        
            new_lex_concepts.append(new_pos)
            #new_lex_concepts.append(new_neutral)
            new_lex_concepts.append(new_neg)
      
        break
            
    with open(output, 'w', newline = '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(new_lex_concepts)
        #count = 0
        for key, value in _dict_positive.items():
            pat_id, note_id = key.split('_')
            note_id = note_id.replace('.txt', '')
            
            li_men = [pat_id, note_id]
            for key2, value2 in value.items():
                #if count == 0:
                #print('key: {} and key2: {}'.format(key, key2))
                li_men.extend([_dict_positive[key][key2], _dict_negative[key][key2]])
            writer.writerow(li_men)
            #count = count + 1

def init_dict(manager, _dict, notes_for_training, dict_gold_lex_cui):
    
    for file in notes_for_training:
        #name = file.replace('.source', '')
        name = file.strip()
        _dict[name] = manager.dict()
        for k, v in dict_gold_lex_cui.items():
            _dict[name][k] = 0
    
    #print(_dict)
    
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
    
def split(a, n):
    return [a[i::int(n)] for i in range(int(n))]
    
def load_gaz_lex(nlp, filename):
    
    _dict = defaultdict(list)
    
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            x_new = clean_text(row[0])
            words = [token.lemma_ for token in nlp(x_new.strip())]
            new_str = join_words(words)
            _dict[row[1].lower()].append(new_str)
            
    return _dict
    
def get_gaz_matches(nlp, matcher, texts):
    
    #for text in texts:
    for i in range(len(texts)):
        text = texts[i]
        doc = nlp(text.lower())
        for w in doc:
            _ = doc.vocab[w.text]
        matches = matcher(doc)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            
            yield (string_id, doc[start:end].text, text, start, end, i, doc[start:end])
                
def create_rule(nlp, words):
    
    POS = ['ADP', 'CCONJ']
    TAG = ['IN', 'CC', 'PRP'] #'RB'
    rule = []
    
    for word in words:  
        doc = nlp(word)
        for token in doc:
            token_rule = {}
            lemma = True
            if token.pos_ in POS:
                token_rule['POS'] = token.pos_
                lemma = False
            if token.tag_ in TAG:
                token_rule['TAG'] = token.tag_
                lemma = False
            if token.is_punct == True:
                token_rule['IS_PUNCT'] = True
                lemma = False
            if lemma:
                token_rule['LEMMA'] = token.lemma_
                
            rule.append(token_rule)
    
    #print(rule)
    return rule
    
def create_matcher(nlp, file_list):
        
    matcher = Matcher(nlp.vocab)

    POS = ['ADP', 'CCONJ', 'PART'] #'PART'
    TAG = ['IN', 'CC', 'PRP', 'TO'] #'RB'
    
    for filename in file_list:
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                col0 = row[0].lower()
                col1 = row[1].lower()
            
                str1, punc, str2 = string_contains_punctuation(col0)
                #print('{} {} {}'.format(str1, punc, str2))
                if (punc != ''):
                    words = [str1, punc, str2]
                    rule = create_rule(nlp, words)
                    matcher.add(col1, None, rule)
            
                else:
                    doc = nlp(col0)
                    #rule = defaultdict(dict)
                    rule = []
                    for token in doc:
                        token_rule = {}
                        lemma = True
                        if token.pos_ in POS:
                            token_rule['POS'] = token.pos_
                            lemma = False
                        if token.tag_ in TAG:
                            token_rule['TAG'] = token.tag_
                            lemma = False
                        if lemma:
                            token_rule['LEMMA'] = token.lemma_
                
                        rule.append(token_rule)
            
                    #print('rule: {}'.format(rule))
                    matcher.add(col1, None, rule)
    
    return matcher

def create_ruler(nlp, file_list):
    
    patterns = []
    POS = ['ADP', 'CCONJ', 'PART'] #'PART'
    TAG = ['IN', 'CC', 'PRP', 'TO'] #'RB'
    
    for filename in file_list:
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                col0 = row[0].lower()
                col1 = row[1].lower()
            
                str1, punc, str2 = string_contains_punctuation(col0)
                #print('{} {} {}'.format(str1, punc, str2))
                if (punc != ''):
                    words = [str1, punc, str2]
                    pattern = create_rule(nlp, words)
                    rule = {}
                    rule['label'] = col1
                    rule['pattern'] = pattern
                    patterns.append(rule)
            
                else:
                    doc = nlp(col0)
                    #rule = defaultdict(dict)
                    rule = {}
                    rule['label'] = col1
                    rule['pattern'] = []
                    for token in doc:
                        #print('token: {}'.format(token))
                        token_rule = {}
                        lemma = True
                        if token.pos_ in POS:
                            token_rule['POS'] = token.pos_
                            lemma = False
                        if token.tag_ in TAG:
                            token_rule['TAG'] = token.tag_
                            lemma = False
                        if lemma:
                            token_rule['LEMMA'] = token.lemma_
                
                        rule['pattern'].append(token_rule)
            
                    patterns.append(rule)
    
    #print(patterns)
    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns(patterns)
    
    return ruler

def break_into_sentences(nlp, text):
    doc = nlp(text)
    sent_list = [t.text for t in doc.sents]
    return sent_list
   
def core_process(nlp_lemma, nlp_neg, matcher, notes, doc_folder, 
                 dict_files_positive, dict_files_negative):
    
    for file in notes:
        with open(os.path.join(doc_folder, file), 'r') as f:
            sent_list = break_into_sentences(nlp_neg, f.read())
            #print(sent_list)
            for string_id, men, text, start, end, i, span in get_gaz_matches(nlp_neg, matcher, sent_list):
                
                
                print(span.start_char - span.sent.start_char, span.end_char - span.sent.start_char)
                words = [token.lemma_ for token in nlp_lemma(men.lower().strip())]
                sent_words = [token.lemma_ for token in nlp_lemma(text.lower().strip())]
                new_str = join_words(words)
                new_str_sent = join_words(sent_words)
                
                name = file.strip()
                content = name + ', [' + new_str_sent + '], ' + men + ', ' + string_id + ', (' + str(start) + ',' + str(end) + '), ' + str(i) + ', ' + '\n'
                print(content)
                
                # additional conditions to detect use case like: '... no fever. Patient has sore throat ...'
                split_strings = new_str_sent.split('.')
                for sub in split_strings:
                    threshold = 2
                    if (len(sub.split()) >= threshold):
                        neg = nlp_neg(sub)
                        for e in neg.ents:
                            #print(e.text)
                            #print(new_str)
                            #if (e.label_ == string_id):
                            if (new_str == e.text) and (e.label_ == string_id):
                                content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + ', ' + string_id +  ', (' + str(start) + ',' + str(end) + '), ' + str(i) +'\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, string_id)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, string_id)
                                
                                break
                                    
def mention_using_gaz(nlp_lemma, gaz_csv_list, notes_for_training, doc_folder, dict_gaz, prefix, output):
    
    manager = mp.Manager()
    dict_files_positive = manager.dict()
    dict_files_negative = manager.dict()
    
    init_dict(manager, dict_files_positive, notes_for_training, dict_gaz)
    init_dict(manager, dict_files_negative, notes_for_training, dict_gaz)
    
    dict_files_final = manager.dict()
    init_dict(manager, dict_files_final, notes_for_training, dict_gaz)
    
    #nlp_neg = scilg.load()
    nlp_neg = spacy.load('en_core_web_sm')
    ruler = create_ruler(nlp_neg, gaz_csv_list)
    nlp_neg.add_pipe(ruler)
    matcher = create_matcher(nlp_neg, gaz_csv_list)
    negex = Negex(nlp_neg, language = "en_clinical", chunk_prefix = ["without", "no"])
    negex.add_patterns(preceding_negations = ['deny', 'absent'])
    nlp_neg.add_pipe(negex, last = True)
    
    num_cores = int(mp.cpu_count())
    print('number of cores in the system: {}'.format(num_cores))
    #num_cores = 8
    #print('number of cores using: {}'.format(num_cores))
    min_files = 4
    cores_needed = 0
    ratio = len(notes_for_training) / num_cores
    if ratio >= min_files:
        cores_needed = num_cores
    else:
        cores_needed = (len(notes_for_training) + min_files) / min_files
    
    chunks = split(notes_for_training, cores_needed)
    #print(chunks)
    processes = []
    for i in range(len(chunks)):
        processes.append(mp.Process(target=core_process, args=(nlp_lemma, nlp_neg, matcher, chunks[i], doc_folder, 
                 dict_files_positive, dict_files_negative, )))
    for i in range(len(chunks)):
        processes[i].start()
    
    for i in range(len(chunks)):
        processes[i].join()
                       
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
    
    gaz_csv_list = [gaz_csv]
    notes_list = read_list_of_notes(notes_csv)
    
    tic = time.perf_counter()
    nlp_lemma = spacy.load('en_core_sci_sm')
    dict_gaz_lex = load_gaz_lex(nlp_lemma, gaz_csv)
    #print(dict_gaz_lex) 
    gaz_men = mention_using_gaz(nlp_lemma, gaz_csv_list, notes_list, doc_folder, dict_gaz_lex, prefix, output_ts)
    toc = time.perf_counter()
    print(f"Finished! Annotation done in {toc - tic:0.4f} seconds")
    
if __name__ == "__main__":
    main()

