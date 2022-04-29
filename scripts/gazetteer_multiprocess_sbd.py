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

def write_mention(_dict, file_path):
    
    with open(file_path, 'a') as file:
        w = csv.DictWriter(file, _dict.keys())

        if file.tell() == 0:
            w.writeheader()


        w.writerow(_dict)

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

def sentence_stats(sent_list):
    return [len(s) for s in sent_list]
  
def core_process(nlp_lemma, nlp_neg, matcher, notes, doc_folder, 
                 dict_files_positive, dict_files_negative, output):
    
    for file in notes:
        with open(os.path.join(doc_folder, file), 'r') as f:
            sent_list = break_into_sentences(nlp_neg, f.read())
            sent_compute = True

            for string_id, men, text, start, end, i, span in get_gaz_matches(nlp_neg, matcher, sent_list):
                
                # print offset
                #print(span.start_char - span.sent.start_char, span.end_char - span.sent.start_char)
              
                if sent_compute:
                    sent_length = sentence_stats(sent_list)
                    sent_compute = False
    
                words = [token.lemma_ for token in nlp_lemma(men.lower().strip())]
                sent_words = [token.lemma_ for token in nlp_lemma(text.lower().strip())]
                new_str = join_words(words)
                new_str_sent = join_words(sent_words)
                
                name = file.strip()
                #content = name + ', [' + new_str_sent + '], ' + men + ', ' + string_id + ', (' + str(start) + ',' + str(end) + '), ' + str(i) + ', ' + '\n'
                #print(content)
                
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
                                #content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + ', ' + string_id +  ', (' + str(start) + ',' + str(end) + '), ' + str(i) +'\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, string_id)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, string_id)
                                
                                # sentence-level mention
                                   38 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/5:0H
   39 root      rt   0       0      0      0 S   0.0  0.0   0:04.69 watchdog/6
   40 root      rt   0       0      0      0 S   0.0  0.0   0:03.06 migration/6
   41 root      20   0       0      0      0 S   0.0  0.0   0:02.51 ksoftirqd/6
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ ls -la
total 15524
drwxr-xr-x 6 gms  domain users     165 Apr 29 12:50 .
drwxr-xr-x 5 gms  root             222 Apr 28 13:11 ..
drwxr-xr-x 2 root root          495616 Apr 29 11:49 clamp_out
drwxr-xr-x 2 root root         6479872 Apr 29 09:28 clamp_out_1
-rw-r--r-- 1 root root           13911 Apr 29 11:49 clamp_out.zip
drwxr-xr-x 2 gms  domain users 3485696 Apr 29 12:47 data_in
drwxr-xr-x 2 gms  domain users  249856 Apr 28 17:42 data_new
-rwxr-xr-x 1 gms  domain users   17116 Apr 29 12:49 final_lex.csv
-rw-r--r-- 1 gms  domain users 2044981 Apr 29 12:49 gs_manifest.csv
-rw-r--r-- 1 root root              15 Apr 29 09:28 nlptab_manifest.txt
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ top
top - 14:29:27 up 19 days, 12:18,  1 user,  load average: 5.45, 4.04, 2.86
Tasks: 442 total,  26 running, 416 sleeping,   0 stopped,   0 zombie
                 k8s_POD_calico-apiserver-54b98ccf5b-mwx97_calico-apiserver_055f84a9-9b68-422c-812e-22f15d64c96f_0
d323f161ed90        k8s.gcr.io/pause:3.5                                                                                        "/pause"                 2 weeks ago         Up 2 weeks                                      k8s_POD_minio-79566d86cb-k5hg5_argo_3014446b-a9c9-4ec7-a7b3-2c8b0509aee5_0
4172d2b7f5c8        k8s.gcr.io/pause:3.5                                                                                        "/pause"                 2 weeks ago         Up 2 weeks                                      k8s_POD_argo-server-5d58f6585d-8tzhl_argo_ba1126e8-a7ae-47d8-a3eb-eac6c0894662_0
01b28f208772        f1bca4d4ced2                                                                                                "start_runit"            2 weeks ago         Up 2 weeks                                      k8s_calico-node_calico-node-jwmjl_calico-system_091b5f07-0f20-4e85-ac85-fc8513a316e0_7
4bbd3dcddf67        9507cf15077f                                                                                                "/sbin/tini -- cal..."   2 weeks ago         Up 2 weeks                                      k8s_calico-typha_calico-typha-7b4578b8b9-r6h5w_calico-system_7360e344-abfc-4852-89e5-b6f35cd693b7_7
aabcd4d0b8ee        k8s.gcr.io/pause:3.5                                                                                        "/pause"                 2 weeks ago         Up 2 weeks                                      k8s_POD_calico-typha-7b4578b8b9-r6h5w_calico-system_7360e344-abfc-4852-89e5-b6f35cd693b7_6
5557b2879516        edeff87e4802                                                                                                "/usr/local/bin/ku..."   2 weeks ago         Up 2 weeks                                      k8s_kube-proxy_kube-proxy-xz58l_kube-system_c65230e8-973e-4580-957a-93ee3667f4e8_7
f35bf0a34d3e        k8s.gcr.io/pause:3.5                                                                                        "/pause"                 2 weeks ago         Up 2 weeks                                      k8s_POD_kube-proxy-xz58l_kube-system_c65230e8-973e-4580-957a-93ee3667f4e8_8
3a8867e7e8fc        k8s.gcr.io/pause:3.5                                                                                        "/pause"                 2 weeks ago         Up 2 weeks                                      k8s_POD_calico-node-jwmjl_calico-system_091b5f07-0f20-4e85-ac85-fc8513a316e0_7
[gms@thalia3 gold_standard]$ rsync -avxS /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/notes/gold_standard/data_new/ ./data_in^C
[gms@thalia3 gold_standard]$ docker commit 00ef2290f18c ahc-nlpie-docker.artifactory.umn.edu/gazetteer:3
sha256:caeed3d9b6ae607a8c1343eb4c421e4c6d40b517e8562a4d127cd54ca2ee964d
[gms@thalia3 gold_standard]$ docker images
REPOSITORY                                       TAG                 IMAGE ID            CREATED             SIZE
ahc-nlpie-docker.artifactory.umn.edu/gazetteer   3                   caeed3d9b6ae        9 seconds ago       1.44 GB
ahc-nlpie-docker.artifactory.umn.edu/gazetteer   2                   9d42b1a0fcd2        17 minutes ago      1.44 GB
quay.io/argoproj/argocli                         latest              3eb54ccaa674        3 hours ago         100 MB
quay.io/argoproj/argocli                         <none>              63e3c753cf76        42 hours ago        100 MB
quay.io/argoproj/argocli                         <none>              87359d07ec5a        45 hours ago        100 MB
quay.io/argoproj/argocli                         <none>              d5968c43a5f8        46 hours ago        100 MB
quay.io/argoproj/argocli                         <none>              a62b8d05d34d        2 days ago          100 MB
quay.io/argoproj/argocli                         <none>              d5261c07fc77        2 days ago          100 MB
quay.io/argoproj/argocli                         <none>              62e72b3e5052        2 days ago          100 MB
quay.io/argoproj/argocli                         <none>              9e06e848d712        3 days ago          100 MB
quay.io/argoproj/argocli                         <none>              b8f5a84de45a        3 days ago          100 MB
ahc-nlpie-docker.artifactory.umn.edu/clmp        11                  23c386561cce        4 days ago          2.75 GB
quay.io/argoproj/argocli                         <none>              aeaa1b0ba73a        4 days ago          100 MB
quay.io/argoproj/argocli                         <none>              7267af087449        6 days ago          100 MB
ahc-nlpie-docker.artifactory.umn.edu/clmp        10                  7664267c4168        9 days ago          2.75 GB
docker.io/minio/minio                            latest              0a812feab730        2 weeks ago         227 MB
ahc-nlpie-docker.artifactory.umn.edu/gazetteer   1                   a811a4fcf2e9        7 weeks ago         1.44 GB
quay.io/tigera/operator                          v1.23.3             0f4db68d6a12        4 months ago        183 MB
docker.io/calico/apiserver                       v3.21.2             71d891eca6f4        4 months ago        193 MB
docker.io/calico/node                            v3.21.2             f1bca4d4ced2        4 months ago        214 MB
docker.io/calico/typha                           v3.21.2             9507cf15077f        4 months ago        128 MB
docker.io/calico/kube-controllers                v3.21.2             b20652406028        4 months ago        132 MB
k8s.gcr.io/kube-proxy                            v1.22.4             edeff87e4802        5 months ago        104 MB
k8s.gcr.io/pause                                 3.5                 ed210e3e4a5b        13 months ago       683 kB
[gms@thalia3 gold_standard]$ docker run -it  -v /var/lib/docker/data/vte/gold_standard/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:3  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py fin
al_lex.csv gs_manifest.csv data_in/ gold_standard vte
number of cores in the system: 24

[gold] 0:gms@thalia3:/var/lib/docker/data/vte/gold_standard*                                                                                                           "thalia3.ahc.umn.edu" 14:29 29-Apr-22
%Cpu(s): 80.2 us,  9.7 sy,  0.0 ni,  7.4 id,  2.7 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 13184876+total, 74259912 free, 52371532 used,  5217316 buff/cache
KiB Swap:        0 total,        0 free,        0 used. 75147240 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
 3334 rmcewan   20   0   74.1g  43.0g  16192 S 301.0 34.2   3341:02 java
 3337 rmcewan   20   0   36.6g   1.4g  16020 S   0.0  1.1   9:06.50 java
 2962 rmcewan   20   0   11.7g   1.2g  15096 S   1.3  0.9  36:09.99 java
 1636 root      20   0 2881448   1.0g   2392 S   1.0  0.8  15:06.44 python
 9892 root      20   0 1633820 599072   4220 R  90.1  0.5   0:04.31 python
 9845 root      20   0 1632772 598148   4408 R  84.2  0.5   0:04.34 python
 9825 root      20   0 1631108 597520   4216 R  73.9  0.5   0:04.35 python
 1297 root      20   0 1606896 587120  20292 S   0.0  0.4   8:03.32 python
 9865 root      20   0 1621984 586528   4220 R  72.6  0.4   0:04.16 python
 9873 root      20   0 1621196 585704   4220 R  70.0  0.4   0:04.18 python
 9876 root      20   0 1618636 583144   4276 R  78.5  0.4   0:04.34 python
 9849 root      20   0 1617724 582024   4216 R  69.0  0.4   0:04.25 python
 9841 root      20   0 1616468 580672   4216 R  67.0  0.4   0:04.07 python
 9833 root      20   0 1615596 579928   4344 R  73.3  0.4   0:04.24 python
 9837 root      20   0 1615096 579452   4344 R  71.9  0.4   0:04.12 python
 9861 root      20   0 1613668 577752   4212 R  75.2  0.4   0:04.17 python
 9808 root      20   0 1613388 577460   4212 R  68.0  0.4   0:03.97 python
 9887 root      20   0 1611768 575904   4216 R  85.5  0.4   0:04.15 python
 9813 root      20   0 1611664 575796   4212 R  71.0  0.4   0:04.25 python
 9817 root      20   0 1611140 575376   4340 R  70.3  0.4   0:04.13 python
 9853 root      20   0 1610216 575324   4408 R  69.6  0.4   0:04.23 python
 9868 root      20   0 1611120 575232   4212 R  73.3  0.4   0:04.13 python
 9889 root      20   0 1610924 575004   4212 R  82.5  0.4   0:04.12 python
 9882 root      20   0 1610540 574808   4348 R  76.2  0.4   0:03.92 python
 9879 root      20   0 1610232 574584   4408 R  85.5  0.4   0:04.48 python
 9857 root      20   0 1610136 574444   4408 R  71.3  0.4   0:04.02 python
 9884 root      20   0 1607152 572316   4216 R  87.5  0.4   0:04.21 python
 9821 root      20   0 1607864 572124   4348 R  63.7  0.4   0:04.01 python
 9829 root      20   0 1607968 572028   4212 R  70.6  0.4   0:04.24 python
 3038 root      20   0  497532 327088  18548 S   0.3  0.2  81:30.47 splunkd
 1535 root      20   0 2775952 117424  41020 S   5.6  0.1 802:54.67 kubelet
15813 root      20   0  871640 105320  20104 S   0.0  0.1  25:24.15 minio
 2058 root      20   0 4071296  95720  17540 S   3.6  0.1 518:12.49 dockerd-current
23307 root      20   0 2911748  61784  23796 S   1.3  0.0 134:57.81 apiserver
 1851 splunk    20   0 2829876  50920  19960 S   1.0  0.0 208:25.38 operator
12436 root      20   0 3003436  48728  22612 S   3.0  0.0 529:45.90 calico-node
12009 polkitd   20   0 2824204  47932  17048 S   0.0  0.0  49:59.97 calico-typha
 5640 polkitd   20   0 2457700  40288  16620 S   0.0  0.0   4:05.80 kube-controller
12437 root      20   0 2560276  39964  17992 S   0.0  0.0   3:52.07 calico-node
 1354 root      20   0  299984  38628  37272 S   0.0  0.0   0:46.46 sssd_nss
  724 root      20   0   81100  38368  37836 S   0.0  0.0  22:05.00 systemd-journal
 1578 root      20   0  958708  37996   9080 S   0.0  0.0  90:27.71 python
12439 root      20   0 2781728  37592  15760 S   0.0  0.0   3:07.73 calico-node
 4441 root      20   0 2412556  32256  18488 S   0.0  0.0   0:55.14 calico-node
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ ls -la
total 15532
drwxr-xr-x 6 gms  domain users     189 Apr 29 14:29 .
drwxr-xr-x 5 gms  root             222 Apr 28 13:11 ..
drwxr-xr-x 2 root root          495616 Apr 29 11:49 clamp_out
drwxr-xr-x 2 root root         6479872 Apr 29 09:28 clamp_out_1
-rw-r--r-- 1 root root           13911 Apr 29 11:49 clamp_out.zip
drwxr-xr-x 2 gms  domain users 3485696 Apr 29 12:47 data_in
drwxr-xr-x 2 gms  domain users  249856 Apr 28 17:42 data_new
-rwxr-xr-x 1 gms  domain users   17116 Apr 29 12:49 final_lex.csv
-rw-r--r-- 1 gms  domain users 2044981 Apr 29 12:49 gs_manifest.csv
-rw-r--r-- 1 root root            7130 Apr 29 14:29 mention_standard
-rw-r--r-- 1 root root              15 Apr 29 09:28 nlptab_manifest.txt
[gms@thalia3 gold_standard]$ cat mention_standard
file,sentence,polarity,men,concept,start,end,span.sent.start_char,span.sent.end_char,sentence_n,sent_lengths
100021480_458031144.txt,post-operatively pt be start on mechanical for dvt prophylaxis .,True,dvt,vte,51,54,0,67,6,"[49, 15, 71, 15, 41, 88, 67, 42, 63, 75, 94, 50, 14, 275, 25, 94, 113, 11, 22, 166, 166, 22, 70, 87, 35, 65, 53, 48, 66, 82, 84, 106]"
100021038_460039863.txt,"[ urine:1495 ; drains:425 ; blood:100 ] pe : gen : nad , lay in bed mildly uncomfortable 2/2 pain .",True,pe,vte,36,38,0,94,17,"[117, 2, 115, 85, 127, 7, 55, 6, 27, 18, 31, 10, 17, 2, 26, 36, 9, 94, 6, 57, 63, 37, 3, 10, 12, 1, 15, 17, 12, 23, 22, 15, 25, 199, 111, 13, 57, 38, 32, 30, 21, 7, 13, 31, 10, 13, 39, 21, 5, 9, 13, 17, 4, 14, 29, 73, 20, 37, 14, 34, 180, 53, 61, 19, 63, 12]"
100021038_460342059.txt,"weight ( kg ) 87.95 87.95 87.95 87.95 pe-aox3 wound-incision heal appropriately , jps in place with serosanguinous output a/p : 61 year old female s/p bilateral mastectomy with immediate te reconstruction -continue abx while drain in place -will start rom exercise in 2 week",True,pe,vte,36,38,0,242,27,"[45, 34, 44, 30, 9, 33, 6, 9, 20, 4, 25, 26, 9, 87, 38, 6, 10, 35, 51, 15, 22, 32, 25, 17, 42, 10, 10, 279, 105, 11]"
100047706_453010077.txt,prophylaxis dvt : scds while in bed gi :,True,dvt,vte,12,15,0,16,21,"[72, 35, 6, 166, 4, 73, 61, 43, 3, 80, 10, 90, 6, 3, 69, 5, 92, 5, 9, 5, 28, 38, 15, 124, 105, 68, 10, 9, 20, 12, 16, 24, 108, 19, 15, 109, 3, 35, 27, 23, 46, 35, 44, 33, 17, 27, 26, 13, 8, 13, 24, 9, 3, 30, 37, 72, 31, 147, 112, 21]"
100047706_451440970.txt,prophylaxis dvt : scds while in bed gi :,True,dvt,vte,12,15,0,16,23,"[107, 6, 9, 168, 4, 73, 61, 43, 3, 50, 25, 15, 10, 90, 6, 3, 69, 5, 92, 5, 9, 5, 28, 38, 15, 124, 60, 45, 68, 10, 9, 20, 12, 16, 24, 108, 19, 15, 109, 35, 6, 35, 27, 60, 9, 95, 18, 17, 27, 26, 35, 24, 9, 3, 30, 255, 21]"
100047706_451441381.txt,prophylaxis dvt : scds while in bed gi :,True,dvt,vte,12,15,0,16,20,"[143, 6, 126, 4, 74, 61, 43, 3, 30, 10, 90, 6, 3, 69, 5, 71, 5, 9, 5, 28, 38, 15, 121, 111, 68, 10, 9, 20, 12, 16, 51, 15, 16, 32, 47, 2, 14, 140, 87, 70, 95, 18, 17, 54, 36, 17, 6, 9, 3, 30, 50, 204, 21]"
100047706_453009855.txt,prophylaxis dvt : scds while in bed gi :,True,dvt,vte,12,15,0,16,21,"[72, 35, 6, 154, 4, 73, 61, 43, 3, 80, 10, 82, 6, 3, 69, 5, 92, 5, 9, 5, 28, 38, 15, 124, 105, 68, 10, 9, 20, 12, 16, 28, 104, 19, 15, 109, 3, 63, 22, 36, 114, 179, 115, 59, 112, 21]"
100021480_457396347.txt,# deep venous thrombosis prophylaxis as per primary team .,True,deep venous thrombosis,vte,2,24,0,57,86,"[52, 70, 91, 19, 37, 13, 16, 75, 17, 11, 46, 13, 6, 21, 51, 6, 10, 52, 20, 10, 8, 11, 10, 8, 22, 3, 8, 10, 16, 46, 6, 37, 4, 41, 50, 47, 27, 162, 38, 51, 5, 7, 22, 7, 23, 19, 30, 15, 10, 6, 26, 23, 19, 80, 126, 6, 21, 13, 13, 45, 100, 68, 11, 25, 41, 41, 48, 17, 32, 13, 15, 35, 13, 13, 13, 55, 31, 17, 56, 52, 15, 30, 64, 17, 15, 25, 57, 10, 25, 35, 58, 48, 30, 31, 30]"
100021480_456829217.txt,# deep venous thrombosis prophylaxis as per primary team .,True,deep venous thrombosis,vte,2,24,0,57,90,"[52, 70, 91, 19, 26, 13, 47, 76, 71, 26, 11, 48, 16, 11, 6, 41, 6, 10, 53, 20, 10, 8, 11, 10, 8, 22, 3, 8, 10, 16, 21, 15, 16, 32, 17, 11, 20, 41, 4, 50, 84, 97, 47, 27, 181, 32, 8, 35, 4, 30, 38, 23, 5, 7, 7, 23, 65, 30, 15, 10, 6, 26, 23, 19, 80, 126, 6, 21, 129, 11, 25, 41, 38, 32, 41, 57, 22, 15, 32, 13, 66, 31, 17, 56, 52, 15, 30, 106, 15, 25, 57, 10, 58, 48, 30, 31, 30]"
100088357_454102563.txt,"-will start lovenox dvt prophylaxis today -monitor drain output , d/c once put -PRON- less than 30cc in 24hr period",True,dvt,vte,20,23,0,115,32,"[40, 7, 26, 58, 9, 33, 6, 9, 22, 4, 8, 14, 74, 6, 39, 90, 10, 58, 15, 16, 32, 17, 11, 14, 2, 7, 13, 22, 4, 64, 22, 212, 115, 110]"
100086857_460022076.txt,"start lovenox dvt ppx today , continue",True,dvt,vte,14,17,0,37,30,"[41, 7, 26, 68, 27, 17, 29, 2, 9, 44, 6, 21, 13, 15, 45, 13, 70, 83, 35, 10, 91, 32, 29, 2, 11, 2, 73, 28, 141, 160, 37, 53, 89]"
100087163_455374890.txt,0 comment : patient should start asa 2 week after surgery and cont to 2 week for dvt ppx .,True,dvt,vte,83,86,0,91,24,"[60, 26, 76, 20, 30, 62, 18, 177, 2, 56, 33, 49, 73, 74, 25, 39, 22, 95, 69, 48, 1, 21, 93, 19, 91, 64, 11, 11, 55, 33, 33, 59, 3, 51, 21, 10, 20, 108, 84, 2, 41, 27, 32, 51, 49, 73, 89, 9, 34, 37, 16, 105, 42, 47, 205, 77, 79, 15, 59, 15, 60, 90, 95, 24, 117, 96, 81, 63, 108, 78, 105, 103, 46, 126, 81, 64, 84, 112, 39, 51, 70, 122, 118, 107, 30, 100, 3, 21, 27]"
100087163_455374890.txt,for dvt ppx associate diagnosis :,True,dvt,vte,4,7,0,33,30,"[60, 26, 76, 20, 30, 62, 18, 177, 2, 56, 33, 49, 73, 74, 25, 39, 22, 95, 69, 48, 1, 21, 93, 19, 91, 64, 11, 11, 55, 33, 33, 59, 3, 51, 21, 10, 20, 108, 84, 2, 41, 27, 32, 51, 49, 73, 89, 9, 34, 37, 16, 105, 42, 47, 205, 77, 79, 15, 59, 15, 60, 90, 95, 24, 117, 96, 81, 63, 108, 78, 105, 103, 46, 126, 81, 64, 84, 112, 39, 51, 70, 122, 118, 107, 30, 100, 3, 21, 27]"
100076200_477132177.txt,"no calf , thigh tenderness to suggest dvt .",False,dvt,vte,37,40,0,41,32,"[84, 13, 33, 25, 12, 12, 39, 22, 7, 30, 14, 13, 4, 12, 45, 87, 5, 8, 23, 44, 2, 13, 42, 12, 12, 6, 5, 26, 29, 48, 11, 22, 41, 6, 73, 59, 17, 14, 36, 17, 14, 3, 50, 3, 30, 29, 14, 59, 13, 11, 3, 2, 23, 39, 15, 33, 13, 27, 17, 30, 20, 3, 49, 16, 13, 48, 23, 58, 26, 22, 2, 63, 9, 18, 2, 88, 2, 29, 45, 2, 57, 7, 21, 2, 58, 24, 9, 2, 33, 24, 2, 22, 2, 19, 3, 24, 3, 18, 3, 27, 3, 16, 3, 30, 3, 34, 3, 14, 8, 35, 5, 49, 17, 55, 31, 34, 36, 63, 87, 27]"
100095410_493266194.txt,continue on p.r.n . pain medicine and deep venous thrombosis prophylaxis per dr . heller 's protocol .,True,deep venous thrombosis,vte,38,60,0,99,30,"[64, 46, 11, 15, 6, 9, 20, 12, 16, 45, 4, 100, 5, 18, 5, 20, 91, 88, 20, 31, 5, 17, 13, 4, 14, 1, 33, 7, 9, 97, 99, 2, 64, 37, 91, 13, 112, 15, 14, 15, 2, 15, 24, 2, 25, 42, 2, 47, 33, 30, 81, 30]"
100117504_460500849.txt,70 m s/p l tka on 2/10/14 -pain control -nausea control -advance diet as tolerate -bowel medication -antibiotics complete -drain out -lovenox for dvt prophylaxis,True,dvt,vte,148,151,0,163,21,"[29, 30, 16, 47, 9, 33, 6, 9, 21, 13, 16, 74, 30, 10, 4, 3, 19, 16, 49, 18, 10, 163, 8, 10, 5, 71]"
100076200_475977938.txt,"no calf , thigh tenderness to suggest dvt .",False,dvt,vte,37,40,0,41,33,"[84, 13, 45, 50, 17, 60, 13, 12, 22, 28, 10, 11, 16, 15, 14, 10, 12, 45, 88, 5, 8, 23, 45, 20, 12, 19, 6, 5, 26, 29, 48, 11, 22, 41, 6, 73, 59, 24, 38, 17, 50, 17, 14, 3, 115, 47, 15, 16, 13, 15, 2, 23, 3, 18, 3, 15, 2, 15, 33, 15, 7, 21, 17, 30, 20, 3, 72, 38, 24, 5, 48, 23, 58, 26, 47, 19, 11, 2, 63, 2, 34, 9, 2, 29, 45, 2, 57, 7, 2, 58, 24, 9, 2, 33, 24, 2, 22, 2, 57, 3, 24, 3, 18, 3, 27, 3, 16, 3, 30, 3, 34, 19, 2, 70, 11, 19, 53, 11, 87, 27]"
100095410_493135800.txt,continue on p.r.n . pain medicine and deep venous thrombosis prophylaxis per dr . heller 's protocol .,True,deep venous thrombosis,vte,38,60,0,99,31,"[84, 85, 10, 11, 15, 6, 9, 20, 12, 16, 45, 4, 100, 5, 18, 5, 21, 93, 88, 20, 31, 5, 17, 13, 4, 14, 1, 33, 7, 9, 118, 99, 1, 101, 1, 7, 4, 35, 64, 37, 91, 13, 112, 15, 14, 15, 2, 15, 24, 2, 25, 42, 2, 47, 33, 67, 30]"
100117504_462303752.txt,start on lovenox on pod 1 for dvt prophylaxis .,True,dvt,vte,32,35,0,48,12,"[12, 118, 32, 31, 36, 70, 29, 136, 34, 16, 16, 44, 48, 71, 49, 104, 22, 9, 98, 95, 25, 140, 34, 90, 144, 2, 81, 54, 11, 11, 109, 1, 158, 61, 82, 29, 49, 124, 2, 44, 11, 1, 21, 31, 51, 23, 51, 54, 126, 61, 87, 40, 49, 54, 39, 63, 78, 87, 15, 79, 15, 123, 95, 24, 44, 72, 3, 85, 39, 25, 34, 19, 54, 89, 27, 38]"
100117504_461177693.txt,70 m s/p l tka on 2/10/14 -pain control -nausea control -advance diet as tolerate -bowel medication -antibiotics complete -drain out -lovenox for dvt prophylaxis,True,dvt,vte,148,151,0,163,21,"[29, 26, 21, 9, 31, 6, 9, 21, 13, 16, 65, 10, 4, 3, 19, 16, 44, 3, 18, 7, 4, 163, 8, 10, 5, 8, 2, 106, 62]"
100117504_462303752.txt,dvt prophylaxis : lovenox discharge medication list as of 2/13/2014 2:48 pm start take these medication detail order for dme cpm p&j home,True,dvt,vte,0,3,0,140,21,"[12, 118, 32, 31, 36, 70, 29, 136, 34, 16, 16, 44, 48, 71, 49, 104, 22, 9, 98, 95, 25, 140, 34, 90, 144, 2, 81, 54, 11, 11, 109, 1, 158, 61, 82, 29, 49, 124, 2, 44, 11, 1, 21, 31, 51, 23, 51, 54, 126, 61, 87, 40, 49, 54, 39, 63, 78, 87, 15, 79, 15, 123, 95, 24, 44, 72, 3, 85, 39, 25, 34, 19, 54, 89, 27, 38]"
100284288_457767449.txt,no diabetes heme : stable dispo : sicu dvt prophylaxis :,False,dvt,vte,37,40,0,53,15,"[37, 19, 257, 51, 89, 38, 119, 23, 59, 31, 46, 32, 72, 169, 19, 53, 46, 20, 12, 42, 161, 19, 2, 12, 25, 6, 9, 21, 41, 6, 10, 35, 9, 9, 2, 26, 36, 9, 112, 14, 6, 48, 82, 157, 73, 96, 6, 4, 4, 9, 4, 6, 26, 72, 37, 6, 9, 5, 4, 3, 83, 49, 24, 1, 29, 13, 20, 51, 85, 60, 62, 59, 4, 91, 32, 34, 70, 11, 5, 11, 24, 11, 30, 7, 31, 28, 11, 25, 154, 11, 55, 96, 75, 55, 15, 29, 14, 4, 8, 20, 85, 33, 26, 104, 25, 99, 16, 26, 68, 35, 64, 11, 27, 172, 25]"
100095410_492747334.txt,continue on p.r.n . pain medicine and deep venous thrombosis prophylaxis per dr . heller 's protocol .,True,deep venous thrombosis,vte,38,60,0,99,29,"[39, 45, 54, 10, 11, 15, 16, 20, 12, 16, 45, 4, 100, 5, 18, 5, 21, 61, 86, 74, 40, 23, 9, 35, 7, 9, 55, 2, 96, 99, 2, 64, 37, 67, 30, 13, 71, 47, 15, 14, 15, 2, 15, 24, 2, 25, 42, 2, 54, 33, 67, 30]"
100076200_478691913.txt,-PRON- be start on appropriate mechanical and pharmacologic dvt prophylaxis postoperatively .,True,dvt,vte,60,63,0,92,11,"[26, 24, 11, 83, 16, 102, 42, 108, 52, 78, 55, 92, 66, 76, 145, 92, 179, 19, 48, 42, 38, 8, 50, 22, 16, 10, 40, 20, 12, 17]"
100429501_464453092.txt,dvt prophylaxis with asa be well tolerate .,True,dvt,vte,0,3,0,44,18,"[57, 3, 57, 29, 66, 3, 56, 9, 78, 10, 55, 150, 12, 10, 129, 35, 40, 63, 44, 64, 112, 22, 65, 69, 2, 75, 2, 23, 20, 7, 3, 44, 47, 8, 51, 113, 68, 100, 28, 83, 15, 86, 29, 47, 59, 92, 59, 3, 84, 146, 219, 55, 15, 40, 147, 95, 30, 100, 61]"
100125375_449764391.txt,dvt : heparin code status :,True,dvt,vte,0,3,0,25,29,"[90, 139, 2, 62, 51, 51, 5, 33, 48, 54, 2, 13, 82, 83, 2, 93, 45, 50, 203, 149, 84, 2, 25, 109, 30, 22, 58, 31, 31, 25, 78, 26, 9, 17, 26, 35, 57, 16, 22, 6, 9, 19, 21, 7, 93, 53, 91, 8, 14, 37, 24, 26, 25, 11, 14, 34, 23, 7, 19, 125, 2, 88, 137, 2, 65, 2, 76, 21, 13, 43, 11, 22, 5, 8, 4, 9, 6, 14, 4, 15, 3, 99, 43, 13, 13, 25, 4, 4, 9, 1, 8, 10, 9, 23, 4, 4, 11, 4, 4, 10, 4, 4, 16]"
100180552_452315796.txt,patient be start on a heparin gtt due to sob pe suspect objective :,True,pe,vte,48,50,0,71,15,"[45, 13, 11, 64, 83, 88, 54, 15, 42, 27, 82, 69, 58, 50, 16, 71, 34, 41, 9, 25, 39, 54, 41, 6, 44, 150, 199, 68, 3, 33, 34, 28, 64, 89, 26, 192, 47, 19]"
100180552_452315796.txt,patient will be transfer to 6232-01 6b due to sob currently on bipap for possible pe .,True,pe,vte,86,88,0,89,33,"[45, 13, 11, 64, 83, 88, 54, 15, 42, 27, 82, 69, 58, 50, 16, 71, 34, 41, 9, 25, 39, 54, 41, 6, 44, 150, 199, 68, 3, 33, 34, 28, 64, 89, 26, 192, 47, 19]"
100410234_452072151.txt,"base on age , extent of thrombus ( just below l renal to femoral ) , i would recommend an aorto-bifem , likely b/l femoral embolectomy .",True,embolectomy,vte,120,133,0,134,42,"[25, 15, 9, 2, 9, 22, 16, 12, 7, 29, 28, 50, 20, 20, 22, 24, 45, 10, 13, 54, 21, 18, 46, 56, 44, 52, 41, 15, 5, 21, 16, 33, 19, 16, 37, 23, 72, 49, 58, 19, 17, 75, 134, 118, 59, 55, 76, 159, 18]"
100117504_460052272.txt,-lovenox start today for dvt prophylaxis,True,dvt,vte,28,31,0,43,33,"[29, 19, 28, 27, 9, 44, 9, 21, 13, 16, 74, 30, 10, 4, 3, 19, 16, 49, 15, 3, 5, 54, 22, 31, 16, 3, 38, 9, 1, 3, 75, 65, 14, 43, 8, 10, 5, 71]"
100410234_455524524.txt,ms . oppegard be a 50-year-old woman with h/o of factor v leiden who present with infrarenal occlusive aortic thrombus with collateral to ble intact now pod # 3 s/p aortobifemoral bypass and right femoral artery embolectomy .,True,embolectomy,vte,213,224,0,225,22,"[40, 55, 11, 35, 33, 43, 14, 12, 9, 22, 12, 25, 36, 26, 39, 88, 73, 36, 6, 15, 8, 16, 225, 34, 6, 20, 129, 180, 77, 7, 9, 39, 18, 54, 29, 23, 13, 13]"
100446488_475362737.txt,general care gi prophylaxis : none as eat dvt prophylaxis,True,dvt,vte,45,48,0,60,24,"[43, 10, 60, 112, 51, 55, 36, 15, 106, 43, 59, 27, 52, 40, 23, 92, 16, 13, 80, 45, 82, 40, 88, 98, 60, 21, 35, 32, 6, 38, 16, 13, 102, 69, 30, 43, 72, 38, 191, 9, 22, 10, 9, 12, 7, 31, 81, 32, 32, 29, 2, 11, 2, 56, 35, 35, 18, 8, 25, 17, 16, 2, 35, 7, 9, 12, 14, 10, 138, 18, 75, 13, 33, 11, 167, 146, 2, 12, 6, 14, 55, 114, 5, 95, 16, 19, 21, 13, 13, 13, 94, 5, 15, 5, 5, 5, 52, 1, 2, 21, 1, 2, 195, 13, 13, 13, 8, 38, 11, 5, 82, 18, 41, 38, 13, 13, 39, 5, 5, 15, 21, 12, 23, 64, 276, 12]"
100127250_450025097.txt,medical history diagnosis date prostate cancer urethral stricture urinary incontinence appendicitis kidney stone no surgery cva ( cerebral infarction ),False,infarction,vte,138,148,113,149,8,"[44, 22, 85, 20, 46, 41, 54, 18, 149, 4, 177, 149, 4, 222, 114, 57, 54, 102, 39, 19, 41, 49, 2, 35, 21, 117, 56, 32, 3, 74, 29, 1, 48, 22, 106, 33, 57, 35, 21, 53, 20, 10, 21, 56, 39, 19, 41, 49, 2, 38, 21, 53, 20, 12, 51, 45, 56, 15, 10, 21, 53, 3, 74, 4, 43, 50, 1, 53, 38, 29, 14, 182, 49, 70, 55, 64, 91, 50, 100, 43, 176, 129, 83, 1, 61, 99, 82, 72, 6, 79, 87, 59, 99, 55, 1, 198, 37, 175, 74, 91, 155, 335, 16, 108, 148, 157, 37]"
100127250_450025097.txt,"recurrent uti meniere 's disease hyperlipidemia unspecified cerebral artery occlusion with cerebral infarction discharge diagnosis : gross hematuria , bladder neck contracture past",True,infarction,vte,99,109,0,177,10,"[44, 22, 85, 20, 46, 41, 54, 18, 149, 4, 177, 149, 4, 222, 114, 57, 54, 102, 39, 19, 41, 49, 2, 35, 21, 117, 56, 32, 3, 74, 29, 1, 48, 22, 106, 33, 57, 35, 21, 53, 20, 10, 21, 56, 39, 19, 41, 49, 2, 38, 21, 53, 20, 12, 51, 45, 56, 15, 10, 21, 53, 3, 74, 4, 43, 50, 1, 53, 38, 29, 14, 182, 49, 70, 55, 64, 91, 50, 100, 43, 176, 129, 83, 1, 61, 99, 82, 72, 6, 79, 87, 59, 99, 55, 1, 198, 37, 175, 74, 91, 155, 335, 16, 108, 148, 157, 37]"
100146746_468084116.txt,prophylaxis dvt : scds while in bed gi : protonix 10 .,True,dvt,vte,12,15,0,16,22,"[90, 15, 6, 210, 4, 74, 61, 43, 3, 82, 10, 63, 81, 6, 3, 139, 5, 71, 5, 9, 5, 28, 51, 6, 10, 17, 151, 64, 1, 66, 68, 10, 9, 17, 12, 16, 50, 15, 16, 32, 32, 14, 2, 84, 12, 21, 156, 35, 27, 35, 31, 78, 68, 72, 31]"
100127250_450025097.txt,medical history diagnosis date prostate cancer urethral stricture urinary incontinence appendicitis kidney stone no surgery cva ( cerebral infarction ),False,infarction,vte,138,148,113,149,11,"[44, 22, 85, 20, 46, 41, 54, 18, 149, 4, 177, 149, 4, 222, 114, 57, 54, 102, 39, 19, 41, 49, 2, 35, 21, 117, 56, 32, 3, 74, 29, 1, 48, 22, 106, 33, 57, 35, 21, 53, 20, 10, 21, 56, 39, 19, 41, 49, 2, 38, 21, 53, 20, 12, 51, 45, 56, 15, 10, 21, 53, 3, 74, 4, 43, 50, 1, 53, 38, 29, 14, 182, 49, 70, 55, 64, 91, 50, 100, 43, 176, 129, 83, 1, 61, 99, 82, 72, 6, 79, 87, 59, 99, 55, 1, 198, 37, 175, 74, 91, 155, 335, 16, 108, 148, 157, 37]"
100438307_454357376.txt,dvt prophylaxis 40 mg lovenox daily 4 .,True,dvt,vte,0,3,0,37,16,"[11, 20, 22, 9, 33, 9, 20, 4, 8, 15, 168, 41, 101, 2, 37, 21, 37, 18, 20]"
100611415_463774655.txt,post-operatively pt be start on mechanical for dvt prophylaxis .,True,dvt,vte,51,54,0,67,9,"[40, 36, 9, 55, 21, 13, 15, 74, 104, 67, 44, 70, 74, 77, 64, 75, 14, 43, 31, 15, 52, 31, 69, 110, 35, 80, 48, 146, 23, 176, 22, 255, 53, 78, 47, 14, 12, 64, 28, 45, 48, 10, 138, 16, 72, 33, 53, 24, 37, 100, 4, 15, 53, 5, 18, 43, 80, 33, 15, 76, 93, 32, 50, 40, 46, 43, 7, 12, 64, 43, 7, 12, 30, 20, 67, 3, 26, 28, 14, 73, 16, 15, 40]"
100127250_450025097.txt,"recurrent uti meniere 's disease hyperlipidemia unspecified cerebral artery occlusion with cerebral infarction procedure : procedure(s ) : combined laser holmium cystoscopy , internal urethrotomy medication prior to admission",True,infarction,vte,99,109,0,121,13,"[44, 22, 85, 20, 46, 41, 54, 18, 149, 4, 177, 149, 4, 222, 114, 57, 54, 102, 39, 19, 41, 49, 2, 35, 21, 117, 56, 32, 3, 74, 29, 1, 48, 22, 106, 33, 57, 35, 21, 53, 20, 10, 21, 56, 39, 19, 41, 49, 2, 38, 21, 53, 20, 12, 51, 45, 56, 15, 10, 21, 53, 3, 74, 4, 43, 50, 1, 53, 38, 29, 14, 182, 49, 70, 55, 64, 91, 50, 100, 43, 176, 129, 83, 1, 61, 99, 82, 72, 6, 79, 87, 59, 99, 55, 1, 198, 37, 175, 74, 91, 155, 335, 16, 108, 148, 157, 37]"
100410234_454548766.txt,wanda oppegard be a 50 year old female pod # 0 from aortobifemoral bypass use a 16 x 8 mm knit bifurcated dacron graft and right femoral artery embolectomy perform for treatment of an infrarenal aortic thrombus with known heterogenous factor v leiden .,True,embolectomy,vte,148,159,0,257,3,"[4, 44, 11, 257, 114, 5, 2, 6, 41, 4, 33, 9, 24, 2, 22, 8, 22, 17, 2, 38, 20, 2, 10, 22, 5, 2, 50, 55, 24, 2, 94, 84, 5, 42, 12, 26, 70, 6, 30, 43, 6, 9, 12, 4, 44, 5, 10, 35, 9, 10, 2, 26, 39, 9, 24, 60, 49, 115, 18, 41, 51, 50, 11, 4, 56, 26, 13, 13, 13, 13, 8, 4, 13, 13, 8, 4, 204, 6, 10, 13, 5, 21, 37, 17, 8, 5, 16, 4, 13, 13, 13, 13, 13, 13, 48, 6, 4, 22, 12, 25, 13, 13, 8, 4, 34, 4, 4, 51, 13, 12, 11, 12, 4, 12, 5, 23, 4, 21, 50, 121, 66, 40, 74, 5, 79, 71, 29, 2, 11, 42, 44, 2, 50, 64, 2, 11, 40, 4, 30, 44, 89, 27]"
100483598_461871359.txt,pe and le dvt scan negative .,True,pe,vte,0,2,0,29,2,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100410234_454548766.txt,dispo : primary team to transfer out of sicu dvt prophylaxis : scds/asa gi prophylaxis : protonix,True,dvt,vte,44,47,0,94,30,"[4, 44, 11, 257, 114, 5, 2, 6, 41, 4, 33, 9, 24, 2, 22, 8, 22, 17, 2, 38, 20, 2, 10, 22, 5, 2, 50, 55, 24, 2, 94, 84, 5, 42, 12, 26, 70, 6, 30, 43, 6, 9, 12, 4, 44, 5, 10, 35, 9, 10, 2, 26, 39, 9, 24, 60, 49, 115, 18, 41, 51, 50, 11, 4, 56, 26, 13, 13, 13, 13, 8, 4, 13, 13, 8, 4, 204, 6, 10, 13, 5, 21, 37, 17, 8, 5, 16, 4, 13, 13, 13, 13, 13, 13, 48, 6, 4, 22, 12, 25, 13, 13, 8, 4, 34, 4, 4, 51, 13, 12, 11, 12, 4, 12, 5, 23, 4, 21, 50, 121, 66, 40, 74, 5, 79, 71, 29, 2, 11, 42, 44, 2, 50, 64, 2, 11, 40, 4, 30, 44, 89, 27]"
100483598_461871359.txt,pe and le dvt scan negative .,True,le dvt,vte,7,13,0,29,2,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100384447_467016914.txt,"dress change pod2 restarted asa from previous med list , mechanical , ambulation for dvt ppx follow up with dr . braman 6 week ,",True,dvt,vte,86,89,0,128,56,"[35, 39, 20, 39, 18, 18, 10, 13, 13, 13, 13, 8, 4, 3, 65, 15, 15, 8, 25, 11, 7, 29, 13, 13, 13, 13, 8, 4, 3, 65, 15, 15, 8, 25, 11, 7, 51, 71, 15, 44, 14, 36, 1, 2, 5, 44, 22, 19, 58, 16, 6, 45, 3, 67, 15, 37, 128, 102]"
100180552_455382736.txt,heparin drip d/c 's due to low suspicion for pe .,True,pe,vte,44,46,0,47,76,"[32, 18, 20, 27, 21, 75, 43, 47, 72, 51, 7, 27, 13, 13, 13, 3, 6, 13, 6, 37, 15, 8, 20, 8, 7, 51, 2, 31, 8, 10, 30, 45, 64, 7, 125, 25, 40, 44, 13, 13, 13, 8, 18, 51, 4, 15, 5, 4, 10, 23, 29, 6, 12, 6, 9, 5, 16, 51, 23, 5, 5, 10, 6, 6, 29, 63, 179, 81, 13, 5, 11, 25, 30, 2, 6, 58, 47, 40, 60, 14, 23, 52, 7, 48, 52, 26, 27, 30, 26, 3, 18, 19, 11, 9, 38, 79, 18, 34, 3, 21, 49, 30, 25, 112, 42, 25, 36, 25, 55, 16, 42, 19, 64, 58, 64, 24, 12, 41, 25, 53, 36, 9, 41]"
100446488_477602232.txt,"-pt/ot-adat-pain control , switch to oral long acting-bowel prophylaxis-dvt prophylaxis-x-rays scoliosis film complete-dc today .",True,dvt,vte,81,84,0,143,30,"[26, 13, 26, 28, 35, 12, 2, 25, 4, 40, 26, 11, 4, 29, 14, 4, 6, 5, 16, 56, 20, 8, 13, 60, 5, 100, 235, 50, 9, 77, 143, 51, 41]"
100446488_476389983.txt,"-pt/ot-adat-pain control , switch to oral long acting-bowel prophylaxis-dvt prophylaxis-x-ray when upright-disposition : when pass pt .",True,dvt,vte,81,84,0,147,28,"[26, 43, 21, 18, 2, 33, 16, 4, 40, 26, 11, 4, 88, 11, 13, 8, 15, 93, 11, 25, 30, 20, 28, 32, 30, 80, 9, 77, 147, 30, 39]"
100536370_453226772.txt,", -PRON- be aware of the risk of dvt and pe and possible death result from pe and continue to refuse pcd 's .",True,dvt,vte,30,33,0,108,99,"[12, 18, 15, 71, 69, 1, 42, 9, 23, 54, 15, 6, 33, 18, 24, 6, 28, 2, 26, 34, 8, 55, 34, 5, 24, 201, 21, 13, 20, 34, 10, 11, 4, 15, 19, 3, 17, 2, 1, 10, 13, 31, 3, 85, 4, 13, 8, 6, 10, 6, 9, 5, 10, 6, 35, 6, 35, 13, 8, 72, 176, 109, 70, 31, 79, 55, 56, 8, 3, 98, 171, 59, 89, 59, 94, 100, 21, 2, 48, 17, 31, 102, 110, 5, 13, 110, 38, 22, 76, 35, 25, 30, 15, 27, 62, 12, 45, 14, 119, 108, 117]"
100536370_453226772.txt,", -PRON- be aware of the risk of dvt and pe and possible death result from pe and continue to refuse pcd 's .",True,pe,vte,38,40,0,108,99,"[12, 18, 15, 71, 69, 1, 42, 9, 23, 54, 15, 6, 33, 18, 24, 6, 28, 2, 26, 34, 8, 55, 34, 5, 24, 201, 21, 13, 20, 34, 10, 11, 4, 15, 19, 3, 17, 2, 1, 10, 13, 31, 3, 85, 4, 13, 8, 6, 10, 6, 9, 5, 10, 6, 35, 6, 35, 13, 8, 72, 176, 109, 70, 31, 79, 55, 56, 8, 3, 98, 171, 59, 89, 59, 94, 100, 21, 2, 48, 17, 31, 102, 110, 5, 13, 110, 38, 22, 76, 35, 25, 30, 15, 27, 62, 12, 45, 14, 119, 108, 117]"
100483598_461871359.txt,assess for dvt comparison :,True,dvt,vte,11,14,0,26,72,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100536370_453226772.txt,", -PRON- be aware of the risk of dvt and pe and possible death result from pe and continue to refuse pcd 's .",True,pe,vte,75,77,0,108,99,"[12, 18, 15, 71, 69, 1, 42, 9, 23, 54, 15, 6, 33, 18, 24, 6, 28, 2, 26, 34, 8, 55, 34, 5, 24, 201, 21, 13, 20, 34, 10, 11, 4, 15, 19, 3, 17, 2, 1, 10, 13, 31, 3, 85, 4, 13, 8, 6, 10, 6, 9, 5, 10, 6, 35, 6, 35, 13, 8, 72, 176, 109, 70, 31, 79, 55, 56, 8, 3, 98, 171, 59, 89, 59, 94, 100, 21, 2, 48, 17, 31, 102, 110, 5, 13, 110, 38, 22, 76, 35, 25, 30, 15, 27, 62, 12, 45, 14, 119, 108, 117]"
100284288_464438433.txt,-PRON- have a complicated vascular history have undergo a left bka 9/23/13 for critical limb ischemia with unsucessful attempt at thrombolysis and result compartment syndrome .,True,thrombolysis,vte,130,142,0,178,18,"[81, 109, 52, 2, 34, 57, 30, 27, 31, 46, 57, 29, 49, 15, 39, 113, 18, 126, 178, 100, 123, 83, 117, 100, 172, 40, 76, 48, 45, 82, 44, 127, 144, 107, 66, 41, 15, 49, 92, 37, 58, 43, 52, 32, 44, 58, 116, 58, 69, 30, 47, 86, 31, 38, 16, 32, 52, 78, 105, 53, 78, 74, 34, 6, 50, 43, 31, 32, 48, 91, 72, 84, 62, 11, 27, 61, 87, 206, 55, 15, 65, 9, 114, 59, 98, 9, 100, 95, 30, 100, 3, 23, 13, 31]"
100483598_461871359.txt,no evidence for dvt in the bilateral low extremity .,False,dvt,vte,16,19,0,55,79,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 5  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 65, in begin_update
    N, mu, var = _get_moments(self.ops, X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 108, in _get_moments
    var = X.var(axis=1, keepdims=True) + 1e-08
  File "/opt/conda/lib/python3.9/site-packages/numpy/core/_methods.py", line 232, in _var
    x = asanyarray(arr - arrmean)
KeyboardInterrupt
Traceback (most recent call last):
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 331, in core_process
    for string_id, men, text, start, end, i, span in get_gaz_matches(nlp_neg, matcher, sent_list):
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 183, in get_gaz_matches
    doc = nlp(text.lower())
  File "/opt/conda/lib/python3.9/site-packages/spacy/language.py", line 445, in __call__
    doc = proc(doc, **component_cfg.get(name, {}))
  File "pipes.pyx", line 398, in spacy.pipeline.pipes.Tagger.__call__
  File "pipes.pyx", line 417, in spacy.pipeline.pipes.Tagger.predict
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 310, in predict
    X = layer(layer.ops.flatten(seqs_in, pad=pad))
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 131, in predict
    y, _ = self.begin_update(X, drop=None)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 375, in uniqued_fwd
    uniq_keys, ind, inv, counts = numpy.unique(
  File "<__array_function__ internals>", line 180, in unique
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 272, in unique
    ret = _unique1d(ar, return_index, return_inverse, return_counts)
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 360, in _unique1d
    ret += (np.diff(idx),)
  File "<__array_function__ internals>", line 180, in diff
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/function_base.py", line 1423, in diff
    a = op(a[slice1], a[slice2])
KeyboardInterrupt
[gms@thalia3 gold_standard]$ docker run -it  -v /var/lib/docker/data/vte/gold_standard/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:3  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv gs_manifest.csv data_in/ gold  vte

[gold] 0:gms@thalia3:/var/lib/docker/data/vte/gold_standard*                                                                                                           "thalia3.ahc.umn.edu" 14:31 29-Apr-22
6, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100643027_458075214.txt,"on 1/25/14 -PRON- groin suddenly become swollen and -PRON- undergo an emergent groin exploration and redo patch angioplasty to the cfa with hemashield patch and thrombectomy of the thrombosed profunda , femoral artery and common femoral artery .",True,thrombectomy,vte,156,168,0,238,13,"[53, 79, 23, 53, 57, 6, 35, 6, 58, 135, 122, 93, 104, 238, 123, 185, 49, 134, 138, 102, 26, 43, 134, 113, 34, 43, 16, 20, 26, 37, 31, 6, 26, 35, 20, 22, 28, 44, 5, 33, 41, 67, 78, 99, 55, 31, 66, 46, 44, 78, 48, 13, 51, 17, 188, 9, 29, 58, 98, 141, 53, 48, 17, 17, 29, 42, 72, 22, 31, 18, 66, 38, 21, 95, 24, 147, 3, 133, 77, 80, 42, 54, 30, 20, 100, 3, 23, 13, 13]"
100483598_461871359.txt,"charle dietz , md ct chest and pulmonary embolism",True,pulmonary embolism,vte,31,49,0,49,81,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100643027_458075214.txt,"on 1/28/14 , -PRON- return to the or for thrombectomy of the cfa , profunda and endarterectomy of the profunda with vein patch .",True,thrombectomy,vte,38,50,0,123,14,"[53, 79, 23, 53, 57, 6, 35, 6, 58, 135, 122, 93, 104, 238, 123, 185, 49, 134, 138, 102, 26, 43, 134, 113, 34, 43, 16, 20, 26, 37, 31, 6, 26, 35, 20, 22, 28, 44, 5, 33, 41, 67, 78, 99, 55, 31, 66, 46, 44, 78, 48, 13, 51, 17, 188, 9, 29, 58, 98, 141, 53, 48, 17, 17, 29, 42, 72, 22, 31, 18, 66, 38, 21, 95, 24, 147, 3, 133, 77, 80, 42, 54, 30, 20, 100, 3, 23, 13, 13]"
100483598_461871359.txt,assess for pulmonary embolism .,True,pulmonary embolism,vte,11,29,0,30,86,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100483598_461871359.txt,no central pe or evidence of right heart strain .,False,pe,vte,11,13,0,48,93,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 65, in begin_update
    N, mu, var = _get_moments(self.ops, X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 108, in _get_moments
    var = X.var(axis=1, keepdims=True) + 1e-08
  File "/opt/conda/lib/python3.9/site-packages/numpy/core/_methods.py", line 232, in _var
    x = asanyarray(arr - arrmean)
KeyboardInterrupt
Traceback (most recent call last):
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 331, in core_process
    for string_id, men, text, start, end, i, span in get_gaz_matches(nlp_neg, matcher, sent_list):
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 183, in get_gaz_matches
    doc = nlp(text.lower())
  File "/opt/conda/lib/python3.9/site-packages/spacy/language.py", line 445, in __call__
    doc = proc(doc, **component_cfg.get(name, {}))
  File "pipes.pyx", line 398, in spacy.pipeline.pipes.Tagger.__call__
  File "pipes.pyx", line 417, in spacy.pipeline.pipes.Tagger.predict
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 310, in predict
    X = layer(layer.ops.flatten(seqs_in, pad=pad))
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 131, in predict
    y, _ = self.begin_update(X, drop=None)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 375, in uniqued_fwd
    uniq_keys, ind, inv, counts = numpy.unique(
  File "<__array_function__ internals>", line 180, in unique
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 272, in unique
    ret = _unique1d(ar, return_index, return_inverse, return_counts)
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 360, in _unique1d
    ret += (np.diff(idx),)
  File "<__array_function__ internals>", line 180, in diff
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/function_base.py", line 1423, in diff
    a = op(a[slice1], a[slice2])
KeyboardInterrupt
[gms@thalia3 gold_standard]$ docker run -it  -v /var/lib/docker/data/vte/gold_standard/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:3  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv gs_manifest.csv data_in/ gold  vte

[gold] 0:gms@thalia3:/var/lib/docker/data/vte/gold_standard*                                                                                                           "thalia3.ahc.umn.edu" 14:39 29-Apr-22
, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100497151_461825013.txt,"patient have undergo mouth floor excision of cancer with free flap , chronic dvt , hyperglycemia of critical illness , hypokalemia : i spend 45 minute while the patient be in this condition , provide pain control , maintain fluid balance , pulmonary toilet and supplemental oxygen , flap hemodynamic monitoring , nutritional support , potassium replacement , insulin for glucose control .",True,dvt,vte,77,80,0,385,19,"[23, 67, 44, 29, 51, 21, 32, 32, 17, 11, 14, 2, 146, 62, 138, 115, 172, 376, 71, 385, 89, 52]"
100410234_452624367.txt,vascular surgery 50 year old female with history of unprovoked dvt 15 year ago with subsequent cessation of anticoagulation by patient .,True,unprovoked dvt,vte,52,66,0,136,0,"[136, 68, 36, 79, 80, 90, 47, 48, 45, 53, 36, 8]"
100284288_459217684.txt,: sicu dvt prophylaxis : ranitidine bid gi prophylaxis :,True,dvt,vte,7,10,0,54,15,"[37, 19, 158, 204, 72, 6, 12, 78, 20, 70, 318, 22, 261, 26, 29, 54, 42, 91, 160, 11, 50, 6, 10, 44, 10, 2, 26, 36, 9, 12, 46, 10, 6, 15, 7, 29, 92, 6, 70, 80, 266, 13, 13, 46, 36, 39, 1, 5, 5, 13, 7, 3, 55, 4, 27, 13, 27, 13, 12, 33, 5, 37, 5, 83, 49, 24, 1, 29, 13, 24, 11, 32, 45, 10, 92, 77, 45, 54, 44, 36, 94, 94, 82, 4, 43, 154, 11, 55, 96, 75, 55, 47, 12, 4, 21, 63, 13, 55, 19, 21, 24, 20, 22, 3, 53, 25, 42, 29, 172, 25]"
100483598_461871359.txt,no evidence of central pulmonary embolism or right heart strain .,False,pulmonary embolism,vte,23,41,0,64,114,"[47, 16, 29, 37, 10, 26, 40, 46, 46, 6, 27, 2, 26, 36, 22, 34, 18, 25, 24, 82, 111, 29, 55, 102, 56, 1, 17, 3, 14, 18, 33, 31, 59, 13, 36, 32, 23, 85, 17, 14, 38, 5, 32, 78, 7, 13, 24, 8, 78, 7, 2, 35, 8, 78, 7, 2, 22, 29, 78, 7, 57, 55, 7, 36, 78, 7, 24, 22, 16, 10, 69, 8, 26, 52, 18, 131, 59, 11, 11, 55, 100, 49, 16, 10, 32, 18, 30, 85, 10, 155, 37, 9, 54, 48, 24, 23, 74, 1, 83, 53, 41, 98, 126, 7, 62, 177, 40, 77, 137, 198, 27, 11, 14, 54, 64, 2, 134, 69, 2, 91, 2, 114, 2, 149, 2, 152, 77, 206, 34, 100, 52, 34, 19, 3, 10, 6, 23, 14, 87, 17, 52, 46, 3, 9, 6, 17, 3, 14, 18, 33, 36, 3, 20, 33, 53, 17, 14, 36, 17, 11, 5, 3, 38, 30, 15, 1, 60, 7, 22, 31, 3, 11, 11, 4, 24, 43, 12, 5, 57, 13, 26]"
100438307_454878916.txt,dvt prophylaxis 40 mg lovenox daily 4 .,True,dvt,vte,0,3,0,37,28,"[11, 55, 45, 9, 33, 10, 9, 20, 4, 8, 15, 52, 15, 16, 32, 17, 14, 2, 12, 11, 5, 104, 138, 89, 47, 2, 41, 47, 37, 18, 23, 40, 131, 127]"
100607223_462966279.txt,"lab : type and cross 2u for or , also check hgb consent : not yet do , will be do in pre-op dvt prophylaxis :",True,dvt,vte,93,96,0,109,6,"[17, 5, 16, 13, 18, 90, 109, 14, 23]"
100611415_460058721.txt,ppi dvt prophylaxis : pcd intravascular access and device : rij event of last 24 hour :,True,dvt,vte,4,7,0,86,21,"[49, 60, 73, 69, 10, 15, 21, 21, 49, 79, 32, 16, 66, 2, 116, 2, 23, 31, 8, 29, 58, 86, 13, 65, 84, 19, 79, 193, 8, 22, 10, 9, 21, 28, 71, 15, 16, 32, 17, 15, 2, 11, 53, 20, 45, 8, 18, 17, 16, 2, 17, 23, 9, 25, 10, 19, 7, 7, 14, 43, 6, 43, 16, 33, 11, 324, 14, 70, 94, 21, 55, 63, 46, 24, 5, 5, 4, 26, 54, 1, 5, 16, 4, 16, 47, 25, 33, 16, 51, 21, 22, 32, 29, 37, 6, 20, 105, 45, 128, 5, 5, 5, 41, 37, 279, 23]"
100607223_463698911.txt,po naloxone repeat hgb in be mechanical dvt prophylaxis,True,dvt,vte,40,43,0,55,8,"[115, 67, 11, 7, 112, 72, 10, 29, 55]"
100611415_462558591.txt,"sf revision with left thoracotomy with dr polly 2/10/14 and dr chipman --pain management , abx , wound care and dvt prophylaxis as per primary team .",True,dvt,vte,110,113,0,146,5,"[147, 76, 99, 7, 11, 146, 19, 33, 21, 15, 13, 2, 6, 14, 24, 29, 20, 24, 2, 41, 39, 68, 49, 22, 35, 21, 19, 48, 2, 14, 21, 13, 10, 26, 9, 86, 2, 13, 15, 31, 38, 35, 2, 49, 9, 42, 2, 28, 28, 8, 28, 8, 15, 20, 28, 28, 13, 186, 16, 21, 23, 81, 30, 6, 43, 3, 33, 11, 95, 77, 70, 43, 16, 19, 21, 13, 8, 4, 13, 13, 13, 38, 14, 31, 9, 4, 4, 31, 4, 60, 27, 2, 3, 3, 1, 2, 21, 3, 3, 1, 2, 2, 18, 5, 4, 42, 37, 84, 67, 13, 13, 13, 13, 31, 6, 4, 4, 6, 21, 6, 4, 8, 6, 4, 4, 6, 121, 5, 32, 1, 104, 5, 5, 35, 33]"
100410234_453046371.txt,a/p : 50 year old female with history of unprovoked dvt 15 year ago with subsequent cessation of anticoagulation by patient .,True,unprovoked dvt,vte,40,54,0,124,25,"[15, 2, 37, 15, 6, 22, 49, 6, 10, 38, 15, 16, 50, 10, 2, 11, 2, 9, 2, 4, 10, 7, 73, 1, 37, 124, 68, 36, 79, 80, 95, 76]"
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ ls
clamp_out  clamp_out_1  clamp_out.zip  data_in  data_new  final_lex.csv  gs_manifest.csv  mention_standard  nlptab_manifest.txt
[gms@thalia3 gold_standard]$ ls -la  mention_standard
-rw-r--r-- 1 root root 76213 Apr 29 14:30 mention_standard
[gms@thalia3 gold_standard]$ rm  mention_standard
rm: remove write-protected regular file ‘mention_standard’? y
[gms@thalia3 gold_standard]$ ls -la
total 15524
drwxr-xr-x 6 gms  domain users     165 Apr 29 14:32 .
drwxr-xr-x 5 gms  root             222 Apr 28 13:11 ..
drwxr-xr-x 2 root root          495616 Apr 29 11:49 clamp_out
drwxr-xr-x 2 root root         6479872 Apr 29 09:28 clamp_out_1
-rw-r--r-- 1 root root           13911 Apr 29 11:49 clamp_out.zip
drwxr-xr-x 2 gms  domain users 3485696 Apr 29 12:47 data_in
drwxr-xr-x 2 gms  domain users  249856 Apr 28 17:42 data_new
-rwxr-xr-x 1 gms  domain users   17116 Apr 29 12:49 final_lex.csv
-rw-r--r-- 1 gms  domain users 2044981 Apr 29 12:49 gs_manifest.csv
-rw-r--r-- 1 root root              15 Apr 29 09:28 nlptab_manifest.txt
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ top
top - 14:40:14 up 19 days, 12:29,  1 user,  load average: 1.09, 2.77, 3.44
Tasks: 422 total,   1 running, 421 sleeping,   0 stopped,   0 zombie
%Cpu(s):  3.6 us,  1.9 sy,  0.0 ni, 94.4 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 13184876+total, 76033504 free, 51269000 used,  4546260 buff/cache
KiB Swap:        0 total,        0 free,        0 used. 76253744 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
13634 root      20   0 2070140 815452   2364 S  89.0  0.6   8:09.59 python
12795 root      20   0 1535100 541268  19952 S  42.9  0.4   4:15.27 python
 1535 root      20   0 2775952 118408  41020 S   3.3  0.1 803:16.76 kubelet
12436 root      20   0 3003436  48868  22612 S   3.3  0.0 529:57.86 calico-node
 2058 root      20   0 4071296  96336  17540 S   1.3  0.1 518:25.07 dockerd-current
 2962 rmcewan   20   0   11.7g   1.2g  15096 S   1.0  0.9  36:18.71 java
 3334 rmcewan   20   0   74.1g  43.1g  16192 S   1.0 34.2   3358:04 java
    9 root      20   0       0      0      0 S   0.3  0.0  37:56.75 rcu_sched
 1524 root      20   0 1814360  25004   4356 S   0.3  0.0 162:45.33 rhel-push-plugi
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 65, in begin_update
    N, mu, var = _get_moments(self.ops, X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 108, in _get_moments
    var = X.var(axis=1, keepdims=True) + 1e-08
  File "/opt/conda/lib/python3.9/site-packages/numpy/core/_methods.py", line 232, in _var
    x = asanyarray(arr - arrmean)
KeyboardInterrupt
Traceback (most recent call last):
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 331, in core_process
    for string_id, men, text, start, end, i, span in get_gaz_matches(nlp_neg, matcher, sent_list):
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 183, in get_gaz_matches
    doc = nlp(text.lower())
  File "/opt/conda/lib/python3.9/site-packages/spacy/language.py", line 445, in __call__
    doc = proc(doc, **component_cfg.get(name, {}))
  File "pipes.pyx", line 398, in spacy.pipeline.pipes.Tagger.__call__
  File "pipes.pyx", line 417, in spacy.pipeline.pipes.Tagger.predict
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 310, in predict
    X = layer(layer.ops.flatten(seqs_in, pad=pad))
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 131, in predict
    y, _ = self.begin_update(X, drop=None)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 375, in uniqued_fwd
    uniq_keys, ind, inv, counts = numpy.unique(
  File "<__array_function__ internals>", line 180, in unique
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 272, in unique
    ret = _unique1d(ar, return_index, return_inverse, return_counts)
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 360, in _unique1d
    ret += (np.diff(idx),)
  File "<__array_function__ internals>", line 180, in diff
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/function_base.py", line 1423, in diff
    a = op(a[slice1], a[slice2])
KeyboardInterrupt
[gms@thalia3 gold_standard]$ docker run -it  -v /var/lib/docker/data/vte/gold_standard/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:3  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv gs_manifest.csv data_in/ gold  vte

[gold] 0:gms@thalia3:/var/lib/docker/data/vte/gold_standard*                                                                                                           "thalia3.ahc.umn.edu" 14:42 29-Apr-22
 1578 root      20   0  958708  37996   9080 S   0.3  0.0  90:30.32 python
 3337 rmcewan   20   0   36.6g   1.4g  16020 S   0.3  1.1   9:08.53 java
12248 root      20   0  487564   7596   1732 S   0.3  0.0   2:19.44 docker-containe
12689 root      20   0    1712    732    560 S   0.3  0.0   5:54.05 bird6
14823 gms       20   0  173084   2712   1644 R   0.3  0.0   0:00.05 top
23307 root      20   0 2911748  61784  23796 S   0.3  0.0 135:01.46 apiserver
    1 root      20   0  192032   5056   2644 S   0.0  0.0  18:55.70 systemd
    2 root      20   0       0      0      0 S   0.0  0.0   0:01.52 kthreadd
    4 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/0:0H
    6 root      20   0       0      0      0 S   0.0  0.0   0:42.38 ksoftirqd/0
    7 root      rt   0       0      0      0 S   0.0  0.0   0:03.28 migration/0
    8 root      20   0       0      0      0 S   0.0  0.0   0:00.00 rcu_bh
   10 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 lru-add-drain
   11 root      rt   0       0      0      0 S   0.0  0.0   1:08.16 watchdog/0
   12 root      rt   0       0      0      0 S   0.0  0.0   0:05.08 watchdog/1
   13 root      rt   0       0      0      0 S   0.0  0.0   0:02.96 migration/1
   14 root      20   0       0      0      0 S   0.0  0.0   0:02.57 ksoftirqd/1
   16 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/1:0H
   18 root      rt   0       0      0      0 S   0.0  0.0   0:05.01 watchdog/2
   19 root      rt   0       0      0      0 S   0.0  0.0   0:02.80 migration/2
   20 root      20   0       0      0      0 S   0.0  0.0   0:07.41 ksoftirqd/2
   22 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/2:0H
   24 root      rt   0       0      0      0 S   0.0  0.0   0:04.64 watchdog/3
   25 root      rt   0       0      0      0 S   0.0  0.0   0:02.90 migration/3
   26 root      20   0       0      0      0 S   0.0  0.0   0:01.96 ksoftirqd/3
   28 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/3:0H
   29 root      rt   0       0      0      0 S   0.0  0.0   0:04.42 watchdog/4
   30 root      rt   0       0      0      0 S   0.0  0.0   0:02.91 migration/4
   31 root      20   0       0      0      0 S   0.0  0.0   0:02.74 ksoftirqd/4
   33 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/4:0H
   34 root      rt   0       0      0      0 S   0.0  0.0   0:04.50 watchdog/5
   35 root      rt   0       0      0      0 S   0.0  0.0   0:03.19 migration/5
   36 root      20   0       0      0      0 S   0.0  0.0   0:02.17 ksoftirqd/5
   38 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/5:0H
   39 root      rt   0       0      0      0 S   0.0  0.0   0:04.69 watchdog/6
[gms@thalia3 gold_standard]$ ls
clamp_out  clamp_out_1  clamp_out.zip  data_in  data_new  final_lex.csv  gs_manifest.csv  nlptab_manifest.txt
[gms@thalia3 gold_standard]$ lls ../icd_negative/
-bash: lls: command not found
[gms@thalia3 gold_standard]$ ls ../icd_negative/
clamp_out                      mention_1649097083.935404.csv  mention_1649099152.809069.csv    negatives_1647444271.216377.csv  negatives_1649098409.906647.csv  negatives_1649100052.371133.csv
clamp_out.zip                  mention_1649097199.664544.csv  mention_1649099685.450958.csv    negatives_1649097083.935404.csv  negatives_1649098833.116109.csv  negatives_1649100851.23886.csv
data_in                        mention_1649097583.277007.csv  mention_1649099771.109788.csv    negatives_1649097199.664544.csv  negatives_1649099093.556175.csv  negatives_1649101361.694788.csv
final_lex.csv                  mention_1649098041.336083.csv  mention_1649100052.371133.csv    negatives_1649097583.277007.csv  negatives_1649099152.809069.csv  nlptab_manifest.txt
manifest_negative.csv          mention_1649098216.889112.csv  mention_1649100851.23886.csv     negatives_1649097958.898895.csv  negatives_1649099606.678618.csv  test_manifext.csv
mention_1646840536.840928.csv  mention_1649098409.906647.csv  mention_1649101361.694788.csv    negatives_1649098041.336083.csv  negatives_1649099685.450958.csv  test_mf.csv
mention_1647444271.216377.csv  mention_1649098833.116109.csv  negatives_1646840536.840928.csv  negatives_1649098216.889112.csv  negatives_1649099771.109788.csv
[gms@thalia3 gold_standard]$ ls
clamp_out  clamp_out_1  clamp_out.zip  data_in  data_new  final_lex.csv  gs_manifest.csv  nlptab_manifest.txt
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ top
top - 15:03:00 up 19 days, 12:51,  1 user,  load average: 30.12, 27.90, 19.23
Tasks: 442 total,  24 running, 418 sleeping,   0 stopped,   0 zombie
    N, mu, var = _get_moments(self.ops, X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/layernorm.py", line 108, in _get_moments
    var = X.var(axis=1, keepdims=True) + 1e-08
  File "/opt/conda/lib/python3.9/site-packages/numpy/core/_methods.py", line 232, in _var
    x = asanyarray(arr - arrmean)
KeyboardInterrupt
Traceback (most recent call last):
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/opt/conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 331, in core_process
    for string_id, men, text, start, end, i, span in get_gaz_matches(nlp_neg, matcher, sent_list):
  File "/home/gazetteer/gazetteer_multiprocess_sbd.py", line 183, in get_gaz_matches
    doc = nlp(text.lower())
  File "/opt/conda/lib/python3.9/site-packages/spacy/language.py", line 445, in __call__
    doc = proc(doc, **component_cfg.get(name, {}))
  File "pipes.pyx", line 398, in spacy.pipeline.pipes.Tagger.__call__
  File "pipes.pyx", line 417, in spacy.pipeline.pipes.Tagger.predict
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 310, in predict
    X = layer(layer.ops.flatten(seqs_in, pad=pad))
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/feed_forward.py", line 40, in predict
    X = layer(X)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 167, in __call__
    return self.predict(x)
  File "/opt/conda/lib/python3.9/site-packages/thinc/neural/_classes/model.py", line 131, in predict
    y, _ = self.begin_update(X, drop=None)
  File "/opt/conda/lib/python3.9/site-packages/thinc/api.py", line 375, in uniqued_fwd
    uniq_keys, ind, inv, counts = numpy.unique(
  File "<__array_function__ internals>", line 180, in unique
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 272, in unique
    ret = _unique1d(ar, return_index, return_inverse, return_counts)
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/arraysetops.py", line 360, in _unique1d
    ret += (np.diff(idx),)
  File "<__array_function__ internals>", line 180, in diff
  File "/opt/conda/lib/python3.9/site-packages/numpy/lib/function_base.py", line 1423, in diff
    a = op(a[slice1], a[slice2])
KeyboardInterrupt
[gms@thalia3 gold_standard]$ docker run -it  -v /var/lib/docker/data/vte/gold_standard/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:3  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv gs_manifest.csv data_in/ gold  vte
number of cores in the system: 24

[gold] 0:gms@thalia3:/var/lib/docker/data/vte/gold_standard*                                                                                                           "thalia3.ahc.umn.edu" 15:03 29-Apr-22
%Cpu(s): 95.5 us,  3.8 sy,  0.0 ni,  0.5 id,  0.1 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 13184876+total, 73524784 free, 53899300 used,  4424680 buff/cache
KiB Swap:        0 total,        0 free,        0 used. 73614760 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
 3334 rmcewan   20   0   74.1g  43.1g  16192 S 306.6 34.3   3399:03 java
 3337 rmcewan   20   0   36.6g   1.4g  16020 S   0.0  1.1   9:12.50 java
 2962 rmcewan   20   0   11.7g   1.2g  15096 S   1.6  0.9  36:39.88 java
13634 root      20   0 4208868   1.0g   2396 S   0.7  0.8  14:46.40 python
15980 root      20   0 1671364 639532   4428 R  91.8  0.5  13:20.04 python
15994 root      20   0 1668200 636476   4424 R  92.5  0.5  13:10.80 python
15908 root      20   0 1668684 636376   4424 R  92.8  0.5  12:41.01 python
15900 root      20   0 1668028 636180   4428 R  77.0  0.5  12:42.15 python
15986 root      20   0 1667404 635028   4424 R  90.8  0.5  13:16.32 python
15972 root      20   0 1666780 634932   4424 R  95.1  0.5  12:53.32 python
15916 root      20   0 1667416 634824   4424 R  94.4  0.5  13:18.77 python
15904 root      20   0 1667892 634708   4424 R  86.6  0.5  12:29.21 python
15920 root      20   0 1666880 634628   4424 S  74.1  0.5  12:44.28 python
15949 root      20   0 1667392 634528   4424 R  81.3  0.5  12:54.86 python
16000 root      20   0 1666520 634448   4424 R  87.5  0.5  13:32.28 python
15933 root      20   0 1664720 631812   4424 R  87.5  0.5  12:52.90 python
15957 root      20   0 1663732 630960   4424 R  80.7  0.5  13:17.47 python
15988 root      20   0 1662236 630516   4424 R  81.6  0.5  13:22.35 python
15945 root      20   0 1662500 630456   4424 R  80.3  0.5  13:16.84 python
15924 root      20   0 1661180 629508   4428 R  93.1  0.5  12:58.00 python
15941 root      20   0 1660884 628124   4424 R  77.7  0.5  12:54.18 python
15976 root      20   0 1659356 627076   4424 R  89.8  0.5  13:08.82 python
15937 root      20   0 1658280 626744   4700 R  82.0  0.5  13:07.30 python
15997 root      20   0 1658168 624768   4424 R  79.0  0.5  13:04.30 python
15928 root      20   0 1656368 624448   4424 R  87.5  0.5  12:52.45 python
15912 root      20   0 1652304 620448   4424 R  73.8  0.5  12:48.84 python
15953 root      20   0 1648812 616840   4424 R  91.1  0.5  13:21.01 python
15984 root      20   0 1647556 615060   4424 R  77.7  0.5  13:04.09 python
12795 root      20   0 1606884 587076  20292 S   0.0  0.4   7:37.80 python
 3038 root      20   0  497532 327048  18548 S   0.7  0.2  81:37.98 splunkd
 1535 root      20   0 2775952 116248  41112 S   6.2  0.1 804:17.97 kubelet
15813 root      20   0  871640 100428  20104 S   0.0  0.1  25:26.68 minio
 2058 root      20   0 4071296  96460  17540 S   1.0  0.1 519:02.45 dockerd-current
23307 root      20   0 2911748  61416  23796 S   1.3  0.0 135:14.82 apiserver
 1851 splunk    20   0 2829876  51396  19960 S   0.7  0.0 208:53.59 operator
12436 root      20   0 3003436  49668  22612 S   2.6  0.0 530:33.49 calico-node
12009 polkitd   20   0 2824204  47404  17048 S   0.0  0.0  50:04.91 calico-typha
  724 root      20   0   97288  45588  45056 S   0.0  0.0  22:08.07 systemd-journal
 5640 polkitd   20   0 2457700  39884  16620 S   0.0  0.0   4:07.83 kube-controller
12437 root      20   0 2560276  39716  17992 S   0.0  0.0   3:52.39 calico-node
 1354 root      20   0  299984  38628  37272 S   0.0  0.0   0:46.53 sssd_nss
 1578 root      20   0  958708  37996   9080 S   0.7  0.0  90:37.09 python
12439 root      20   0 2781728  37768  15760 S   0.0  0.0   3:07.96 calico-node
 4441 root      20   0 2412556  32260  18488 S   0.0  0.0   0:55.63 calico-node
[gms@thalia3 gold_standard]$ tmux attach -t gold
[detached]
[gms@thalia3 gold_standard]$ ls
clamp_out  clamp_out_1  clamp_out.zip  data_in  data_new  final_lex.csv  gs_manifest.csv  mention_1651260651.770565.csv  nlptab_manifest.txt
[gms@thalia3 gold_standard]$ ls -la
total 17572
drwxr-xr-x 6 gms  domain users     202 Apr 29 14:47 .
drwxr-xr-x 5 gms  root             222 Apr 28 13:11 ..
drwxr-xr-x 2 root root          495616 Apr 29 11:49 clamp_out
drwxr-xr-x 2 root root         6479872 Apr 29 09:28 clamp_out_1
-rw-r--r-- 1 root root           13911 Apr 29 11:49 clamp_out.zip
drwxr-xr-x 2 gms  domain users 3485696 Apr 29 12:47 data_in
drwxr-xr-x 2 gms  domain users  249856 Apr 28 17:42 data_new
-rwxr-xr-x 1 gms  domain users   17116 Apr 29 12:49 final_lex.csv
-rw-r--r-- 1 gms  domain users 2044981 Apr 29 12:49 gs_manifest.csv
-rw-r--r-- 1 root root         1629553 Apr 29 15:03 mention_1651260651.770565.csv
-rw-r--r-- 1 root root              15 Apr 29 09:28 nlptab_manifest.txt
[gms@thalia3 gold_standard]$ ls -la
total 17572
drwxr-xr-x 6 gms  domain users     202 Apr 29 14:47 .
drwxr-xr-x 5 gms  root             222 Apr 28 13:11 ..
drwxr-xr-x 2 root root          495616 Apr 29 11:49 clamp_out
drwxr-xr-x 2 root root         6479872 Apr 29 09:28 clamp_out_1
-rw-r--r-- 1 root root           13911 Apr 29 11:49 clamp_out.zip
drwxr-xr-x 2 gms  domain users 3485696 Apr 29 12:47 data_in
drwxr-xr-x 2 gms  domain users  249856 Apr 28 17:42 data_new
-rwxr-xr-x 1 gms  domain users   17116 Apr 29 12:49 final_lex.csv
-rw-r--r-- 1 gms  domain users 2044981 Apr 29 12:49 gs_manifest.csv
-rw-r--r-- 1 root root         1639233 Apr 29 15:03 mention_1651260651.770565.csv
-rw-r--r-- 1 root root              15 Apr 29 09:28 nlptab_manifest.txt
[gms@thalia3 gold_standard]$ history | grep gaze
  274  docker pull ahc-nlpie-docker.artifactory.umn.edu/gazetteer
  358  docker  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python -u gazetteer_multiprocess_sbd.py /data/test_lex.csv /data/manifest.csv /data/data_in/ out vte
  359  docker  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /vte/test_lex.csv /data/manifest_positive.csv /data/icd_positive/data_in/ /data/positives vte
  360  docker run -it  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /vte/test_lex.csv /data/manifest_positive.csv /data/icd_positive/data_in/ /data/positives vte
  361  docker run -it  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /data/vte_lex.csv /data/manifest_positive.csv /data/icd_positive/data_in/ /data/positives vte
  363  docker run -it  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /data/vte_lex.csv /data/manifest_positive.csv /data/icd_positive/data_in/ /data/positives vte
  365  docker run -it  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  -u gazetteer_multiprocess_sbd.py /data/vte_lex.csv /data/manifest_positive.csv /data/icd_positive/data_in/ /data/positives vte
  481  docker run -it  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /data/vte_lex.csv /data/manifest_positive.csv /data/icd_positive/data_in/ /data/positives vte
  482  docker run -it  -v /data/ahc-ie/TignanelliC-Req02515/Workspace/extraction/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /data/vte_lex.csv /data/manifest_negative.csv /data/icd_negative/data_in/ /data/negatives vte
  483  docker run -it  -v /var/lib/docker/data/vte/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python  gazetteer_multiprocess_sbd.py /data/final_lex.csv /data/manifest_negative.csv /data/icd_negative/data_in/ /data/negatives vte
  507  docker pull ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1
  508  docker run -it  -v /var/lib/docker/data/vte/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer  python -u  gazetteer_multiprocess_sbd.py final_lex.csv manifest_negative.csv icd_negative/data_in/ icd_negatives/negatives vte
  509  docker run -it  -v /var/lib/docker/data/vte/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  gazetteer_multiprocess_sbd.py final_lex.csv manifest_negative.csv icd_negative/data_in/ icd_negatives/negatives vte
  510  docker run -it  -v /var/lib/docker/data/vte/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv manifest_negative.csv icd_negative/data_in/ icd_negatives/negatives vte
  511  docker run -it  -v /var/lib/docker/data/vte/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv manifest_nmaanifest_negative.csv icd_negative/data_in/ icd_negative/negatives vte
  512  docker run -it  -v /var/lib/docker/data/vte/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv manifest_negative.csv icd_negative/data_in/ icd_negative/negatives vte
  513  docker run -it  -v /var/lib/docker/data/vte/icd_negative/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv manifest_negative.csv data_in/ negatives vte
  517  docker run -it  -v /var/lib/docker/data/vte/icd_negative/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv manifest_neg                    if (len(sub.split()) >= threshold):
                        neg = nlp_neg(sub)
                        for e in neg.ents:
                            #print(e.text)
                            #print(new_str)
                            #if (e.label_ == string_id):
                            if (new_str == e.text) and (e.label_ == string_id):
                                #content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + ', ' + string_id +  ', (' + str(start) + ',' + str(end) + '), ' + str(i) +'\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, string_id)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, string_id)

                                # sentence-level mention
                                mention = { "file": name,
                                            "sentence": new_str_sent,
                                            "polarity": men_bool,
                                            "men": e.text,
                                            "concept": string_id,
                                            "start": span.start_char,
                                            "end": span.end_char,
                                            "span.sent.start_char": span.sent.start_char,
                                            "span.sent.end_char": span.sent.end_char,
                                            "sentence_n": i,
                                            "sent_lengths": sent_length }

                                write_mention(mention, 'mention_' + output.split('_')[1])

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
"gazetteer_multiprocess_sbd.py" [dos] 473L, 16786B                                                                                                                                        377,65        83%
                    if (len(sub.split()) >= threshold):
                        neg = nlp_neg(sub)
                        for e in neg.ents:
                            #print(e.text)
                            #print(new_str)
                            #if (e.label_ == string_id):
                            if (new_str == e.text) and (e.label_ == string_id):
                                #content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + ', ' + string_id +  ', (' + str(start) + ',' + str(end) + '), ' + str(i) +'\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, string_id)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, string_id)

                                # sentence-level mention
                                mention = { "file": name,
                                            "sentence": new_str_sent,
                                            "polarity": men_bool,
                                            "men": e.text,
                                            "concept": string_id,
                                            "start": span.start_char,
                                            "end": span.end_char,
                                            "span.sent.start_char": span.sent.start_char,
                                            "span.sent.end_char": span.sent.end_char,
                                            "sentence_n": i,
                                            "sent_lengths": sent_length }

                                write_mention(mention, 'mention_' + output.split('_')[1])

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
"gazetteer_multiprocess_sbd.py" [dos] 473L, 16786B                                                                                                                                        377,65        83%
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
-- VISUAL LINE --                                                                                                                                                               473       1,1           Top
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


search hit BOTTOM, continuing at TOP
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

def write_mention(_dict, file_path):

    with open(file_path, 'a') as file:
        w = csv.DictWriter(file, _dict.keys())

        if file.tell() == 0:
            w.writeheader()


        w.writerow(_dict)

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
/mention

                name = file.strip()
                #content = name + ', [' + new_str_sent + '], ' + men + ', ' + string_id + ', (' + str(start) + ',' + str(end) + '), ' + str(i) + ', ' + '\n'
                #print(content)

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
                                #content = name + ', [' + new_str_sent + '], ' + e.text + ', ' + str(not e._.negex) + ', ' + string_id +  ', (' + str(start) + ',' + str(end) + '), ' + str(i) +'\n'
                                #print(content)
                                men_bool = not e._.negex
                                if men_bool:
                                    update_mdict(dict_files_positive, name, string_id)
                                if men_bool == False:
                                    update_mdict(dict_files_negative, name, string_id)

                                # sentence-level mention
                                mention = { "file": name,
                                            "sentence": new_str_sent,
                                            "polarity": men_bool,
                                            "men": e.text,
                                            "concept": string_id,
                                            "start": span.start_char,
                                            "end": span.end_char,
                                            "span.sent.start_char": span.sent.start_char,
                                            "span.sent.end_char": span.sent.end_char,
                                            "sentence_n": i,
                                            "sent_lengths": sent_length }

                                write_mention(mention, 'mention_' + output.split('_')[1])

                                break

def mention_using_gaz(nlp_lemma, gaz_csv_list, notes_for_training, doc_folder, dict_gaz, prefix, output):

    manager = mp.Manager()
    dict_files_positive = manager.dict()
    dict_files_negative = manager.dict()

    init_dict(manager, dict_files_positive, notes_for_training, dict_gaz)
    init_dict(manager, dict_files_negative, notes_for_training, dict_gaz)

                                                                                                                                                                                          368,50        81%
                                
                                write_mention(mention, 'mention_' + output.split('_')[1])
                                
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
                 dict_files_positive, dict_files_negative, output, )))
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

