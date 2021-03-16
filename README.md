# covid_symptoms_gazetteer

covid_symptom_gazetteer (hereby referred as "COVID-19 gazetteer") is a high throughput annotation system for real-time indexing of COVID-19 unstructred clinical notes. 

## COVID-19 Gazetteer Architecture

The COVID-19 gazetteer utilizes spaCy’s Matcher [1] class along withEntityRuler [2] class to add the terms in gazetteer lexicon to the spaCy en_core_web_sm [3] model. The Matcher instance reads in ED admission notes and returns symptom mentions and the span of text containing each mention. Returned spans are further processed by the spaCypipeline to search for custom entities added by the EntityRuler. Theoutput is then lemmatized to convert the text to its canonical form.The NegEx component of spaCy (negspaCy [4]) is used for negation detection.

Rule-based matching [5] of COVID-19 gazetteer lexicon is automated using the following token attributes in spaCy: base form of the word (LEMMA); universal part-of-speech tag (POS); detailed POS tag (TAG); and punctuation (IS PUNCT) [6].

## Data

1. Lexicon of symptoms clustered around 11 major Center for Disease Control and Prevention (CDC) COVID-19 symptoms [7]: GAZ_group.csv

## Requirements

- pandas==1.1.3
- scispacy==0.3.0
- spacy==2.3.5
- negspacy==0.1.9

## Creating and Executing COVID-19 Gazetteer

To create a docker image, simply run the following in the main directory:

```docker build -t ahc-nlpie-docker.artifactory.umn.edu/covid_gazetteer .```

To run the created docker, type:

```docker run -it -v mount_dir:/data ahc-nlpie-docker.artifactory.umn.edu/covid_gazetteer:latest python gazetteer_multiprocess.py GAZ_group.csv /data/notes_to_process.csv /data/data_in /data/ann_out```

The important arguments to docker command are:

   - notes_to_process.csv: notes to be annotated. The file should not contain any header.
   - data_in: directory containing the notes.
   - mount_dir: directory containing notes_to_process.csv and data_in directory.
   - ann_out: annotated output.

References:

1. spaCy Matcher: https://spacy.io/api/matcher

2. spaCy EntityRuler: https://spacy.io/api/entityruler

3. spaCy models: https://spacy.io/usage/models

4. negspaCy: https://spacy.io/universe/project/negspacy

5. spaCy Rule-Based Matching: https://spacy.io/usage/rule-based-matching

6. spaCy Token: https://spacy.io/api/token

7. Symptoms of Coronavirus: https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html
