# covid_symptoms_gazetteer

covid_symptom_gazetteer (hereby referred as "COVID-19 gazetteer") is a high throughput annotation system for real-time indexing of COVID-19 unstructred clinical notes. 

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

```docker run -it -v mount_dir:/data ahc-nlpie-docker.artifactory.umn.edu/covid_gazetteer:latest python umls_rb_hybrid_gazetteer.py GAZ_group.csv /data/notes_to_process.csv /data/data_in /data/ann_out prefix_term```

The important arguments to docker command are:

   - notes_to_process.csv: notes to be annotated. The file should not contain any header.
   - data_in: directory containing the notes.
   - mount_dir: directory containing notes_to_process.csv and data_in directory.
   - ann_out: annotated output.
   - prefix_term: phrase to prefix the features in the output.

Note: The umls_rb_hybrid_gazetteer.py was designed for only the listed 11 CDC COVID-19 symptoms [7]. To run it, please make the sure your symptom labels match the 11 CDC COVID-19 symptoms labels present in GAZ_group.csv.

Docker usage:
`docker run -it  -v /var/lib/docker/data/vte/icd_negative/:/data ahc-nlpie-docker.artifactory.umn.edu/gazetteer:1  python -u  /home/gazetteer/gazetteer_multiprocess_sbd.py final_lex.csv manifest_negative.csv data_in/ negatives vte`

References:

1. spaCy Matcher: https://spacy.io/api/matcher

2. spaCy EntityRuler: https://spacy.io/api/entityruler

3. spaCy models: https://spacy.io/usage/models

4. negspaCy: https://spacy.io/universe/project/negspacy

5. spaCy Rule-Based Matching: https://spacy.io/usage/rule-based-matching

6. spaCy Token: https://spacy.io/api/token

7. Symptoms of Coronavirus: https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html
