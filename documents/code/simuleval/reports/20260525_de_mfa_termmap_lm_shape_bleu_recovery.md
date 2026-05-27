# En-De MFA Term-Map Shape Analysis

This diagnostic aligns HN1024 tau=0.78 runtime term maps to MFA-timed ACL sentence intervals. It is a post-hoc failure analysis, not a tau/model selection step.

## Metric And Shape Summary

| run | lm | BLEU | TERM_ACC | TERM_FCR | nonempty call rate | avg terms/call | marginal 0.78-0.80 | stale lookback refs | sent gold recall | sent noise ref rate | NONE prompts | omitted empty prompts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| newv9_none_block | 1 | 26.25 | 0.8428 | 0.2157 | 0.5190 | 0.9342 | 0.1366 | 0.0000 | 0.9166 | 0.4720 | 3588 | 0 |
| newv9_none_block | 2 | 29.93 | 0.8599 | 0.2016 | 0.6719 | 1.4691 | 0.1327 | 0.0000 | 0.9176 | 0.4982 | 1785 | 0 |
| newv9_none_block | 3 | 30.81 | 0.8556 | 0.1645 | 0.7673 | 1.9591 | 0.1209 | 0.0000 | 0.9248 | 0.5336 | 1174 | 0 |
| newv9_none_block | 4 | 31.70 | 0.8492 | 0.1461 | 0.8198 | 2.4583 | 0.1267 | 0.0000 | 0.9269 | 0.5632 | 803 | 0 |
| tmv4_omit | 2 | 31.03 | 0.8235 | 0.1199 | 0.6719 | 1.4691 | 0.1327 | 0.0000 | 0.9176 | 0.4982 | 0 | 10 |
| tmv4_omit | 4 | 32.53 | 0.8406 | 0.1108 | 0.8198 | 2.4583 | 0.1267 | 0.0000 | 0.9269 | 0.5632 | 0 | 3 |

## Runtime Reference Identity

- lm=2: NewV9 and TMV4-omit post-tau references identical=True
- lm=4: NewV9 and TMV4-omit post-tau references identical=True

## Verified No-RAG Comparison Points

- lm=2: BLEU=30.0676, TERM_ACC=0.6364
- lm=4: BLEU=33.3008, TERM_ACC=0.6909

## Interpretation

- The post-tau reference lists are identical for NewV9 and TM-SFT at the same lm; BLEU differences are therefore SLM/prompt-response differences, not retriever differences.
- `empty_term_map_policy=omit` removes `term_map:NONE` prompts for empty retrievals, but lm=4 TM-SFT still remains below the verified no-RAG BLEU target, so NONE blocks are not the only issue.
- A large share of sentence-aligned runtime references are not source-supported in the overlapping ACL sentence, which means the SLM must be trained to ignore plausible but locally unsupported terminology, not only to adopt retrieved terms.
- MFA timestamps show no post-tau references outside the current runtime window in these logs (`stale_lookback_ref_rate=0`), so the immediate issue is not stale lookback leakage.
- The risky shape is local over-exposure: many sentence-aligned references are unsupported by the overlapping source sentence, and roughly 12-14% of references sit just above tau in the 0.78-0.80 band.

## Recommended SLM Adjustment

Use the MFA distribution to build a short rescue SFT variant rather than another tagged-only variant:

1. Keep `empty_term_map_policy=omit` at inference and make training match it: no `term_map:NONE` user blocks for empty maps.
2. For no-GT chunks, do not zero the map; inject HN1024-style retrieved maps, but bucket them by runtime shape: empty, 1-2 terms, 3-5 terms, 6+ terms.
3. Add negative/noise exposure targets: if a retrieved term translation is not in the future assistant span up to the message end, leave the assistant unwrapped and do not force adoption.
4. Down-weight or dropout the riskiest references during SFT data construction: score 0.78-0.80 and sentence-unsupported terms. This changes SLM behavior without ACL tau tuning.
5. Gate the next training candidate on de/lm=4 BLEU against verified no-RAG, but select the data rule from train/dev distribution, not ACL.

## Case Pointers

- Full sentence table: `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_sentences.tsv`
- Call-level shape table: `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_calls.tsv`
- Summary table: `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_summary.tsv`

### TM-SFT lm=4 Dense/False-Copy Examples

#### 2022.acl-long.268 sent=18 126.385-136.232s

- chrF=95.161; map_count=16; noise=9; false_copy=NLP task->NLP-Aufgabe@0.808
- term_map: source->Quelle@0.951; words->Wörtern@0.942; vocabulary->Wortschatz@0.871; word->Wort@0.826; Detecting->Erkennung@0.884; lexical->lexikalische@0.831; task->Aufgabe@0.943; parsing->Parsing@0.942; NLP->NLP@0.905; tasks->Aufgaben@0.897; NLP task->NLP-Aufgabe@0.808; machine translation->maschinelle Übersetzung@0.917; machine learning->maschinelles Lernen@0.820; translation->Übersetzung@0.807; generalization->Verallgemeinerung@0.790; question generation->Fragengenerierung@0.784
- source: And in fact, automatically detecting lexical borrowings ah has proven to be useful for NLP downstream tasks such as parsing, text-to-speech synthesis or machine translation.
- reference: Die automatische Erkennung lexikalischer Entlehnungen erwies sich als nützlich für NLP und nachgelagerte Aufgaben wie Parsing, Text-zu-Sprache-Synthesen oder die maschinelle Übersetzung.
- hypothesis: die automatische Erkennung lexikalischer Entlehnungen sich als nützlich für nachgeschaltete NLP-Aufgaben erwiesen, wie beispielsweise das Parsing von Text, die Textzusammenfassung oder die maschinelle Übersetzung.Es

#### 2022.acl-long.117 sent=439 488.768-496.884s

- chrF=85.792; map_count=14; noise=8; false_copy=domains->Domänen@0.810
- term_map: datasets->Datensätzen@0.935; question->Frage@0.930; context->Kontext@0.923; domain->Domäne@0.898; contexts->Kontexte@0.888; dataset->Datensatz@0.852; questions->Fragen@0.824; domains->Domänen@0.810; data->Daten@0.954; training data->Trainingsdaten@0.942; training->Trainingssatz@0.836; augmentation->Aufbaus@0.850; data augmentation->Datenaufbaus@0.828; annotation->Annotation@0.793
- source: We experiment with six out of domain datasets and present results here, where data is the training data is doubled in augmentation.
- reference: Wir experimentieren mit sechs von den Datensätzen der Domäne und präsentieren hier die Ergebnisse, wobei es bei den Daten um die Trainingsdaten geht und beim Aufbau verdoppelt werden.
- hypothesis: experimentieren mit sechs außerhalb des Domänenbereichs liegenden Datensätzen und präsentieren hier die Ergebnisse, wobei die Trainingsdaten verdoppelt wurden.Dabei

#### 2022.acl-long.117 sent=445 541.444-549.217s

- chrF=94.479; map_count=13; noise=8; false_copy=domains->Domänen@0.784
- term_map: model->Modell@0.959; QA->QA@0.820; Modeling->Modellierung@0.807; domain->Domäne@0.904; question->Frage@0.949; datasets->Datensätzen@0.950; dataset->Datensatz@0.876; questions->Fragen@0.851; evaluation->evaluation@0.826; data->Daten@0.823; domains->Domänen@0.784; models->Modelle@0.937; baselines->Baselines@0.840
- source: We also experiment with open domain QA setting where the model only sees the question and once again we evaluate on four out of domain datasets.
- reference: Wir experimentieren auch mit einer offenen Domäne-QA -Einstellung, bei der das Modell nur die Frage sieht. Wir bewerten wieder vier von den Datensätzen der Domäne.
- hypothesis: Wir experimentieren außerdem mit einem offenen Domänen-QA-Setting, bei dem das Modell lediglich die Frage sieht,und bewerten erneut auf vier außerhalb des Domänenbereichs liegenden Datensätzen.

#### 2022.acl-long.117 sent=465 714.491-718.376s

- chrF=94.845; map_count=11; noise=8; false_copy=domains->Domänen@0.822
- term_map: generalization->Verallgemeinerung@0.937; domain->Domäne@0.900; augmentation->Aufbaus@0.865; domains->Domänen@0.822; generalize->verallgemeinert@0.810; annotation->Annotation@0.798; data augmentation->Datenaufbaus@0.781; semantically->semantisch@0.889; RGF->RGF@0.878; Semantic->Semantic@0.802; counterfactual->kontrafaktisch@0.788
- source: Augmentation improves out of domain generalization and neighborhood consistency.
- reference: Der Aufbau verbessert sich dank der Domäneverallgemeinerung und der Konsistenz der Nachbarschaft.
- hypothesis: Aufbaus verbessern die Verallgemeinerung außerhalb des Domänenbereichs und die Nachbarschaftskonsistenz,und wir stellen

#### 2022.acl-long.110 sent=307 323.792-329.177s

- chrF=96.522; map_count=14; noise=7; false_copy=utterances->Äußerungen@0.859
- term_map: approach->Ansatz@0.953; propose->vorzuschlagen@0.939; method->Methode@0.937; Language Model->Sprachmodell@0.907; masked language model->maskiertes Sprachmodell@0.802; methods->Methoden@0.788; language->Sprache@0.785; language models->Sprachmodelle@0.784; model->Modell@0.783; parsing->Parsing@0.889; utterance->Äußerung@0.860; utterances->Äußerungen@0.859; graph->Graph@0.806; parser->Parser@0.804
- source: First approach combines a language model completion with full utterance to graph parsing.
- reference: Der erste Ansatz kombiniert eine Sprachmodell-Vervollständigung mit einer vollständigen Äußerung zum Graph-Parsing.
- hypothesis: Der erste Ansatz kombiniert eine Sprachmodell-Vervollständigung mit vollständigen Äußerungen zur Graph-Parsing.

#### 2022.acl-long.117 sent=438 480.485-488.607s

- chrF=93.846; map_count=14; noise=7; false_copy=questions->Fragen@0.954
- term_map: baselines->Baselines@0.962; RGF->RGF@0.877; augmentation->Aufbaus@0.815; model->Modell@0.965; questions->Fragen@0.954; reading comprehension->Leseverständnis@0.948; question->Frage@0.930; datasets->Datensätzen@0.935; context->Kontext@0.923; domain->Domäne@0.898; contexts->Kontexte@0.888; dataset->Datensatz@0.852; domains->Domänen@0.810; data->Daten@0.797
- source: How base how do the baselines and RGF ah augmentation perform on reading comprehension where the model has access to question and context?
- reference: Welche Leistung erbringen die Baselines, RGF und der Aufbau beim Leseverständnis, wo das Modell Zugriff auf Frage und Kontext hat?
- hypothesis: Wie schneiden die Baselines und die RGF-Aufbaus bei der Leseverständnis-Aufgabe ab, bei der das Modell Zugriff auf Fragen und Kontext hat?Wir

#### 2022.acl-long.117 sent=398 222.569-229.408s

- chrF=87.742; map_count=11; noise=7; false_copy=generated->generiert@0.837
- term_map: answer->beantwortet@0.952; contexts->Kontexte@0.944; context->Kontext@0.918; question generation->Fragengenerierung@0.907; generation->Generierung@0.900; model->Modell@0.876; question->Frage@0.946; generated->generiert@0.837; questions->Fragen@0.829; generating->Generierung@0.813; question answering->Fragenbeantwortung@0.806
- source: Following this step, the question generation model conditions on these alternate answers to generate a question that corresponds to them.
- reference: Im Anschluss an diesen Schritt konditioniert dieses Fragengenerierungsmodell diese alternativen Antworten, um eine ihnen entsprechende Frage zu generieren.
- hypothesis: Anschließend bedingt das Fragengenerierungsmodell die generierten Fragen auf diesen alternativen Antworten, um eine Frage zu erzeugen, die zu ihnen passt.

#### 2022.acl-long.117 sent=459 651.384-661.014s

- chrF=84.146; map_count=10; noise=7; false_copy=counterfactual->kontrafaktisch@0.825
- term_map: qualitative->qualitative@0.907; evaluation->evaluation@0.907; generated->generiert@0.914; counterfactual->kontrafaktisch@0.825; generating->Generierung@0.826; questions->Fragen@0.949; question->Frage@0.961; generalize->verallgemeinert@0.789; generation->Generierung@0.789; question generation->Fragengenerierung@0.785
- source: In fact, a qualitative inspection of the kinds of counterfactuals generated show that the generated questions contain several diverse perturbations.
- reference: Tatsächlich zeigt eine qualitative Überprüfung der verschiedenen Arten von Kontrafaktoren, dass die generierten Fragen mehrere unterschiedliche Störungen enthalten.
- hypothesis: qualitative Untersuchung der generierten kontrafaktischen Beispiele zeigt, dass die generierten Fragen mehrere verschiedene Störungen enthalten.Zum

#### 2022.acl-long.110 sent=331 462.115-469.867s

- chrF=97.710; map_count=10; noise=6; false_copy=model->Modell@0.906; Online->Online@0.823
- term_map: parser->Parser@0.960; graph->Graph@0.909; graphs->Graphen@0.835; TreeDST->TreeDST@0.789; parsing->Parsing@0.857; Online->Online@0.823; datasets->Datensätzen@0.967; model->Modell@0.906; dataset->Datensatz@0.903; data->Daten@0.833
- source: Our graph based parser when operating offline, achieves state-of-the-art performance on parsing on both datasets.
- reference: Unser auf dem Graphen basierte Parser erreicht, wenn er offline betrieben wird, beste Leistungen beim Parsing für beide Datensätze.
- hypothesis: graphbasiertes Parser-System, wenn es offline betrieben wird,erreicht bei der Parsing-Aufgabe einen State-of-the-art-Leistungswert. Auf beiden Datensätzen. Das Online-Modell mit vollständiger

#### 2022.acl-long.117 sent=384 121.944-126.834s

- chrF=73.913; map_count=9; noise=6; false_copy=counterfactual->kontrafaktisch@0.799; questions->Fragen@0.821
- term_map: question answering->Fragenbeantwortung@0.927; question->Frage@0.894; generating->Generierung@0.850; generated->generiert@0.845; questions->Fragen@0.821; counterfactual->kontrafaktisch@0.799; task->Aufgabe@0.941; knowledge->Wissen@0.939; tasks->Aufgaben@0.914
- source: There are more challenges to generating counterfactuals for question answering specifically.
- reference: Es gibt mehr Herausforderungen für die Generierung von Kontrafakten als für die spezifische Beantwortung der Frage.
- hypothesis: Herausforderungen bei der Generierung von kontrafaktischen Aussagen für Fragenbeantwortung.
