# DE lm=2 RASST BLEU Case Analysis

## Global evidence

- NewV9 RASST: BLEU=29.926444569587574 TERM_ACC=0.8599 REAL_TERM_ADOPT=0.877021 TERM_FCR=0.201635
- TM-SFT + HN1024: BLEU=31.121069182103 TERM_ACC=0.8353 REAL_TERM_ADOPT=0.853383 TERM_FCR=0.147139
- no-RAG verified: BLEU=30.06764803241279 TERM_ACC=0.6364
- Runtime term maps are identical between NewV9 and TM-SFT + HN1024: True.
- NewV9 raw output tag counts: <term>=877 </term>=874; TM-SFT=0; no-RAG=0.
- Sentence rows written to `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/reports/20260525_de_lm2_rasst_bleu_case_report.tsv`.

## Distribution

- Sentences: 468; with aligned term_map: 443; NewV9 worse than TM-SFT by chrF>5: 85; among term_map sentences: 78.
- NewV9 false-copied at least one exposed non-gold term in 74 sentence rows.
- NewV9 emitted `<term>` in chunks overlapping 402 sentence rows.

## Top NewV9-worse Cases With Term Maps

### 2022.acl-long.110 sent=304 306.539-309.766s

- chrF NewV9/TM-SFT/no-RAG: 30.645 / 83.871 / 80.645 (TM-New=53.226)
- term_map: approach->Ansatz; propose->vorzuschlagen; token->Token; tokens->Token
- gold terms: approach->Ansatz
- NewV9 false copy from term_map: token->Token; tokens->Token
- NewV9 overlapping `<term>` chunks: 2
- source: In particular, we propose a two step approach.
- reference: Insbesondere wollen wir einen zweistufigen Ansatz vorschlagen.
- NewV9: jedem Token.Insbesondere,
- TM-SFT+HN1024: Insbesondere schlagen wir eine zweistufige Methode vor.Ein
- no-RAG: Insbesondere schlagen wir eine zweistufige Methode vor,

### 2022.acl-long.367 sent=179 609.285-613.255s

- chrF NewV9/TM-SFT/no-RAG: 37.647 / 78.824 / 78.824 (TM-New=41.176)
- term_map: context->Kontext; contexts->Kontexte; previous->früheren; task->Aufgabe; tasks->Aufgaben; texts->Text
- gold terms: news->Nachrichten; task->Aufgabe
- NewV9 false copy from term_map: NONE
- NewV9 overlapping `<term>` chunks: 2
- source: For the news categorization task, we find mixed results.
- reference: Bei der Aufgabe zur Kategorisierung der Nachrichten bekamen wir gemischte Ergebnisse.
- NewV9: finden wir gemischte Ergebnisse.
- TM-SFT+HN1024: Aufgabe der Nachrichtenkategorisierungfinden wir gemischte Ergebnisse,
- no-RAG: Aufgabe der Nachrichtenkategorisierungfinden wir gemischte Ergebnisse,

### 2022.acl-long.117 sent=385 127.055-129.237s

- chrF NewV9/TM-SFT/no-RAG: 66.667 / 100.000 / 100.000 (TM-New=33.333)
- term_map: knowledge->Wissen; question answering->Fragenbeantwortung; task->Aufgabe; tasks->Aufgaben
- gold terms: task->Aufgabe
- NewV9 false copy from term_map: NONE
- NewV9 overlapping `<term>` chunks: 1
- source: This task requires background knowledge.
- reference: Diese Aufgabe erfordert Hintergrundwissen.
- NewV9: Diese Aufgabe erfordert eine
- TM-SFT+HN1024: Diese Aufgabe erfordert Hintergrundwissen.
- no-RAG: Diese Aufgabe erfordert Hintergrundwissen.

### 2022.acl-long.110 sent=280 161.580-165.690s

- chrF NewV9/TM-SFT/no-RAG: 72.152 / 100.000 / 100.000 (TM-New=27.848)
- term_map: information->Informationen; user->Benutzer; utterance->Äußerung; utterances->Äußerungen
- gold terms: user->Benutzer
- NewV9 false copy from term_map: NONE
- NewV9 overlapping `<term>` chunks: 1
- source: This process goes on until we receive the full user utterance.
- reference: Dieser Prozess geht weiter, bis wir die vollständige Benutzeräußerung erhalten.
- NewV9: , bis wir die vollständige Äußerung des Benutzers erhalten.
- TM-SFT+HN1024: Dieser Prozess geht weiter, bis wir die vollst ändige Benutzeräußerung erhalten.
- no-RAG: Dieser Prozess geht weiter, bis wir die vollständige Benutzeräußerung erhalten.

### 2022.acl-long.110 sent=303 303.493-306.096s

- chrF NewV9/TM-SFT/no-RAG: 58.824 / 86.275 / 76.471 (TM-New=27.451)
- term_map: token->Token; tokens->Token
- gold terms: token->Token
- NewV9 false copy from term_map: NONE
- NewV9 overlapping `<term>` chunks: 1
- source: So, we reparse from scratch after each token.
- reference: Also parsen wir nach jedem Token von Grund auf neu.
- NewV9: Daher parsen wir von Grund auf, bei
- TM-SFT+HN1024: wahren. Daher parsen wir von Grund auf bei jedem Token.
- no-RAG: wahren. Daher analysieren wir von jedem Token neu.

### 2022.acl-long.367 sent=124 151.020-159.696s

- chrF NewV9/TM-SFT/no-RAG: 63.704 / 90.370 / 80.000 (TM-New=26.667)
- term_map: document->Dokument; sentence->Satz
- gold terms: NONE
- NewV9 false copy from term_map: document->Dokument
- NewV9 overlapping `<term>` chunks: 2
- source: For example here, we have John twarahamubonye biradutangaza, which means we were surprised to find John there.
- reference: Zum Beispiel haben wir hier den Satz: „John twarahamubonye biradutangaza“. Das bedeutet: „Wir waren überrascht, John dort anzutreffen.“
- NewV9: Beispielsweise haben wir hier ein Dokument, das bedeutet, dass wir überrascht waren, John hier zu
- TM-SFT+HN1024: Zum Beispiel haben wir hier den Satz „John tuwara hamwe n'ibiro du tanga za“, was bedeutet :Wir waren überrascht, John dort zu finden.
- no-RAG: Zum Beispiel haben wir hier John Twara hamubwo n'ibira du tangaza, was bedeutet : Wir waren überrascht, John dort zu

### 2022.acl-long.268 sent=37 259.205-264.737s

- chrF NewV9/TM-SFT/no-RAG: 69.811 / 96.226 / 96.226 (TM-New=26.415)
- term_map: training->Trainingssatz; word->Wort; words->Wörtern
- gold terms: training->Trainingssatz; words->Wörtern
- NewV9 false copy from term_map: NONE
- NewV9 overlapping `<term>` chunks: 1
- source: So there would be minimal overlap in words and topics between the training set and test set.
- reference: Es gäbe also minimale Überschneidungen bei Wörtern und Themen zwischen dem Trainingssatz und dem Testsatz.
- NewV9: wenig Überschneidung zwischen Wörtern und Themen geben sollte. Daraus ergab sich,
- TM-SFT+HN1024: der so schwierig wie möglich war, sodass es ein Minimum an Überschneidung in Wörtern und Themen geben würde. Ah zwischen dem Trainingssatz und dem Testdatensatz.
- no-RAG: der so schwierig wie möglich war, damit es eine minimale Überlappung an Wörtern und Themen gäbe. Ah zwischen dem Trainings- und dem Testdatensatz.

### 2022.acl-long.117 sent=408 290.446-297.445s

- chrF NewV9/TM-SFT/no-RAG: 73.510 / 95.364 / 92.053 (TM-New=21.854)
- term_map: answer->beantwortet; context->Kontext; contexts->Kontexte; generation->Generierung; inference->Inferenz; model->Modell; previous->früheren; question->Frage; question generation->Fragengenerierung; retrieved->abgerufen
- gold terms: context->Kontext; previous->früheren; question->Frage; question generation->Fragengenerierung; retrieved->abgerufen
- NewV9 false copy from term_map: NONE
- NewV9 overlapping `<term>` chunks: 4
- source: During inference we supply the question generation model, the alternative answer and context that we retrieved in the previous step.
- reference: Während der Interferenz liefern wir das Fragengenerierungsmodell, die alternative Antwort und den Kontext, die wir im früheren Schritt abgerufen haben.
- NewV9: Während der Inferenz liefern wir dem Fragengenerierung-Modell die alternative Antwort und den Kontext. Beispielsweise
- TM-SFT+HN1024: Während der Inferenz wenden wir das Fragengenerierungsmodell auf die alternativen Antwort und den Kontexten an, die wir im vorherigen Schritt abgerufen haben.
- no-RAG: Während der Inferenz geben wir dem Frageerstellungsmodell die alternative Antwort und den Kontext, den wir im vorherigen Schritt abgerufen haben.


## Top NewV9 False-Copy Cases

### 2022.acl-long.110 sent=304 306.539-309.766s

- chrF NewV9/TM-SFT/no-RAG: 30.645 / 83.871 / 80.645 (TM-New=53.226)
- term_map: approach->Ansatz; propose->vorzuschlagen; token->Token; tokens->Token
- gold terms: approach->Ansatz
- NewV9 false copy from term_map: token->Token; tokens->Token
- NewV9 overlapping `<term>` chunks: 2
- source: In particular, we propose a two step approach.
- reference: Insbesondere wollen wir einen zweistufigen Ansatz vorschlagen.
- NewV9: jedem Token.Insbesondere,
- TM-SFT+HN1024: Insbesondere schlagen wir eine zweistufige Methode vor.Ein
- no-RAG: Insbesondere schlagen wir eine zweistufige Methode vor,

### 2022.acl-long.367 sent=124 151.020-159.696s

- chrF NewV9/TM-SFT/no-RAG: 63.704 / 90.370 / 80.000 (TM-New=26.667)
- term_map: document->Dokument; sentence->Satz
- gold terms: NONE
- NewV9 false copy from term_map: document->Dokument
- NewV9 overlapping `<term>` chunks: 2
- source: For example here, we have John twarahamubonye biradutangaza, which means we were surprised to find John there.
- reference: Zum Beispiel haben wir hier den Satz: „John twarahamubonye biradutangaza“. Das bedeutet: „Wir waren überrascht, John dort anzutreffen.“
- NewV9: Beispielsweise haben wir hier ein Dokument, das bedeutet, dass wir überrascht waren, John hier zu
- TM-SFT+HN1024: Zum Beispiel haben wir hier den Satz „John tuwara hamwe n'ibiro du tanga za“, was bedeutet :Wir waren überrascht, John dort zu finden.
- no-RAG: Zum Beispiel haben wir hier John Twara hamubwo n'ibira du tangaza, was bedeutet : Wir waren überrascht, John dort zu

### 2022.acl-long.367 sent=171 548.672-554.866s

- chrF NewV9/TM-SFT/no-RAG: 70.435 / 89.565 / 76.522 (TM-New=19.130)
- term_map: baselines->Baselines; kinyabert->KinyaBERT; models->Modelle
- gold terms: GLUE->GLUE; KinyaBERT->KinyaBERT; models->Modelle
- NewV9 false copy from term_map: baselines->Baselines
- NewV9 overlapping `<term>` chunks: 3
- source: For the GLUE benchmark, we find that KinyaBERT consistently outperforms baseline models.
- reference: Bei der GLUE-Benchmark haben wir festgestellt, dass KinyaBERT durchweg besser abschneidet als die Baseline-Modelle.
- NewV9: die GLUE-Benchmark, finden wir, dass KinyaBERT die Baselines hier konsistent übertroffen hat,
- TM-SFT+HN1024: nun gehen wir zu den Ergebnissen für die GLUE-Benchmark. Wir stellen fest, dass KinyaBERT konsistent die Baseline-Modelle übertrifft.
- no-RAG: wir zur Ergebnisse. Für die GLUE-Benchmark, finden wir, dass Kinyabert konsistent die Basismodelle übertrifft.

### 2022.acl-long.117 sent=438 480.485-488.607s

- chrF NewV9/TM-SFT/no-RAG: 83.077 / 95.385 / 93.077 (TM-New=12.308)
- term_map: augmentation->Aufbaus; baselines->Baselines; context->Kontext; contexts->Kontexte; model->Modell; question->Frage; questions->Fragen; reading comprehension->Leseverständnis; rgf->RGF
- gold terms: baselines->Baselines; context->Kontext; model->Modell; question->Frage; reading comprehension->Leseverständnis; RGF->RGF
- NewV9 false copy from term_map: questions->Fragen
- NewV9 overlapping `<term>` chunks: 6
- source: How base how do the baselines and RGF ah augmentation perform on reading comprehension where the model has access to question and context?
- reference: Welche Leistung erbringen die Baselines, RGF und der Aufbau beim Leseverständnis, wo das Modell Zugriff auf Frage und Kontext hat?
- NewV9: Wie groß sind die Baselines und RGF-Aufbaus bei der Lesekompetenz? Wo das Modell Zugriff auf Fragen und Kontext hat,
- TM-SFT+HN1024: Wie groß sind die Baselines und wie performen die RGF-Aufbaus bei der Leseverst ändniss-Aufgaben, bei denen das Modell Zugriff auf Frage und Kontext hat?Wir
- no-RAG: Wie groß sind die Baselines und die semantische Perturbation von RGF bei der Leseverständnis, bei der das Modell Zugriff auf Frage und Kontext hat?Wir

### 2022.acl-long.367 sent=143 302.092-305.776s

- chrF NewV9/TM-SFT/no-RAG: 77.632 / 89.474 / 84.211 (TM-New=11.842)
- term_map: algorithm->Algorithmus; algorithms->Algorithmen; speech->Sprache; tagging->Tagging; unsupervised->nicht überwachten; user->Benutzer
- gold terms: algorithm->Algorithmus; tagging->Tagging; unsupervised->nicht überwachten
- NewV9 false copy from term_map: algorithms->Algorithmen
- NewV9 overlapping `<term>` chunks: 2
- source: We use an unsupervised part of speech tagging algorithm.
- reference: Wir verwenden einen nicht überwachten Teil eines Sprach-Tagging-Algorithmus.
- NewV9: Wir verwenden eine nicht überwachten Tagging</ >-Algorithmen, um
- TM-SFT+HN1024: Wir verwenden einen nicht überwachten Wortart-Tagging -Algorithmus.Ein Faktorenmodell
- no-RAG: Wir verwenden eine unüberwachte Wortart-Tagging -Algorithmus, einen Faktor-Modell,

### 2022.acl-long.590 sent=232 354.169-367.613s

- chrF NewV9/TM-SFT/no-RAG: 84.492 / 95.722 / 94.652 (TM-New=11.230)
- term_map: attention->Aufmerksamkeit; decoder->Decoders; encoder->Encoder
- gold terms: input->Eingabe
- NewV9 false copy from term_map: encoder->Encoder
- NewV9 overlapping `<term>` chunks: 2
- source: This directly influenced the cause of cross attention, which depends not on the input length N, but the constant K, representing the pooled length.
- reference: Dies hatte einen direkten Einfluss auf die Ursache der Cross-Attention. Diese hängt nicht von der Länge der Eingabe N ab, sondern von der Konstante K, welche die gepoolte Länge darstellt.
- NewV9: Encoder eingeführt wird.Dies beeinflusst direkt die Kosten der Aufmerksamkeit, die nicht von der Eingabewortlänge abhängt und die konstante K. Darstellend die Pooling-Länge. Dies
- TM-SFT+HN1024: Encoder-Schicht eingeführt wird.Dies beeinflusst direkt die Kosten der Kreuz-Attention, die nicht von der Eingabewertlänge abhängt und stattdessen die konstante k. Die Pool-Länge darstellend. Dies führt dazu, dass informiert
- no-RAG: Encoderebene eingeführt wird.Dies beeinflusst direkt die Kosten der Kreuz-Attention, die nicht von der Eingabedauer abhängt und die konstante k. Dargestellt in der Pooling-Ebene.Dies führt dazu, dass die Anzahl der ausgewählten

### 2022.acl-long.268 sent=91 628.676-640.698s

- chrF NewV9/TM-SFT/no-RAG: 83.654 / 93.750 / 82.212 (TM-New=10.096)
- term_map: embedding->Einbettung; embeddings->Einbettungen; english->Englischen; finetuning->Feinabstimmung; language identification->Sprachidentifikation; lince->LinCE; pre-training->Vortrainings; pretraining->Vortraining; transformer->Transformer; transformer encoder->Transformer-Encoder; transformers->Transformer
- gold terms: dataset->Datensatz; embeddings->Einbettungen; language identification->Sprachidentifikation; LinCE->LinCE; pretrained->vortrainiert; transformer->Transformer
- NewV9 false copy from term_map: finetuning->Feinabstimmung; pretraining->Vortraining
- NewV9 overlapping `<term>` chunks: 9
- source: Well these are um embeddings that are have been fine tuned transformer-based embeddings that have been pretrained for language identification on the Spanish English section of the LinCE code switching dataset.
- reference: Dies sind Einbettungen, die auf Transformer-basierte Einbettungen abgestimmt wurden. Diese wurden für die Sprachidentifikation im Spanisch-Englisch-Abschnitt des LinCE-Code-Switching-Datensatzes vortrainiert.
- NewV9: Nun, das sind Einbettungen, die zur Hälfte aus Feinabstimmung stammen, Einbettungen basierend auf Transformer, die im Vortraining für die Sprachidentifikation im spanisch-Englischen Teil des LINT -Datensatzes
- TM-SFT+HN1024: Nun, das sind Einbettungen, die feinabgestimmt wurden, um Transformer-basierte Einbettungen zu verwenden, die im Vortraining für die Sprachidentifikation im spanisch-englischen Abschnitt des Code-Switching-Datensatzes
- no-RAG: Nun, das sind Embeddings, die darauf trainiert wurden, transformer-basierte Embeddings zu verfeinern, die bereits auf dem spanisch-englischen Abschnitt des Code-Switching-Datensatzes LINTA

### 2022.acl-long.110 sent=332 470.048-477.241s

- chrF NewV9/TM-SFT/no-RAG: 70.000 / 80.000 / 77.857 (TM-New=10.000)
- term_map: baselines->Baselines; data->Daten; dataset->Datensatz; datasets->Datensätzen; model->Modell; parser->Parser; parsing->Parsing
- gold terms: BLEU->BLEU; compared->im Vergleich zu; model->Modell
- NewV9 false copy from term_map: baselines->Baselines
- NewV9 overlapping `<term>` chunks: 4
- source: The LM complete model also achieves nontrivial BLEU gain compared with the simple baseline of node completion.
- reference: Das LM-Complete-Modell erzielt auch eine nicht triviale BLEU -Verstärkung im Vergleich zur einfachen Basislinie der Knotenvervollständigung.
- NewV9: Online-Modell ebenfalls beachtliche Ergebnisse. Im Vergleich zur einfachen Baselines ohne Vervollständigung.
- TM-SFT+HN1024: Das Online-Modell erreicht ebenfalls eine signifikante Verbesserung im Vergleich zur einfachen Baseline ohne Vervollständigung.
- no-RAG: Das Online-Modell erreicht außerdem eine nennenswerte Verbesserung im Vergleich zur einfachen Baseline ohne Vervollständigung.
