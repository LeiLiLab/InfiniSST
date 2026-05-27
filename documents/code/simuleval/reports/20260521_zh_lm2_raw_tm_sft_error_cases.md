# zh lm2 raw: no-TM-SFT vs RASST vs V3-real term adoption cases

Compared files:

- `origin`: `/mnt/gemini/data2/jiaxuanluo/tagged_acl_origin_bsz4_tau073_baseline_20260520T1010_ragreset_full_mp3/full/zh/dtagacl_origin_bsz4_tau073_lm2_k10_th0.73_gacl6060_tagged_gt_raw_min_norm2/term_adoption.json`
- `rasst`: `/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_llmgen_sft_tau073_full_20260521T050150/zh/dtagacl_llmgen_sft_tau073_lm2_k10_th0.73_gacl6060_tagged_gt_raw_min_norm2/term_adoption.json`
- `v3_real`: `/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v3_speech_llm_3x3_20260521T024609/real/zh/dtagacl_v3_real_zh_lm2_raw_tau073_lm2_k10_th0.73_gacl6060_tagged_gt_raw_min_norm2/term_adoption.json`

## Aggregate term-level diff

- Shared evaluated terms: `1173`
- no-TM-SFT correct: `1033`
- RASST correct: `1023`
- V3-real correct: `932`
- no-TM-SFT correct but RASST missed: `55`
- no-TM-SFT correct but V3-real missed: `138`
- RASST correct but no-TM-SFT missed: `45`
- V3-real correct but no-TM-SFT missed: `37`

## Representative cases

### sentence 62

- Rates: no-TM-SFT `1.0`, RASST `0.5`, V3-real `0.25`
- no-TM-SFT correct but RASST missed: token -> 令牌, word -> 单词
- no-TM-SFT correct but V3-real missed: binary -> 二进制, token -> 令牌, word -> 单词
- Both RASST and V3-real missed while no-TM-SFT got it: token -> 令牌, word -> 单词
- Source: These are binary features such as the word or the token in upper case?
- Reference: 这些都是二进制特征，例如大写的单词或令牌？
- no-TM-SFT hyp: 二进制特征，例如单词或令牌
- RASST hyp: 这些是二进制特征，例如该词或标记
- V3-real hyp: 二元特征，例如该词或词元
- aligned term_map: binary->二进制, features->特征, token->令牌, tokens->令牌, word->单词, words->单词

### sentence 22

- Rates: no-TM-SFT `1.0`, RASST `0.5`, V3-real `0.75`
- no-TM-SFT correct but RASST missed: extracting -> 提取, words -> 单词
- no-TM-SFT correct but V3-real missed: words -> 单词
- Both RASST and V3-real missed while no-TM-SFT got it: words -> 单词
- Source: Which means that we are interested in extracting ah words borrowed from other languages that are being used in Spanish newspapers but that have not been integrated or assimilated into the recipient language.
- Reference: 也就是说，我们感兴趣的是提取从其他语言中借用的单词，这些单词正在西班牙语报纸中使用，但尚未融入或同化到接收者的语言中。
- no-TM-SFT hyp: 这意味着我们关注从西班牙语新闻专线中提取来自西班牙语报纸中使用的其他语言的单词，但这些单词尚未被整合或尚未被吸收进目标语言中。
- RASST hyp: 也就是说，我们关注的是从西班牙报纸中借用但尚未被吸收或同化的其他语言词汇
- V3-real hyp: 这意味着我们关注的是提取来自其他语言的词汇借用，这些词汇正在被西班牙语报纸使用，但尚未被吸收或同源化进目标语言中。
- aligned term_map: extracting->提取, language->语言, language model->语言模型, language models->语言模型, languages->语言, newswire->新闻专线, word->单词, words->单词

### sentence 189

- Rates: no-TM-SFT `0.75`, RASST `0.25`, V3-real `0.75`
- no-TM-SFT correct but RASST missed: models -> 模型, transformer -> 转换器
- no-TM-SFT correct but V3-real missed: transformer -> 转换器
- Both RASST and V3-real missed while no-TM-SFT got it: transformer -> 转换器
- Source: Hello, my name is Michał Pietruszka and it is my pleasure to present to you the paper titled Sparsifying Transformer Models with Trainable Representation Pooling.
- Reference: 大家好，我叫Michał Pietruszka，我很高兴向大家介绍这篇题为《用可训练的表示池来对转换器模型进行稀疏化》的论文。
- no-TM-SFT hyp: 大家好，我叫米哈·彼得鲁什卡，这是我荣幸地向大家介绍一篇题为“稀疏化转换器模型，使用可训练的表示池
- RASST hyp: 大家好，我叫米哈·彼得鲁什卡，这是我荣幸地向大家介绍一篇题为《稀疏化可训练表示池化方法的Tr
- V3-real hyp: 大家好，我叫米哈·彼得鲁什卡，这是我荣幸地向大家介绍一篇题为《指定可训练表示池化方法的Transformer模型》的论文。
- aligned term_map: model->模型, models->模型, paper->论文, representation->表示, representations->表示, transformer->转换器, transformer encoder->转换器编码器, transformers->转换器

### sentence 80

- Rates: no-TM-SFT `1.0`, RASST `0.875`, V3-real `0.5`
- no-TM-SFT correct but RASST missed: contextualized -> 情境化
- no-TM-SFT correct but V3-real missed: contextualized -> 情境化, multilingual -> 多语言, multilingual bert -> 多语言BERT, transformer -> 转换器
- Both RASST and V3-real missed while no-TM-SFT got it: contextualized -> 情境化
- Source: What we found out was that transformer-based embeddings performed better than non contextualized embeddings, that the combination of English BERT and Spanish BETO embeddings outperform multilingual BERT embeddings.
- Reference: 我们发现，基于转换器的嵌入比非情境化的嵌入表现得更好，英语BERT和西班牙语BETO嵌入的组合比多语言BERT嵌入表现得更好。
- no-TM-SFT hyp: 我们发现，基于转换器的嵌入表现优于非情境化嵌入。此外，英语嵌入与西班牙语BETO嵌入的组合表现优于多语言BERT嵌入。
- RASST hyp: 我们发现，基于转换器的嵌入表现优于非上下文感知的嵌入。英语BERT和西班牙BETO嵌入的组合优于多语言BERT嵌入。
- V3-real hyp: 我们发现，基于Transformer的嵌入向量表现优于非上下文感知的嵌入向量。英语BERT和西班牙语BETO嵌入向量的组合表现优于多语BERT嵌入向量。
- aligned term_map: bert->BERT, beto->BETO, binary->二进制, bpe->BPE, context->上下文, contexts->上下文, contextualized->情境化, embedding->嵌入, embeddings->嵌入, english->英语, modeling->建模, multilingual->多语言 ...

### sentence 197

- Rates: no-TM-SFT `1.0`, RASST `0.75`, V3-real `0.5`
- no-TM-SFT correct but RASST missed: transformer -> 转换器
- no-TM-SFT correct but V3-real missed: token -> 令牌, transformer -> 转换器
- Both RASST and V3-real missed while no-TM-SFT got it: transformer -> 转换器
- Source: In the vanilla transformer, with full attention connectivity, relations of each token to every other token have to be calculated.
- Reference: 在普通的转换器中，在完全注意力连通的情况下，必须计算每个令牌与其他令牌之间的关系。
- no-TM-SFT hyp: 在标准的转换器中，所有位置之间都存在完整的注意力连接，每个令牌都与所有其他令牌建立关系。这些关系需要进行计算。
- RASST hyp: 在原始的Transformer中，全连接注意力机制会计算每个令牌与所有其他令牌之间的关系。
- V3-real hyp: 在原始的Transformer中，通过全注意力连接，每个词元与所有其他词元之间的关系必须进行计算。
- aligned term_map: attention->注意力, qualitative->定性, relation->关系, relations->关系, token->令牌, tokens->令牌, transfer learning->迁移学习, transformer->转换器, transformer encoder->转换器编码器, transformers->转换器

### sentence 277

- Rates: no-TM-SFT `1.0`, RASST `0.5`, V3-real `0.0`
- no-TM-SFT correct but RASST missed: token -> 令牌
- no-TM-SFT correct but V3-real missed: graph -> 图形, token -> 令牌
- Both RASST and V3-real missed while no-TM-SFT got it: token -> 令牌
- Source: For example, each time we see a new token, we predict a new graph.
- Reference: 例如，每当我们看到一个新的令牌，我们就会预测一个新的图形。
- no-TM-SFT hyp: 例如，每次看到一个新令牌时，我们都会预测一个新图形。
- RASST hyp: 例如，每次看到一个新标记时，我们都会预测一个新图形。
- V3-real hyp: 例如，每次看到一个新词元，我们都会预测一个新图。
- aligned term_map: graph->图形, graphs->图形, predicates->谓词, prediction->预测, token->令牌, tokens->令牌

### sentence 290

- Rates: no-TM-SFT `1.0`, RASST `0.5`, V3-real `0.0`
- no-TM-SFT correct but RASST missed: graph -> 图形
- no-TM-SFT correct but V3-real missed: graph -> 图形, tokens -> 令牌
- Both RASST and V3-real missed while no-TM-SFT got it: graph -> 图形
- Source: Then, with more tokens, we predict a totally new graph, but part of it has already being executed.
- Reference: 然后，使用更多的令牌，我们预测一个全新的图形，但其中一部分已经被执行。
- no-TM-SFT hyp: 点的执行。随着更多令牌的输入，我们预测出一个全新的图形，但其中一部分已经开始执行，
- RASST hyp: 好。随着更多令牌的输入，我们预测出一个全新的程序图，但其中一部分已经在执行中，
- V3-real hyp: 口已确定。随着更多标记的输入，我们预测出一个全新的程序图，但其中一部分已经开始执行，
- aligned term_map: graph->图形, graphs->图形, predicates->谓词, prediction->预测, token->令牌, tokens->令牌

### sentence 91

- Rates: no-TM-SFT `0.875`, RASST `0.75`, V3-real `0.75`
- no-TM-SFT correct but RASST missed: transformer -> 转换器
- no-TM-SFT correct but V3-real missed: transformer -> 转换器
- Both RASST and V3-real missed while no-TM-SFT got it: transformer -> 转换器
- Source: Well these are um embeddings that are have been fine tuned transformer-based embeddings that have been pretrained for language identification on the Spanish English section of the LinCE code switching dataset.
- Reference: 这些是经过微调的基于转换器的嵌入，这些嵌入已经在LinCE语码转换数据集的西班牙语英语部分经过预训练 用于 语言识别。
- no-TM-SFT hyp: 这些嵌入是经过微调的转换器嵌入，这些嵌入是在西班牙语-英语部分的LinguaCodeSwitching数据集上进行语言识别任务的预训练得到的。
- RASST hyp: 这些是经过微调的嵌入，即基于Transformer的嵌入，它们在西班牙语-英语部分的Lingua代码切换数据集上进行了语言识别的预训练。
- V3-real hyp: 量。那么，代码切换嵌入向量是什么？嗯，这些嵌入向量是经过微调的Transformer基础嵌入向量，它们在西班牙语-英语部分的代码切换数据集上进行了预训练，以进行语言识别。
- aligned term_map: embedding->嵌入, embeddings->嵌入, encode->编码, english->英语, finetuning->微调, language->语言, language identification->语言识别, lince->LinCE, pre-training->预训练, pretrained->预训练, pretraining->预训练, training->训练 ...

### sentence 89

- Rates: no-TM-SFT `1.0`, RASST `0.8571428571428571`, V3-real `0.8571428571428571`
- no-TM-SFT correct but RASST missed: transformer -> 转换器
- no-TM-SFT correct but V3-real missed: transformer -> 转换器
- Both RASST and V3-real missed while no-TM-SFT got it: transformer -> 转换器
- Source: So, we run the same BiLSTM-CRF model that we had run using flare, but instead of using these unadapted transformer-based BETO and BERT embeddings, we used code switch embeddings.
- Reference: 因此，我们运行了使用flare运行的相同的BiLSTM-CRF模型，但我们使用了语码转换嵌入，而不是使用这些未调整的基于转换器的BETO和BERT嵌入。
- no-TM-SFT hyp: 因此，我们运行了相同的BiLSTM-CRF模型，即我们之前使用的BiLSTM-CRF模型，以及BERT模型，但这次使用的是FLR而非FLAIR。我们不再使用这些未经调整的转换器编码器，而是直接使用BETO和BERT嵌入。
- RASST hyp: 尤其是在代码切换的情况下？于是我们运行了相同的BiLSTM-CRF模型，该模型使用了FLAIR，但不再使用这些未经调整的Transformer模型，如BETO和BERT嵌入，而是使用了代码切换嵌入。
- V3-real hyp: 特别是在代码切换的情况下？于是我们运行了相同的BiLSTM-CRF模型，该模型使用了FLAIR库，但不再使用这些未经调整的Transformer模型，如BETO和BERT嵌入向
- aligned term_map: annotated->注释, bert->BERT, bert model->BERT模型, beto->BETO, bilstm-crf->BiLSTM-CRF, bio->BIO, crf->CRF, embedding->嵌入, embeddings->嵌入, encoding->编码, flr->FLR, language->语言 ...

### sentence 118

- Rates: no-TM-SFT `0.5714285714285714`, RASST `0.42857142857142855`, V3-real `0.42857142857142855`
- no-TM-SFT correct but RASST missed: tokens -> 令牌
- no-TM-SFT correct but V3-real missed: tokens -> 令牌
- Both RASST and V3-real missed while no-TM-SFT got it: tokens -> 令牌
- Source: The second challenge is that even if one had access to an oracle morphological analyzer, replacing BPE tokens with morphemes is not enough to express the morphological compositionality.
- Reference: 第二个挑战是，即使人们能够使用甲骨文形态化分析器，用语素替换BPE令牌也不足以表达形态学组合性。
- no-TM-SFT hyp: 第二个挑战是，即使能够访问一个形态分析器，用语素替换BPE令牌，也并不足以表达形态上的组合性。
- RASST hyp: 第二个挑战是，即使能够访问一个形态分析器，用语素替换BPE标记也还不够表达形态的组合性。
- V3-real hyp: 第二个挑战是，即使能够访问到形态分析器，用语素替换BPE词元也并不足以表达形态组合性。
- aligned term_map: analysis->分析, bpe->BPE, compositionality->组合性, morphemes->语素, morphological->形态化, morphological analysis->形态分析, morphological analyzer->形态化分析器, morphologically->形态上, morphology->形态学, token->令牌, tokens->令牌

