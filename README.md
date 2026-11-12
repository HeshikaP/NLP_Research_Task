# Hindi NLP Pipeline: Punctuation Restoration, Sentence Reordering and English Translation

## Project Overview

This project focuses on reconstructing meaningful text from a dataset of seven Hindi files containing unpunctuated and shuffled sentences, followed by translating the reconstructed text into English.

The task involves three main stages:

1. **Punctuation restoration and sentence segmentation**
2. **Logical sentence reordering**
3. **Hindi-to-English translation**, with every five consecutive reordered sentences grouped into a paragraph.

I approached this as a multi-stage NLP problem rather than trying to solve everything using a single model. Each stage was experimented with separately, and the methods used in the final pipeline were selected based on their observed performance, computational feasibility and limitations.

The project was developed using Python, PyTorch, Hugging Face Transformers, and other NLP libraries. General-purpose generative LLMs were not used to perform the actual text processing.

## Repository Structure

The submission contains two notebooks:

- **`Experimentation=https://colab.research.google.com/drive/1vvCE3AhtD9ILiedHF1KnZMKaqjMPim93?usp=sharing`** — Contains the experiments, intermediate implementations, debugging, model comparisons, and approaches tested during development.
- **`Main=https://www.kaggle.com/code/heshikapareek/nlp-research-task`** — Contains the consolidated implementation of the selected pipeline, based on the approaches that performed best during experimentation.

The experimentation notebook documents the development process, whereas the main notebook is intended to provide a cleaner and more reproducible execution of the final approach.

The notebooks may also contain intermediate outputs and diagnostic results to help understand the decisions made during development.

## 1. Understanding the Dataset

The dataset consists of seven Hindi text files, `sample-1.txt` to `sample-7.txt`, containing approximately 5,800 words.

The main challenges were:

- Missing punctuation and unclear sentence boundaries.
- Sentences shuffled within each file.
- Multiple related topics appearing within the same document.
- Context-dependent expressions such as "यह", "इस", and "उपरोक्त".
- Dialogue-heavy text, where speaker changes make segmentation difficult.
- Occurrences of the literal `NULL` token, representing missing or corrupted portions of text.

An important constraint was that **sentence reordering had to remain within each file**. Sentences from different files were never meant to be combined, even when they discussed similar topics.

The task also restricted GPU memory usage to below 1.5 GB, so CPU inference and relatively compact pretrained models were preferred wherever necessary.

---

## 2. Stage One: Punctuation Restoration and Sentence Segmentation

### Approach Used

For punctuation restoration, I used the pretrained model:

`Horizon-Labs/punctuation-restoration-small`

This model predicts punctuation labels such as periods, commas, question marks and colons.

The model's predictions were aligned with the original text using tokenizer offset mappings, allowing punctuation to be inserted without unnecessarily changing the Hindi words.

The `NULL` markers were handled separately during preprocessing to avoid confusing them with ordinary Hindi words.

### What Worked

The model performed reasonably well on straightforward declarative sentences.

For example, it successfully identified the boundary between:

"भगवान राम के पूर्वज महाराज शिवि ने अपना वानप्रस्थ गढ़ मुक्तेश्वर में ही व्यतीत किया था"

and

"चेंगलपट स्टेशन से अरकोनम जाने वाली लाइन पर कांजीवरम स्टेशन है"

These were originally present without punctuation separating them.

### What Did Not Work

The model sometimes merged unrelated sentences, particularly when the text contained abrupt topic changes or conversational dialogue.

It also struggled with lists, incomplete expressions and some `NULL` markers.

This was especially noticeable in samples 6 and 7, which contained dialogue.

### Improvement Attempted

I introduced a hybrid segmentation approach that considered the punctuation model's predictions along with selected additional boundaries around `NULL` markers.

This increased the number of segments in the dialogue-heavy files and reduced some extremely long merged passages.

However, `NULL` was not treated as a universally reliable sentence boundary because it sometimes occurred within a valid sentence.

The resulting segmentation contained **320 provisional sentence segments** across the seven files.

This was an improvement in segmentation granularity, but it does not guarantee that all 320 boundaries are linguistically correct.

---

## 3. Stage Two: Logical Sentence Reordering

This was the most challenging part of the assignment.

Unlike punctuation restoration, sentence ordering requires understanding not only what two sentences discuss, but also how they relate logically.

For example, a sentence introducing a genetically modified cotton seed should appear before a sentence referring to "इस बीज" (this seed).

### Experiment 1: TF-IDF Similarity

I initially experimented with character n-gram TF-IDF to identify related sentences based on overlapping words and character patterns.

**Observation:** It captured some lexical similarity but did not provide enough contextual information for reliable ordering.

Two sentences can share several words without being consecutive in the original passage.

### Experiment 2: Multilingual Sentence Embeddings

I then used:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This generated 384-dimensional sentence embeddings, which were compared using cosine similarity.

The embeddings were considerably more useful for identifying sentences discussing similar subjects.

For example, sentences about religious locations and pilgrimage sites received relatively high semantic similarity scores.

**Limitation:** Semantic similarity is not the same as chronological or logical order.

A sentence explaining a concept may be highly similar to its introduction, but cosine similarity cannot determine which should come first.

Greedy nearest-neighbor ordering therefore produced several incoherent sequences.

### Experiment 3: Topic Clustering

I experimented with K-means and hierarchical clustering to group related sentences.

This was particularly useful for sample 4, which contained two broad topics: cotton cultivation and silk production.

Grouping sentences by topic reduced some unnecessary transitions between unrelated subjects.

However, clustering only determines which sentences belong together, not their correct internal order.

Some sentences were also assigned to imperfect clusters because of overlapping terminology.

### Experiment 4: Discourse Constraints

I examined relationships involving pronouns, references, explanations and conclusions.

For example:

- Introduction of a new cotton seed → explanation referring to "इस बीज".
- Introduction of desirable cotton properties → enumeration of those properties → reference to "उपरोक्त गुण".

These relationships provided useful directional constraints.

However, identifying a few dependencies was insufficient to reconstruct an entire document containing dozens of shuffled sentences.

### Experiment 5: BERT Next Sentence Prediction

I experimented with the pretrained multilingual BERT model:

`google-bert/bert-base-multilingual-cased`

using its Next Sentence Prediction (NSP) head.

For pairs of sentences, I compared the probability of one sentence following another in both directions.

In a small manual test, the model preferred the expected direction for two out of three selected relationships.

This suggested that NSP could provide some useful ordering information.

I then constructed pairwise NSP score matrices and experimented with sequence optimization.

### What Went Wrong

Although optimization improved the numerical NSP transition scores, the resulting passages were not consistently coherent.

Some sequences began with dependent sentences, while others interrupted natural explanation chains.

This revealed an important limitation:

**A sequence with a high sum of local next-sentence scores is not necessarily a logically correct document.**

NSP estimates whether sentences are plausible continuations; it does not directly reconstruct the original global ordering.

### Further Refinements

I also experimented with:

- Constrained topological ordering.
- Separate ordering of thematic groups.
- Hierarchical clustering followed by local ordering.
- Opening-sentence penalties for context-dependent expressions.
- Insertion-based refinement and beam-search-based ordering.

These approaches improved some local relationships and reduced certain topic switches, but none reliably solved the complete discourse-ordering problem.

### Final Ordering Strategy

The final experimental approach combined semantic relationships, NSP-based transition scoring, discourse constraints and sequence-search heuristics.

The purpose was to obtain a more plausible ordering than simple similarity-based greedy selection.

The final output preserves the file boundaries and attempts to arrange sentences into coherent sequences.

**Limitation:** Since the original correct sentence order was not available as ground truth, the ordering cannot be claimed to be fully accurate. The final sequence is a model-generated approximation, and some discourse inconsistencies may remain.

---

## 4. Stage Three: Hindi-to-English Translation

### Initial Experiment: MarianMT

The first translation model tested was:

`Helsinki-NLP/opus-mt-hi-en`

This model was relatively lightweight and could translate all seven files successfully.

However, the translation quality was unsatisfactory.

It frequently distorted proper nouns, especially railway station names, and sometimes changed or omitted important factual relationships.

For example, names such as Chengalpattu, Arakkonam and Mughalsarai were translated incorrectly.

This made the model unsuitable for the final translation stage, despite its successful execution.

### Improved Model: IndicTrans2

I then experimented with:

`ai4bharat/indictrans2-indic-en-dist-200M`

This model is designed specifically for translation between Indian languages and English.

It produced substantially more meaningful translations on the tested Hindi railway passage, especially for geographical names and factual information.

The model was used with the IndicTransToolkit preprocessing and postprocessing utilities.

During implementation, I encountered compatibility issues involving Hugging Face libraries and the model's generation cache.

The generation error was resolved by disabling caching using `use_cache=False`.

### Comparing Two Translation Methods

I tested two approaches:

**Method A — Paragraph-level translation**

Combine five consecutive Hindi sentences and translate the combined paragraph.

**Method B — Sentence-level translation**

Translate each sentence individually and then combine every five English translations into a paragraph.

On the tested railway passage, Method B produced more complete and accurate sentences.

Method A produced an incomplete final sentence, while Method B preserved the relationship between Mathura station and the railway lines more clearly.

### Selected Translation Approach

I selected **sentence-level IndicTrans2 translation followed by five-sentence grouping**.

This was chosen because it showed better preservation of individual sentence meanings in the qualitative comparison.

The translations were generated in batches to improve inference efficiency while keeping each sentence as a separate model input.

The final incomplete group in each file was retained rather than discarded.

One methodological distinction is that grouping translations afterward is not identical to translating an entire five-sentence paragraph as one input. This is a practical compromise based on the observed translation quality.

---

## 5. Overall Pipeline

The complete approach can be summarized as:

**Raw shuffled Hindi text**

↓

**Punctuation restoration using a pretrained punctuation model**

↓

**Hybrid sentence segmentation and boundary repair**

↓

**Sentence representations using TF-IDF and multilingual embeddings**

↓

**Sentence ordering using semantic similarity, NSP scores and discourse-aware constraints**

↓

**IndicTrans2 Hindi-to-English translation**

↓

**Grouping every five consecutive translated sentences**

↓

**Final English output**

Each stage was kept separate so that problems could be identified and tested independently.

## 6. Key Observations and Learnings

The main learning from this assignment was that **different NLP tasks require different types of linguistic information**.

Punctuation restoration mainly depends on local linguistic structure, whereas sentence ordering requires broader context, discourse relationships and logical dependencies.

Semantic similarity was helpful for identifying related sentences, but it could not determine their original order.

Similarly, improving numerical NSP scores did not necessarily improve the overall coherence of the reconstructed document.

For translation, I observed that model selection mattered significantly. The more specialized IndicTrans2 model performed better than MarianMT in the qualitative comparison, particularly for Indian place names and railway-related information.

I also learned the importance of checking actual text outputs rather than relying only on successful execution or numerical scores.

## 7. Limitations

There are several limitations to the final approach:

1. Some sentence boundaries may still be incorrect, particularly in dialogue-heavy files.
2. Sentence reordering remains approximate because the original order is unavailable.
3. NSP and semantic similarity do not fully capture document-level coherence.
4. Translation quality may be affected by errors inherited from punctuation restoration and ordering.
5. The translation comparison was qualitative and performed on a limited number of examples rather than a benchmark dataset.

Therefore, I do not claim that the reconstructed text is identical to the original documents.

## 8. Conclusion

This assignment was an opportunity to explore how different NLP techniques can be combined to solve a multi-stage text reconstruction problem.

The final pipeline was developed through experimentation with punctuation restoration, sentence embeddings, clustering, next-sentence prediction, constrained ordering and machine translation.

Some methods were useful individually but did not solve the entire problem. In particular, sentence ordering remained difficult because local semantic relationships did not always translate into global coherence.

The final implementation represents the most practical combination of the approaches tested within the available time and computational constraints, while acknowledging that further improvements are possible, especially in sentence segmentation and discourse-aware ordering.