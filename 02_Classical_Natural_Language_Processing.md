# Questions on Classical Natural Language Processing

## 1. What is tokenization? Give me a difference between lemmatization and stemming?

**Answer:**
Tokenization is the process of breaking text into individual units (tokens) like words, phrases, or sentences for analysis.

**Stemming** removes prefixes/suffixes using simple rules (running → run, better → bett). It's fast but may produce non-words.

**Lemmatization** reduces words to their dictionary form using vocabulary and morphological analysis (running → run, better → good). It's slower but produces valid words and considers context.

## 2. Explain the concept of Bag of Words (BoW) and its limitations.

**Answer:**
Bag of Words represents text as a collection of words, ignoring grammar and word order. Each document becomes a vector where each dimension represents word frequency.

**Limitations:** Loses word order and context, creates sparse high-dimensional vectors, doesn't capture semantic meaning, treats "good" and "excellent" as completely different, and struggles with synonyms and polysemy.

## 3. How does TF-IDF work, and how is it different from simple word frequency?

**Answer:**
TF-IDF (Term Frequency-Inverse Document Frequency) weighs word importance by considering both frequency in document (TF) and rarity across corpus (IDF).

**Formula:** TF-IDF = (word_count/total_words) × log(total_docs/docs_containing_word)

Unlike simple frequency, TF-IDF reduces weight of common words (like "the", "and") and increases weight of rare, discriminative words that better characterize documents.

## 4. What is word embedding, and why is it useful in NLP?

**Answer:**
Word embeddings are dense vector representations of words in continuous space where semantically similar words are positioned closer together.

**Benefits:** Captures semantic relationships, enables arithmetic operations (king - man + woman ≈ queen), reduces dimensionality compared to one-hot encoding, handles synonyms naturally, and provides better input features for machine learning models.

## 5. What are some common applications of NLP in real-world systems?

**Answer:**
**Text Analysis:** Sentiment analysis, spam detection, document classification
**Information Extraction:** Named entity recognition, relationship extraction, keyword extraction  
**Generation:** Chatbots, machine translation, text summarization
**Search:** Search engines, recommendation systems, question answering
**Business:** Customer service automation, content moderation, market research

## 6. What is Named Entity Recognition (NER), and where is it applied?

**Answer:**
NER identifies and classifies named entities in text into predefined categories like Person, Organization, Location, Date, Money, etc.

**Applications:** Information extraction from documents, knowledge graph construction, content analysis, search enhancement, automated form filling, and biomedical text mining for drug/disease identification.

## 7. How does Latent Dirichlet Allocation (LDA) work for topic modeling?

**Answer:**
LDA assumes documents are mixtures of topics, and topics are distributions over words. It uses probabilistic modeling to discover hidden topics in document collections.

**Process:** Assigns words to topics randomly, then iteratively reassigns based on word-topic and document-topic distributions. Each document gets a topic probability distribution, and each topic gets a word probability distribution, revealing underlying thematic structure.

## 8. What are transformers in NLP, and how have they impacted the field?

**Answer:**
Transformers use self-attention mechanisms to process sequences in parallel, replacing recurrent architectures with attention-based computation.

**Impact:** Enabled larger models (BERT, GPT), eliminated sequential processing bottlenecks, captured long-range dependencies better than RNNs, revolutionized transfer learning in NLP, and became foundation for modern language models achieving state-of-the-art performance across tasks.

## 9. What is transfer learning, and how is it applied in NLP?

**Answer:**
Transfer learning uses pre-trained models on large datasets as starting points for specific tasks, leveraging learned representations.

**NLP Application:** Pre-train on large text corpora (like BERT on Wikipedia), then fine-tune on task-specific data (sentiment analysis, NER). This approach requires less data, achieves better performance, and reduces training time compared to training from scratch.

## 10. How do you handle out-of-vocabulary (OOV) words in NLP models?

**Answer:**
**Subword Tokenization:** Use BPE, WordPiece, or SentencePiece to break unknown words into known subword units.
**Special Tokens:** Replace OOV words with <UNK> tokens during training.
**Character-level Models:** Process text at character level to handle any word.
**Pre-trained Embeddings:** Use models trained on larger vocabularies that cover more words.

## 11. Explain the concept of attention mechanisms and their role in sequence-to-sequence tasks.

**Answer:**
Attention allows models to focus on relevant parts of input sequence when generating each output element, rather than relying on fixed-size context vectors.

**Mechanism:** Computes attention weights for each input position, creates weighted context vector, and uses it for current prediction. This solves information bottleneck in encoder-decoder models and enables better handling of long sequences and alignment in translation tasks.

## 12. What is a language model, and how is it evaluated?

**Answer:**
A language model predicts the probability of word sequences, learning patterns and structure of language from text data.

**Evaluation Metrics:**
**Perplexity:** Measures how well model predicts test data (lower is better)
**BLEU/ROUGE:** For generation tasks comparing with reference text
**Downstream Tasks:** Performance on specific applications like classification or QA
**Human Evaluation:** For fluency, coherence, and relevance assessment

---

*These concise answers cover classical NLP fundamentals. Focus on key concepts and be ready to expand with examples if asked during interviews.*