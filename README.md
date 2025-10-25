# 🧩 BoardGeekGames Corpus

**BoardGeekGames Corpus** is a Python-based project for building and analyzing an annotated textual corpus of **board game reviews**.  
The project focuses on sentiment analysis, linguistic annotation, and lexicon-based modeling from user-generated content gathered from the [BoardGameGeek](https://boardgamegeek.com) platform.

---

## 🚀 Overview

This project automates the **collection, processing, and annotation** of board game reviews to create a reusable **linguistic corpus** for Natural Language Processing (NLP) and sentiment classification tasks.

### Main Components

1. **Data Acquisition**

   - **Web Crawling / Scraping**: Retrieve user reviews and metadata directly from the BoardGameGeek website.
   - **API Integration**: Access structured data through the official BoardGameGeek API.

2. **Corpus Construction**

   - Implemented with **object-oriented NLTK** corpus interfaces.
   - The corpus can be **saved in JSON format** or **stored in MongoDB** for scalable querying and access.

3. **Corpus Labeling**

   - Reviews are classified into **positive**, **neutral**, and **negative** classes based on user ratings.
   - Data balancing is achieved via:
     - **Undersampling** for overrepresented classes.
     - **Data augmentation** for minority classes using libraries such as [`nlpaug`](https://github.com/makcedward/nlpaug).

4. **Text Preprocessing**

   - Automatic **language detection** using `langdetect`.
   - **Normalization** of text:
     - Emoji handling.
     - Date and numeric normalization.
     - Removal of HTML artifacts and special characters.

5. **Linguistic Annotation**

   - Tokenization, lemmatization, stemming, POS tagging, and dependency parsing using **spaCy** and **NLTK**.
   - The annotated corpus can be exported or stored for downstream NLP tasks.

6. **Lexicon-Based Analysis**
   - Integration of specialized lexicons for:
     - Positive / Negative sentiment.
     - **Boosters** (intensifiers).
     - **Mitigators** (downtoners / hedges).
     - **Negations**.
     - **Domain-specific vocabulary** (board game terminology).
   - Custom lexicon management through the `SentimentLexicon` class.

---

## 🧠 Project Objectives

- Build a domain-specific sentiment corpus for **board game reviews**.
- Explore **linguistic features** and **lexical resources** relevant to sentiment and discourse analysis.
- Provide an extensible foundation for:
  - Sentiment classification models (supervised or lexicon-based).
  - Domain adaptation in NLP.
  - Studies of user-generated language in hobbyist communities.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/lucasmr19/BoardGeekGames-Corpus.git
cd BoardGeekGames-Corpus
```
