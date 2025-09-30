# TF–IDF Text Analysis of Epidemics

This script performs TF–IDF analysis on textual data from J. F. C. Hecker’s *The Epidemics of the Middle Ages*. It allows for unigram and bigram analysis, generates visualizations, and can split the original text into separate files for each epidemic if the files have not been generated yet.

---

## Requirements

Install the required Python packages:

```bash
pip install matplotlib nltk pandas numpy wordcloud
```

Download the NLTK stopword list:

```python
import nltk
nltk.download('stopwords')
```

---

## Configurable Flags

You can modify the following flags at the top of the script to control its behavior:

```python
num_grams = 1
should_generate_section_files = False
generate_word_cloud = False
generate_bar_chart = False
should_print_idf_table = False
should_print_tf_tables = False
should_print_tfidf_tables = False
```

| Flag                           | Type  | Description                                                                                        |
|--------------------------------|-------|----------------------------------------------------------------------------------------------------|
| `num_grams`                    | int   | Number of tokens per n-gram. Set to `1` for unigrams, `2` for bigrams, etc.                        |
| `should_generate_section_files`| bool  | If `True`, splits the full text into separate files for each epidemic using regex.                 |
| `generate_word_cloud`          | bool  | If `True` and `num_grams = 1`, generates word clouds for the top 20 TF–IDF terms in each document. |
| `generate_bar_chart`           | bool  | If `True`, generates horizontal bar charts of the top 10 TF–IDF terms.                             |
| `should_print_idf_table`       | bool  | If `True`, prints the top IDF values to the console.                                               |
| `should_print_tf_tables`       | bool  | If `True`, prints top TF values for each document.                                                 |
| `should_print_tfidf_tables`    | bool  | If `True`, prints top TF–IDF tables for each document.                                             |
