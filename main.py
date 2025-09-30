import matplotlib
from nltk.corpus import stopwords
from nltk import ngrams
import re
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

num_grams = 2
should_generate_section_files = False
generate_word_cloud = False
generate_bar_chart = False
should_print_idf_table = False
should_print_tf_tables = False
should_print_tfidf_tables = False

def split_epidemics(filename, new_filenames):
    with open(filename, "r") as f:
        text = f.read()
    epidemics = re.findall(r"CHAPTER I\..*?(?=APPENDIX)", text, flags=re.DOTALL)
    for i in range(len(epidemics)):
        epidemics[i] = re.sub(r"^.*(?:SECT\.|CHAPTER).*$\n?", "", epidemics[i], flags=re.MULTILINE)
        if i == 2:
            epidemics[i] = re.sub(r"CATALOGUE OF WORKS.*", "", epidemics[i], flags=re.DOTALL)
    for i in range(len(new_filenames)):
        with open(new_filenames[i], "w") as f:
            f.write(epidemics[i])

def clean_file(filename, n):
    with open(filename) as f:
        text = f.read()
    text = text.lower()
    text = re.sub(r"['’.]", "", text)
    text = re.sub(r"[^a-z1-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word) > 1]
    if n > 1:
        text = list(ngrams(text, n))
    return text

def calculate_tfidf(filenames, column_names, n):
    doc_word_table = pd.DataFrame()
    for i in range(len(filenames)):
        text = clean_file(filenames[i], n)
        num_words = len(text)
        counts = pd.Series(text).value_counts()
        data = pd.DataFrame({'Word' : counts.index, column_names[i] : counts.values / num_words})
        if i == 0:
            doc_word_table = data
        else:
            doc_word_table = pd.merge(doc_word_table, data, how='outer', on='Word')
    doc_word_table.fillna(0, inplace=True)
    num_docs = (doc_word_table[column_names] > 0).sum(axis=1)
    doc_word_table['IDF'] = np.log(len(filenames) / (num_docs + 1))
    for column in column_names:
        doc_word_table[column + 'IDF'] = doc_word_table[column] * doc_word_table['IDF']
    return doc_word_table

def get_document_stats(doc_word_table, column):
    return doc_word_table[['Word', column, 'IDF', column + 'IDF']].sort_values(by=[column + 'IDF'], ascending=False)

def get_document_tfidf(doc_word_table, column):
    tfidf = doc_word_table.rename(columns={column + 'IDF': 'TFIDF'})
    return tfidf[['Word', 'TFIDF']].sort_values(by=['TFIDF'], ascending=False)

def plot_word_cloud(tdidf_table, column):
    word_cloud = WordCloud(width=1000,
                           height=500,
                           background_color="white",
                           relative_scaling=0.5,
                           prefer_horizontal=0.95,
                           colormap="viridis",
                           max_words=30,
                           max_font_size=150)
    word_cloud.generate_from_frequencies(dict(zip(tdidf_table["Word"], tdidf_table["TFIDF"])))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.imshow(word_cloud, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(4)
    fig.suptitle(column[:-3], fontsize=40, weight="bold", y=0.93)
    filename = column[:-3].lower().split()
    filename = filename[0] + "_" + filename[1] + "_word_cloud.png"
    plt.savefig("generated_visuals/" + filename)

def plot_bar_chart(tfidf_table, column):
    tfidf_table["Word"] = tfidf_table["Word"].apply(lambda x: " ".join(x) if isinstance(x, tuple) else x)
    if column == "Black Death TF": color_map_name = 'inferno'
    elif column == "Dancing Mania TF": color_map_name = 'plasma'
    else: color_map_name = 'cividis'
    cmap = matplotlib.colormaps[color_map_name]
    norm = plt.Normalize(tfidf_table["TFIDF"].min(), tfidf_table["TFIDF"].max())
    colors = cmap(norm(tfidf_table["TFIDF"]))
    plt.figure(figsize=(10, 6), dpi=300)
    plt.barh(tfidf_table["Word"], tfidf_table["TFIDF"], color=colors, edgecolor="black")
    plt.gca().invert_yaxis()
    plt.tick_params(axis='y', labelsize=28)
    plt.tick_params(axis='x', labelsize=15)
    plt.xlabel("TF-IDF Score", fontsize=15, fontweight="bold")
    plt.title(column[:-3], fontsize=25, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    filename = column[:-3].lower().split()
    if num_grams == 1: filename = filename[0] + "_" + filename[1] + "_unigram_bar_chart.png"
    elif num_grams == 2: filename = filename[0] + "_" + filename[1] + "_bigram_bar_chart.png"
    else: filename = filename[0] + "_" + filename[1] + "_ngram_bar_chart.png"
    plt.savefig("generated_visuals/" + filename)

def print_idf_table(data):
    idf_table = data.copy()
    idf_table.set_index('Word', inplace=True)
    idf_table.index.name = None
    print(idf_table[['IDF']].sort_values(by=['IDF'], ascending=False).head(15))

def print_tf_table(data, column):
    tf_table = data.copy()
    tf_table.set_index('Word', inplace=True)
    tf_table.index.name = None
    print(tf_table[[column]].sort_values(by=[column], ascending=False).head(15))


if (__name__ == "__main__"):
    epidemic_filenames = ["generated_sections/black_death.txt",
                          "generated_sections/dancing_mania.txt",
                          "generated_sections/sweating_sickness.txt"]
    if should_generate_section_files: split_epidemics("the_epidemics_of_the_middle_ages.txt", epidemic_filenames)
    column_names = ["Black Death TF", "Dancing Mania TF", "Sweating Sickness TF"]
    data = calculate_tfidf(epidemic_filenames, column_names, num_grams)
    if should_print_idf_table: print_idf_table(data)
    for column in column_names:
        if should_print_tf_tables: print_tf_table(data, column)
        tfidf = get_document_tfidf(data.copy(), column).head(20)
        if should_print_tfidf_tables: print(tfidf)
        if num_grams == 1 and generate_word_cloud:
            plot_word_cloud(tfidf, column)
        if generate_bar_chart:
            plot_bar_chart(tfidf.head(10).copy(), column)