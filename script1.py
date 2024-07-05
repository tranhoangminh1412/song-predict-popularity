import csv
import nltk
from nltk.tokenize import LineTokenizer
import numpy as np
import math
import os
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pandas import Series
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

def load_embeddings(filename):
    """
    This function loads the embedding from a file and returns 2 things
    1) a word_map, this is a dictionary that maps words to an index.
    2) a matrix of row vectors for each word, index the word using the vector.

    :param filename:
    :return: word_map, matrix
    """
    count = 0
    matrix = []
    word_map = {}
    with open(filename, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            items = line.split()
            word = items[0]
            rest = items[1:]
            word_map[word] = count
            count += 1

            rest = list(map(float, rest))
            matrix.append(rest)
    matrix = np.array(matrix)
    return word_map, matrix

def load_compute_impact_vectors(filepath, word_map, matrix):
    df = pd.read_csv(filepath)

    # pull word columns from df
    negative_words = np.array(df[df['n (without dups)'].notna()]['n (without dups)'])
    positive_words = np.array(df[df['p (without dups)'].notna()]['p (without dups)'])

    #reduce
    negative_vector = reduce_sum_word_list(negative_words, word_map, matrix)
    positive_vector = reduce_sum_word_list(positive_words, word_map, matrix)

    return negative_vector, positive_vector

def reduce_sum_word_list(words, word_map, matrix):
    """
    Take a list of words and summarize them as a vector using 'mean'.
    returns a numpy vector
    :param words:
    :param word_map:
    :param matrix:
    :return:
    """
    vec = np.zeros(matrix.shape[1])
    for word in words:
        word = word.lower()
        if word in word_map:
            index = word_map[word]
            vec = vec + matrix[index]
    return vec

def cossim(vA, vB):
    """
    Calcuate the cosine similarity value.
    Returns the similarity value, range: [-1, 1]
    :param vA:
    :param vB:
    :return: similarity
    """
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))

def plotSimilarity(df, song):
    similarities = df[['Negative cossim', 'Positive cossim']]

    sns.heatmap(similarities)
    plt.title(song)
    #plt.show()


def computeCommentVectors(word_map, matrix, comments_df):
    # contains a vector representing all of the songs lyrics
    comment_database = []
    comment_vectors_database = []

    for index, row in comments_df.iterrows():
        comment = row['Comments']

        # remove the single and double quotes
        # not sure this works
        comment = comment.replace('\"', '')
        comment = comment.replace('\'', '')
        # remove punctuation
        comment.strip(string.punctuation)

        # add the sentence to the database
        word_list = nltk.word_tokenize(comment)

        # if the sentence is empty, continue
        if len(word_list) <= 0:
            continue

        # use the embedding map and matrix
        comment_vec = reduce_sum_word_list(word_list, word_map, matrix)

        # add the transformed data to the data storage variables,
        # need two data bases one for the new vector, one for original info.
        comment_database.append(comment)
        comment_vectors_database.append(comment_vec)

    return comment_vectors_database

def findSongImpact(df, negative_vector, positive_vector):
    impact_values = []

    songs = df["Songs"].unique()
    for song in songs:
        song_comment_vectors = df.loc[df['Songs'] == song]["comment_vectors"]
        neg_count = 0
        pos_count = 0
        neutral_count = 0

        neg_cossims = []
        pos_cossims = []
        for comment_vector in song_comment_vectors:
            # calculate cosine similarity
            neg_sim = cossim(comment_vector, negative_vector)
            pos_sim = cossim(comment_vector, positive_vector)
            # add to list for later
            neg_cossims.append(neg_sim)
            pos_cossims.append(pos_sim)
            # compare comment vector to negative and positive vectors
            if (pos_sim > neg_sim):
                # impact_values.append("negative")
                pos_count += 1
            elif (neg_sim > pos_sim):
                # impact_values.append("positive")
                neg_count += 1
            else:
                # impact_values.append("neutral")
                neutral_count += 1

        result = np.argmax([neg_count, pos_count, neutral_count])
        print("Song Name", song)
        print("Result: ", result)
        # result index 0
        if (result == 0):
            # majority negative values
            impact_values.append("-1")
        # result index 1
        elif (result == 1):
            # majority positive values
            impact_values.append("1")
        # result index 2
        else:
            # majority neutral values
            impact_values.append("0")

        print("Negative: ", neg_count)
        print("Positive: ", pos_count)
        print("Neutral: ", neutral_count)
        print("Break")

        # single songs comments plot 
        cossim_df = []

        comments = df.loc[df['Songs'] == song]['Comments']
        comments = comments.reset_index(drop=True)
        cossim_df = pd.DataFrame(data ={'Comments': comments, 'Negative cossim': neg_cossims, 'Positive cossim':pos_cossims})
        #plotSimilarity(cossim_df, song)

    # dataframe with each song and its impact
    return pd.DataFrame({
        'Songs': songs,
        'Impact': impact_values
    })

def label_popularity(row):
    streams = row['Streams']
    if streams < 100000000:
        return 'T4 Popular'
    if streams < 1000000000:
        return 'T3 Popular'
    if streams < 1500000000:
        return 'T2 Popular'
    return 'T1 Popular'

def main():
    # load the embeddings
    data_folder = './data/'
    embeddings_filepath = data_folder + 'glove.6B.50d.txt'
    comments_filepath = data_folder + 'Taylor Swift Songs_Comments - SONGS_COMMENTS (Genres).csv'
    # other data
    # filepath = data_folder + 'Taylor Swift Songs_Comments - SONGS_COMMENTS (Only Taylor).csv'

    word_map, matrix = load_embeddings(embeddings_filepath)

    df = pd.read_csv(comments_filepath)
    df = df.dropna()

    comment_vectors = computeCommentVectors(word_map, matrix, df)

    df['comment_vectors'] = comment_vectors

    impact_filepath = data_folder + 'p_and_n.csv'
    negative_vector, positive_vector = load_compute_impact_vectors(impact_filepath, word_map, matrix)

    impact_df = findSongImpact(df, negative_vector, positive_vector)

    output_df = df.drop(columns=['Comments', 'comment_vectors'])
    output_df = output_df.drop_duplicates()
    output_df = output_df.reset_index(drop=True)


    results = pd.merge(left=output_df, right=impact_df, on='Songs')

    results['Popularity'] = results.apply(label_popularity, axis=1)
    results = results.drop(columns=["Genre"])
    print(results)


    # export to file that will be used in second script
    results.to_csv('data/output.csv')


main()



# def plotEmbeddings(vectors, labels, song):

#     vectors = song_comment_vectors
#     print(negative_vector)
#     vectors.loc['negative'] = negative_vector
#     vectors.loc['positive'] = positive_vector
#     # reduce dimensionality
#     pca = PCA(n_components=2)
#     pca_results = pca.fit_transform(vectors)

#     plt.figure(figsize=(10, 8))
#     plt.scatter(pca_results[:-2, 0], pca_results[:-2, 1])

#     # add labels to points
#     # for i, label in enumerate(labels):
#     #     plt.annotate(label, (pca_vectors[i, 0], pca_vectors[i, 1]))

#     print(pca_results)

#     plt.scatter(pca_results[-2, 0], pca_results[-2, 1], c='red')
#     plt.annotate("Negative", (pca_results[-2, 0], pca_results[-2, 1]))
#     plt.scatter(pca_results[-1, 0], pca_results[-1, 1], c='blue')
#     plt.annotate("Positive", (pca_results[-1, 0], pca_results[-1, 1]))

#     plt.title(song)

#     plt.show()