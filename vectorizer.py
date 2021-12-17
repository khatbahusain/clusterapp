from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(corpus):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(corpus)

    return vectors



def useml(corpus):

    import tensorflow_text
    import tensorflow_hub as hub    

    model_embedding = hub.load("universal-sentence-encoder-multilingual-large")

    vectors = model_embedding(corpus)


    return vectors