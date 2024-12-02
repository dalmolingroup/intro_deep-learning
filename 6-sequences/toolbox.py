import numpy as np
from tensorflow.keras import layers

def pretrained_embedding(vocab):

    '''
    embedding = layers.Embedding(input_dim=vocab_size, 
                                 output_dim=embed_dim,
                                 weights=[embedding_weights],
                                 trainable=False)
    '''
    embed_dim = 100
    embed_dict = {}
    
    vocab_size = len(vocab)

    with open(f"../dataset/glove.6B.{embed_dim}d.txt") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embed_dict[word] = coefs
    
        print(f"Found {len(embed_dict)} word vectors.")

    hits = 0
    misses = 0
    
    # making a vocabulary disctionary from corpus:
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    
    # prepare embedding weights array:
    embedding_weights = np.zeros((vocab_size, embed_dim))
    
    for word, i in vocab_dict.items():
        embedding_vector = embed_dict.get(word)
        if embedding_vector is not None:
            '''
            Words that are not in the embedding index will be all zeros. 
            This also applies to the representations for "padding" and 
            "out of vocabulary (OOV)." 
            '''
            embedding_weights[i] = embedding_vector
            hits += 1
    
        else:
            misses += 1
    
    print(f"Converted {hits} words ({misses} misses)")

    return embedding_weights, embed_dict, embed_dim


def zen_of_python(max_len, embed_dim):
        
    # 1. import corpus:
    corpus = ["Beautiful is better than ugly",
              "Explicit is better than implicit",
              "Simple is better than complex",
              "Complex is better than complicated",
              "Flat is better than nested",
              "Sparse is better than dense",
              "Readability counts",
              "Special cases aren't special enough to break the rules",
              "Although practicality beats purity",
              "Errors should never pass silently",
              "Unless explicitly silenced",
              "In the face of ambiguity, refuse the temptation to guess",
              "There should be one -- and preferably only one -- obvious way to do it",
              "Although that way may not be obvious at first unless you're Dutch",
              "Now is better than never",
              "Although never is often better than right now",
              "If the implementation is hard to explain, it's a bad idea",
              "If the implementation is easy to explain, it may be a good idea",
              "Namespaces are one honking great idea -- let's do more of those!"]

    # 2. transform each sentence into a list of token IDs:
    vectorize = layers.TextVectorization(max_tokens=None,
                                         standardize='lower_and_strip_punctuation',
                                         split='whitespace',
                                         output_mode='int',
                                         output_sequence_length=max_len)
    
    vectorize.adapt(corpus)

    # 3. make word-embedding:
    embedding = layers.Embedding(input_dim=vectorize.vocabulary_size(), output_dim=embed_dim)
    
    # get the embedded tokens
    return embedding(vectorize(corpus)).numpy()
    
    