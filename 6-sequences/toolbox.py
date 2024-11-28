import numpy as np

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
    