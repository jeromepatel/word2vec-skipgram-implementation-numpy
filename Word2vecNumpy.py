# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 08:29:19 2021

@author: jyotm

Objective: Implement Simple Language Model (preferrabley Word2Vec) from scratch in Numpy, I have used Skip-gram Model (alternative approach is  Continuous Bag-of-Words).

Note: there are different ways to train word2vec models notably using negative sampling known as Skipgram with Negative Sampling although we have not used it here.
To implement skipgram with negative sampling, we take "positive" as well as "negative" context words which gives provides better embeddings and faster training. 
Here, instead I have demonstrated only positive sampling which takes context words with input word into account. 

Steps:
    1. Preprocess text corpus or obtain dataset
    2. Convert into training dataset with features + labels (from input word and context words, respectively)
    3. Initialize language model weights (embeddings and context)
    4. implement training process (feedforward + backpropogation)
    5. start the training and after completion, obtain trained embeddings
    6. Verify the word embeddings by similarity score
    7. Congrats! u completed the job.


references: 
    https://jalammar.github.io/illustrated-word2vec/
    https://arxiv.org/pdf/1301.3781v3.pdf
    Numpy documentation
"""


from keras.preprocessing.text import Tokenizer
import numpy as np

class Dataset:
    def __init__(self, filepath, window_size = 2):
        self.filepath = filepath
        self.train = None
        self.tokenizer = None
        self.content = ""
        self.vocabulary_size = 1000
        self.word_index = None
        self.window_size = window_size
        self.inputs = None
        self.labels = None
        self.index_word = None
    
    def get_raw_text(self):
        with open(self.filepath, "r", encoding = 'utf-8') as f:
            self.content = str(f.read())
    
    def get_training_set(self):
        '''
        Create training data with labels from generated numerical sequence using one hot encoding for each word in corpus.
        Generate input features with focused word, label with the words in window_size. 

        Returns
        -------
        inputs : input feature vector for training.
        labels : corresponding onehot vectors from context.
        '''
        
        sequences = self.text_to_numerical_sequences()         
        
        inputs = []
        labels = []
        
        #loop over all words in the dataset
        for sequence in sequences:
            for i, w_index_val in enumerate(sequence):
                
                
                #convert each word numerical index into onehot encoding, create zero vector of size vocab, one hot enocode that value in vector
                #as zero index is reserved for out of vocab words, we initialize the one hot vector with vocab_size + 1 size 
                word_input_vector = np.zeros(self.vocabulary_size+1)
                word_input_vector[w_index_val] = 1.0
                
                #get word vectors in context of input vector window size, we have used 5
                w_context_vectors = []
                
                for cntx_ind in range(i - self.window_size, i + self.window_size + 1):
                    #check if index is not out of range and not input vector
                    if cntx_ind != i and cntx_ind >= 0 and cntx_ind < len(sequence): 
                        #finally add the onehot encoded context vector to context vectors list
                        context_vect = np.zeros(self.vocabulary_size+1)
                        context_vect[sequence[cntx_ind]] = 1.0
                        w_context_vectors.append(context_vect)
                
                #number of context vector is equal to 2*windows size 
                w_context_vectors = np.array(w_context_vectors)
                
                inputs.append(word_input_vector)
                labels.append(w_context_vectors)
                
        #remove first 2 and last 2 samples as they contain less than 4 context labels,
        #not strictly necessary condition but I used it to make uniform labels numpy array for easy calculation
        inputs = inputs[2:-2]
        labels = labels[2:-2]
                
        inputs = np.array(inputs)
        labels = np.array(labels)
        # labels = np.dstack(labels)
        
        # print(inputs.shape, labels.shape)        
        self.inputs, self.labels = inputs, labels
        return (inputs, labels)
        
        
    def text_to_numerical_sequences(self):
        '''
        I took the liberty to use Tokenizer to get text preprocessed and numerical encoding (integer mapping). As the objective was to focus on training
        the language model rather than dataset. 

        Returns
        -------
        sequnces: numerical sequences

        '''
        
        #read the file 
        self.get_raw_text()
        
        #initialize the tokenizer 
        self.tokenizer = Tokenizer(
            num_words=None,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=" ",
            char_level=False,
            oov_token=None,
            document_count=0)
        
        #use above text to create internal vocab for tokenizer, we will use that to generate sequences
        self.tokenizer.fit_on_texts([self.content])
        
        self.word_index = self.tokenizer.word_index
        self.vocabulary_size = len(self.tokenizer.word_index.keys())
        self.index_word = dict((i, word) for i, word in enumerate(self.word_index))
        sequences = self.tokenizer.texts_to_sequences([self.content])
        # print(sequences)
        return sequences


class word2vec:
    def __init__(self, dataset = 2, embeddings_size = 3, window_size = 2, epochs = 30, learning_rate = 0.15 ):
        #hyperparameters
        self.windows_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.embedding_size = embeddings_size
        
        #variable defining
        self.embedding_weights = None
        self.context_weights = None
        self.train_set = None
        self.dataset = None
        self.loss = 0
        
    def forward_pass(self, input_vector):
        
        #forward propogation using embeddings weights and context weights,
        #we use softmax as our activation function
        
        #hidden shape is input_vector.shape[0], embeddings_size
        # print(input_vector.shape, "the input vector shape, &", self.embedding_weights.shape)
        hidden_vector = np.dot( self.embedding_weights.T, input_vector)
        
        # print(hidden_vector.shape, "the shape of hidden vector is") 
        #output_vector shape is same as input_vector shape
        output_vector = np.dot( self.context_weights.T, hidden_vector)
        
        prediction = self.softmax(output_vector)
        
        # print(prediction.shape)
        
        return (hidden_vector, output_vector,  prediction)
        
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    
        
    def backpropogation(self, input_vector,hidden_vector, output_vector, prediction, labels):
        '''
        We loop through all the target/labels context words for every input feature. Here for windows size of 2, we loop for 4 context word and
        calculate error by summing the different of predication with each context word onehot vector. 

        '''
        #calculate error for all context words
        error_context = [np.subtract(prediction, target) for target in labels]
        Error = np.sum(error_context, axis = 0)
        # print(Error.shape, "the error shape is")
        
        #propogate the error using sgd
        self.update_weights_sgd(input_vector, hidden_vector, Error)
        
        #calculate loss, the loss described here is presented in the paper, skipgram https://arxiv.org/pdf/1411.2738.pdf
        #it consist of two parts, first part is negative sum of all output vectors b4 softmax,
        #second part is log of exp of all outputs multiplied by number of context words
        num_context = labels.shape[0]
        # print(output_vector.shape,"output vector:")
        #part 1, find vector values related to only actual target indices present in labels,
        #for eg. here we look for value in onehot vector in place of index k where k is the index where corresponding context vector value is 1 in onehot vector
        part_1 = [output_vector[np.where(word == 1)] for word in labels]
        part_1 = -np.sum(part_1)
        #part 2 is number of context words multiplied by log of sum of all of exponen
        part_2 = num_context * np.log(np.sum(np.exp(output_vector)))
        # loss = part 1 + part 2
        self.loss += part_1 + part_2
        
        
    def update_weights_sgd(self, input_vector,hidden_vector,error):
        
        #calculate derivative for context and embeddings vectors 
        #the dContex shape is embed size, vocab size 
        delta_context = np.outer(hidden_vector, error)
        delta_embeddings = np.outer(input_vector, np.dot(self.context_weights, error).T)
        
        #now used sgd formula to update weights
        self.embedding_weights -= self.learning_rate * delta_embeddings
        self.context_weights -= self.learning_rate * delta_context
        
        
    def initialize_weights(self):
        
        #initialize weights for embeddings and contexts
        self.embedding_weights = np.random.uniform(-1.0, 1.0, size = (self.dataset.vocabulary_size + 1,self.embedding_size))
        # print(self.embeddings_weights.shape)
        
        self.context_weights = np.random.uniform(-1.0,1.0, size = (self.embedding_size, self.dataset.vocabulary_size + 1))
        # print(self.context_weights.shape)    
    
    def get_dataset(self):
        self.dataset = Dataset("C:/Users/jyotm/Documents/Insternship_work/Word2Vec_Numpy/Dataset/nlp.txt")
        self.train_set = self.dataset.get_training_set()
        
        
    def train_model(self):
        self.get_dataset()
        
        self.initialize_weights()
        
        # print(self.embedding_weights)
        
        inputs, labels = self.train_set
        
        # print(inputs.shape,labels.shape)
        
        for e in range(self.epochs):
            #initialize loss with zero
            self.loss = 0.0
            
            #loop through all the input vectors and corresponding targets
            for i , feature_vec in enumerate(inputs):
                #forward propogation for each feature vector
                hidden_vect, output_vect, prediction = self.forward_pass(feature_vec)
                context_targets = labels[i]
                self.backpropogation(feature_vec,hidden_vect, output_vect, prediction, context_targets)
                
            print(f"Epoch: {e}, loss: {self.loss}")
        
    def word2vec(self, word):
        #get the trained vector embeddings 
        word_ind = self.dataset.word_index[word]
        return self.embedding_weights[word_ind]
    
    def word_similiarity(self, word, top_n_sim):
        #get the top n similar words by cosine similarity 
        word_vec = self.word2vec(word)
        
        word_sim_list = {}

        for i in range(self.dataset.vocabulary_size):
            # Find the cosine sim score b/n two vectors 
             word_vec2 = self.embedding_weights[i]
             #find cosine similarity
             cosine_sim = (word_vec @ word_vec2.T) / (np.linalg.norm(word_vec)*np.linalg.norm(word_vec2))

             word = self.dataset.index_word[i]
             word_sim_list[word] = cosine_sim

        words_sim_sorted = sorted(word_sim_list.items(), key=lambda k: k[1], reverse=True)

        for word, sim in words_sim_sorted[:top_n_sim]:
            print(word, sim)
    
    
print("Starting Word2Vec training:")
train = word2vec()
train.train_model()
train.word_similiarity("machine", 5)