class Params():
    def __init__(self):
        self.gpu_exist = True
        self.device_id = 0
        self.batch_size = 30
        self.hidden_size = 1024
        self.embedding_size = 30
        self.bidirectional = False
        self.epochs = 72
        self.nr_classes = 4
        self.predict_prop = True
        self.nr_prop = 1
        self.rnn_type = 'gru' 
        self.learning_rate = 0.001
        self.latent_size = 196
        self.n_layers = 1
        self.save_every = 10
        self.word_dropout = 0.1
        self.vocab_size = 33 #len(char2ind) 
        self.anneal_function = 'logistic'
        self.k0 = 2500
        self.x0 = 0.0025
        self.save_dir = ''
        



