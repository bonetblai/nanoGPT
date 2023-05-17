# nanoGPT

Implementation of Andrej Karpathy's simple decoder transformer for
generating text a la Shakespeare, following video in youtube.

Official nanoGPT code in Karpathy's github is more elaborate as it
supports multiple options for training, like multiple GPUs, decaying
learning rate, etc. Yet, the model should be very similar to the one
here.

Hyperparameters in final version of Karpathy's model are:
```
# Hyperparameters
g_batch_size =   64  # number of independent sequences processed in parallel
g_block_size =  256  # maximum context size for prediction
g_opt_steps  = 5000  # number of optimization steps
g_eval_freq  =  500  # frequency of evaluation
g_eval_iters =  200  # number of random batches used to evaluate loss
g_gen_length =  500  # length of generated text at the end

g_n_embed    =  384  # number of embedding dimensions
g_n_heads    =    6  # number of heads for multi-head attention
g_n_layers   =    6  # number of blocks
g_dropout    =  0.2  # percentage for dropout

g_learning_rate = 3e-4
```
With those numbers, training must be done in a "good GPU machine".


## Limitations

Theoretically, there is no limitation to the context size. In practice,
it is not clear how to efficiently handle arbitrary sized contexts.
Besides needing a more powerful and general positional encoding, like
the one in the "Attention is all you need" paper, there are other issues
that arise when feeding such contexts to the net.


# Bigram.py

Bigram.py implements a super simple bigram model for text generation,
on which the transformer model is later built.

