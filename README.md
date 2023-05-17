# nanoGPT

Implementation of Andrej Karpathy's simple decoder transformer for
generating text a la Shakespeare, following video in youtube.

Official nanoGPT code in Karpathy's github is more elaborate as it
supports multiple options for training, like multiple GPUs, decaying
learning rate, etc. Yet, the model should be very similar to the one
here.

Bigram.py implements a super simple bigram model for text generation,
on which the transformer model is later built.

