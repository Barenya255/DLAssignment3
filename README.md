# DLAssignment3
Machine translation using seq2seq model and then an attention layer mechanism,

# train.py instructions:

train.py takes the following arguments:
- --load -ld : load a model and test instead of training from scratch
- --bestConf -bf : train the model with best Configuration and do not log. Note: to train the model, must specify -ld no
- --wandb_project : project name for the wandb runs.
- --wandb_entity : entity name for the wandb runs.
- --epochs : number of epochs (default to 20)
- --encoderDropOut : encoder Dropout
- --decoderDropOut : decoder Dropout
- --hiddenSize : specify hidden size
- --embedding : specify embedding size
- --attention : specify whether to use attention or not

# DLAssignment3 

- contains the sequence to sequence model all operations performed so far
- contains code for making the confusion matrix 
- contains cell for evaluating on the test data
- dumps the csv file for the text predictions.

# WithAttention.ipynb

- contains the attention layer implementation of the same previous Sequence to Sequence model.
- contains dumps the predictions as a CSV file
- evaluates Accuracy, plots confusion matrix and plots 0 heat maps of the attention on different words.
- Note : the paddings were not removed and therefore some high attention might be taken in there.
