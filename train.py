''' make all necessary imports'''

import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np


''' Arg praser args'''

import argparse

''' The following lines add the argparsing feature to take in user input.
    all values need to be give as is, either with short hands or not.
    The entity and project name are made to mine.
    I have no idea what is to be done with the wandb key for login() purposes.
'''

parser = argparse.ArgumentParser(description='Training the model as per arguments passed')
parser.add_argument('-ld', '--load', type = str, default = "no", help = "enter name of model to load into and place the model in the same directory.")
parser.add_argument('-wp', '--wandb_project', type = str, default = "DLAssignment3", help = "Default has been set to my project name. Please change as per required")
parser.add_argument('-we', '--wandb_entity', type = str, default = "cs22m028")
parser.add_argument('-e', '--epochs',type = int, default = 20)
parser.add_argument('-b', '--batchSize',type = int, default = 256)
parser.add_argument('-lr', '--learningRate', type = float, default = 1e-3)
parser.add_argument('-ed', '--encoderDropOut', type = float, default = 0.3)
parser.add_argument('-dd', '--decoderDropOut', type = float, default = 0.5)
parser.add_argument('-bi', '--biDirectional', type = str, default = "No", help = "yes for a bidirectional LSTM.... If not LSTM, please don't specify. Incorrect dimensions may result")
parser.add_argument('-rnn', '--rnnType', type = str, default = "GRU", help = "GRU, LSTM, RNN")
parser.add_argument('-atn', '--attention', type = str, default = "no", help = "yes for adding attention to the network")
parser.add_argument('-bf', '--bestConf', type = str, default = "yes", help = "yes/no to use best configuration from hyper parameter tuning")
parser.add_argument('-hs', '--hiddenSize', type = int, default = 300, help = "hidden size")
parser.add_argument('-es', '--embedding', type = int, default = 128, help = "embedding size" )
parser.add_argument('-pl', '--plot', type = str, default = "no", help = "To plot or to not" )


args = parser.parse_args()



''' preliminary variables that we maintain global'''
noDict = True
Testing = False
attentionRecord = []
bestConfig = True if args.bestConf == "yes" else False
loaded = True if args.load == "yes" else False
attn = True if args.attention == "yes" else False
plot = True if args.plot == "yes" else False


class PrepText():


    ''' Class for pre processing data. This object will be instantiated once.
        It will contain the dictionary and certain functions for the purpose.
        vectorizeOnWord will be used to convert words into vectors.
        vectorToWord will be used to convert back a vector to a word.
    '''


    def __init__ (self, maxSize):
        ''' initialize the class. We initialze both way dictionaries for english and hindi
            This also initializes the encoding Length and maintains self.noDict to judge whether to make 
            a dictionary or not.
        '''


        self.textToNumX = {}
        self.numToTextX = {}
        self.textToNumY = {}
        self.numToTextY = {}
        self.encodingLength = maxSize
        self.noDict = True


    def makeDict(self, wordsX, wordsY):
        #print ("creating the dictionary.")
        ''' This function makes the dictionary 
            args -> wordsX = the source dataSet.
                    wordsY = the target dataset.

            returns : void. 
                    The dictionaries are just stored in the self of the class
        '''


        self.textToNumX["PAD"] = 0
        self.textToNumX["SOS"] = 1
        self.textToNumX["EOS"] = 2
        self.count = 3
        for word in wordsX:
            for letter in word:
                if letter not in self.textToNumX:
                    self.textToNumX[letter] = self.count
                    self.count+=1

        
        for letter, number in self.textToNumX.items():
            self.numToTextX[number] = letter

        self.textToNumY["PAD"] = 0
        self.textToNumY["SOS"] = 1
        self.textToNumY["EOS"] = 2
        self.count = 3
        for word in wordsY:
            for letter in word:
                if letter not in self.textToNumY:
                    self.textToNumY[letter] = self.count
                    self.count+=1

        
        for letter, number in self.textToNumY.items():
            self.numToTextY[number] = letter
        print (self.textToNumY)
        print (self.textToNumX)
        print ("=============")
        
        print (self.numToTextX)
        print (self.numToTextY)
        
        self.noDict = False
    
    def lenOutput(self):
        ''' args -> none
            return int length of the dictionary
        '''

        return len(self.numToTextY)


    def lenInput(self):
        ''' args -> none
            return int length of the dictionary.
        '''


        return len(self.numToTextX)
    
    def getHinDict (self):
        ''' return the hindi dictionary... This is required for the attention network.'''
        
        return self.textToNumY

        
    def vectorizeOneWord(self, wordX, wordY):
        ''' args -> wordX == source words (in english)
            args -> wordY == target words (in hindi)

            return-> vector encodings of the strings.
        '''



        self.vectorX = torch.zeros(self.encodingLength, dtype = torch.int)
        self.vectorY = torch.zeros(self.encodingLength, dtype = torch.int)


        #print("encoding english word: " + wordX + " encoding hindi word: " + wordY)

        self.count = 1
        self.vectorX[0] = self.textToNumX['SOS']
        for letter in wordX:
            if letter not in self.textToNumX:
                self.vectorX[self.count] = -1
                continue
            self.vectorX[self.count] = self.textToNumX[letter]
            self.count += 1
        self.vectorX[self.count] = self.textToNumX['EOS']



        self.count = 1
        self.vectorY[0] = self.textToNumY['SOS']
        for letter in wordY:
            if letter not in self.textToNumY:
                self.vectorY[self.count] = -1
                continue
            self.vectorY[self.count] = self.textToNumY[letter]
            self.count += 1
        self.vectorY[self.count] = self.textToNumY['EOS']
        
        self.count = 1

        return self.vectorX, self.vectorY

    def vectorToWord (self, x):
        ''' args -> str x
            return string representation (unicode) of the input vector.
        '''


        wordA = ""

        for element in x:
            if element.item() == -1:
                wordA += "</unk>"
                continue
            if element.item () == 0 or element.item() == 1 or element.item() == 2:
                continue
            wordA += self.numToTextY[element.item()]


        return wordA
    


class AksharantarData(Dataset):
    ''' The Aksharantar data set is used to build the data Loader.
        This class inherits the torch DataSet class and is used to build the dataset.
    '''

    def __init__(self, rootPath, max_size, prepTextObj):
        ''' bootstrapped the init class for torch.util.data.dataset.
            This will help just initialize what all is required for the dataset to manufacture.
        '''

        self.root  = rootPath
        self.df = pd.read_csv(self.root, names = ["english", "hindi"])


        self.english = self.df["english"]
        self.hindi = self.df["hindi"]


        self.vocab = prepTextObj
        
        if self.vocab.noDict == True:
            self.vocab.makeDict(self.english, self.hindi)

    
    def convertBack(self, inputX, inputY):
        ''' convert back two inputs to the outputs'''
        return self.vocab.vectorToWord(inputX, inputY)


    def lenOutput(self):
        ''' get the length of the output dictionary (taget dictionary)'''


        return self.vocab.lenOutput()


    def lenInput(self):
        ''' return back the length of the input dictionary (source dictionary)'''


        return self.vocab.lenInput()

    def getDictEng (self):
        '''' return back the English dictionary'''


        return self.vocab.textToNumX

    def getDictHin (self):
        '''' return back the hindi dictionary'''


        return self.vocab.textToNumY

    
    def __len__(self):
        ''' bootstrapped from torch.util.data.dataset, returns the size of the dataset.'''

        return len(self.df)


    def __getitem__ (self, idx):
        ''' return on item. Torch parallelizes here.'''

        #print(idx)

        self.englishWord = self.english[idx]
        #print(self.englishWord)
        self.hindiWord = self.hindi[idx]
        #print(self.hindiWord)
        self.vecEncodedX, self.vecEncodedY = self.vocab.vectorizeOneWord(self.englishWord, self.hindiWord)
        return (self.vecEncodedX, self.vecEncodedY)
    



class EncoderRNN(nn.Module):
    ''' Encoder RNN class. 
        As requested, on embedding layer, one RNN layer and one output layer.

    '''


    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, biDirection, RNN):
        ''' initialize the function. 
            create all relevant layers here as per the pytorch pattern.
        '''


        super(EncoderRNN, self).__init__()
        self.dropout = nn.Dropout(p)
        self.RNN = RNN

        self.embedding = nn.Embedding(input_size, embedding_size)
        
        if (RNN == "LSTM"):
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, batch_first=True, bidirectional = biDirection )
        if (RNN == "GRU"):
            self.rnn = nn.GRU (embedding_size, hidden_size, num_layers, dropout=p, batch_first=True, bidirectional = biDirection )
        if (RNN == "RNN"):
            self.rnn = nn.RNN (embedding_size, hidden_size, num_layers, dropout=p, batch_first=True, bidirectional = biDirection )
        
    def forward(self, x):
        ''' forward prop
            Return the input after forward prop.
            No need for softmax as using nn.CrossEntropyLoss
        '''

        embedding = self.dropout(self.embedding(x))
        
        if self.RNN == "LSTM":
            outputs, (hidden, cell) = self.rnn(embedding)
            
        if self.RNN == "GRU":
            outputs, hidden = self.rnn (embedding)
            
        if self.RNN == "RNN":
            outputs, hidden = self.rnn (embedding)
        
        del (embedding)
        torch.cuda.empty_cache()
        
        
        if self.RNN == "LSTM":
            return outputs, hidden, cell
        else:
            return outputs, hidden
        


class DecoderRNN(nn.Module):
    ''' Decoder RNN will make the Decoder class so that it takes the context vector and generates the relevant output'''


    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, biDirection, RNN):
        ''' initialize all relevant layers and parameters'''
        
        super(DecoderRNN, self).__init__()
        self.dropout = nn.Dropout(p)
        self.RNN = RNN
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.Wattn = nn.Linear (hidden_size, hidden_size, bias = False)
        
        self.Uattn = nn.Linear (hidden_size, hidden_size, bias = False)
        
        self.Vattn = nn.Linear (hidden_size, 1, bias = False)

        self.inputRNN = embedding_size

        if attn == True:
            self.inputRNN = hidden_size + embedding_size
        
        if RNN == "LSTM":
            self.rnn = nn.LSTM(self.inputRNN, hidden_size, num_layers, dropout=p, bidirectional = biDirection, batch_first = True)
            
        if RNN == "RNN":
            self.rnn = nn.RNN (self.inputRNN, hidden_size, num_layers, dropout=p, bidirectional = biDirection, batch_first = True)
            
        if RNN == "GRU":
            self.rnn = nn.GRU (self.inputRNN, hidden_size, num_layers, dropout=p, bidirectional = biDirection, batch_first = True)

        self.fc = nn.Linear(hidden_size*(int(biDirection)+1), output_size)



    def forward(self, x, encoder_output, hidden, cell, batch_size):
        ''' Forward propagate through the layers and return the prediction'''



        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        

        if attn : 
            embedding = embedding.permute (1,0,2)


            Uattn = self.Uattn (encoder_output)
            Wattn = self.Wattn (hidden[-1])
            
            temp = Uattn + Wattn.resize (batch_size, 1, self.hidden_size)
            
            temp1 = torch.nn.functional.tanh (temp)
            
            
            ejt = self.Vattn (temp1)
            
            ajt = torch.nn.Softmax (dim = 1)(ejt)
            
            
            ct = torch.bmm (ajt.transpose(1,2), encoder_output)
            
            
            hello = torch.cat((embedding, ct), dim = 2)
            
            if Testing == True:
                #print ("appending to attentionRecord")
                attentionRecord.append (ajt)
                #print (ajt)
            #print ("here in decoder RNN")
        
        else :
            hello = embedding
            
        
        if (self.RNN == "LSTM"):
            outputs, (hidden, cell) = self.rnn(hello)
        else :
            outputs, hidden = self.rnn (hello)

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)
        
        
        del (outputs)
        if attn:
            del (hello)
            del (ct)
            del (ajt)
            del (ejt)
            del (temp1)
            del (temp)
            del (Uattn)
            del (Wattn)
        del (embedding)
        
        torch.cuda.empty_cache()
        if self.RNN == "LSTM":
            return predictions, hidden, cell
        else:
            return predictions, hidden
        

import gc
class Seq2Seq(nn.Module):
    ''' This is the encoder decoder class and is the class that joins both the encoder and decoder'''


    def __init__(self, encoder, decoder, outputSize, RNN):
        ''' initialize the encoder and decoder and also record which RNN is to be used.'''
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.outputSize = outputSize
        self.RNN = RNN


    def forward(self, source, target, teacherForce = 0):
        ''' forward prop through the layers and generate predictions'''


        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.outputSize


        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        
        
        if self.RNN == "LSTM":
            encoder_output, hidden, cell = self.encoder(source)
        
        else :
            encoder_output, hidden = self.encoder(source)
            
        
        x = target[:,0]
        outputs[0] = torch.ones(outputs[0].shape)

        for t in range(1, target_len):
            # go through the input values one after the other.
            
            if self.RNN == "LSTM":
                output, hidden, cell = self.decoder(x, encoder_output, hidden, cell, batch_size)
                
            else :
                output, hidden = self.decoder (x, encoder_output, hidden, None, batch_size)
            
              
            output = output.resize (batch_size,67)    
            
            outputs[t] = output

            best_guess = output.argmax(dim =1)
    

            x = target[:,t] if random.random() < teacherForce  else best_guess
        
        
        del (hidden)
        if self.RNN == "LSTM":
            del (cell)
        del (best_guess)
        del (target)
        del (x)
        torch.cuda.empty_cache()


        return outputs



def createData(encodingLength, batchSize):  
    ''' Function to make dataLoaders for efficient transfer of data during training/ validation or testing'''
    

    dataPrepper = PrepText (encodingLength)

    path  = input("Enter path to dataset please. In Relative: ")

    
    
    # training data.
    trainData = AksharantarData(path + "/hin/hin_train.csv", encodingLength, dataPrepper)

    # validation data.
    valData = AksharantarData(path + "/hin/hin_valid.csv", encodingLength, dataPrepper) 

    # testing data.
    testData = AksharantarData(path + "/hin/hin_test.csv", encodingLength, dataPrepper)


    # determine the lengths of the different datasets.
    lenIn = trainData.lenInput()
    lenOut = trainData.lenOutput()


    # train data loader.
    trainLoader = DataLoader(trainData, shuffle = False, batch_size = batchSize)

    # validation data loader.
    valLoader = DataLoader(valData, shuffle = False, batch_size = batchSize)

    # test data loader.
    testLoader = DataLoader(testData, shuffle = False, batch_size = batchSize)

    # currently set it to false for debugging purposes.
    input_size = lenIn+1
    output_size = lenOut+1
    
    
    return lenIn, lenOut, dataPrepper, trainLoader,valLoader, testLoader

            
def accuracy(model,dataLoader,batch_size):
    ''' accuracy function to evaluate the number of correct predictions. '''
    
    correct=0
    
    
    for x,y in dataLoader:
        src=x.to(device)
        target=y.to(device)
        output=model(src,target,0)
        
        predictions=torch.argmax(output,dim=2)
        pred=predictions.T
        target1=target
        x= pred
        y = target1
        #print(x.shape)
        #print (y.shape)
       
        for i in range(len(x)):
            mask = torch.eq(y[i], 0).int()
            x[i] = (1-mask) * x[i]

            if torch.equal(x[i][1:], y[i][1:]):
                correct += 1
    return correct




def compile (inputSize, embeddingSize, hiddenSize, outputSize, numLayers, eDrop, dDrop, biDirection, cell_type):
    ''' compile the model. Make the encoder and the decoder and the encoderDecoder model. Return the model'''
    
    
    encoder=EncoderRNN(inputSize, embeddingSize, hiddenSize, numLayers, eDrop, biDirection, cell_type).to(device)
    decoder=DecoderRNN(outputSize, embeddingSize, hiddenSize, outputSize, numLayers,dDrop, biDirection, cell_type).to(device)
    
    model = Seq2Seq (encoder, decoder, outputSize, cell_type).to(device)
    
    return model
    



def fit(lenIn, lenOut, num_layers, enc_dropout, dec_dropout, num_epochs, learning_rate, batchSize, embedding_size,hidden_size, cell_type, trainLoader, valLoader, testLoader, encodingLength):
    ''' the is the traininng loop
        Take in all relevant parameters and run the loop and get model.train () and model.train (false) used to train the model.
    '''
    

    model = compile (lenIn, embedding_size, hidden_size, lenOut, num_layers, enc_dropout, dec_dropout, False, cell_type)
    
    print (model.parameters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    criterion=nn.CrossEntropyLoss(reduction='sum')
    
    patience = 0
    
    
    for epoch in tqdm(range(num_epochs)):
        
        model.train()
        
        
        trainAcc = 0.0
        trainLoss = 0.0
        correct = 0
        total_predictions = 0
        
        
        for x,y in trainLoader:
            
            
            x,y = x.to(device), y.to(device)
            output = model (x,y)
            
            out = output.reshape(-1, output.shape[2])
            y = y.T.reshape(-1)
            
            optimizer.zero_grad()
            
            loss = criterion(out, y.to(torch.long))
            trainLoss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            del (x)
            del (y)
            del (output)
            del (out)
            
        
        correct = accuracy (model, trainLoader, batchSize)
        trainAcc = (correct / (len(trainLoader)*batchSize))*100
            
        print (correct)
        print (f"trainAcc = {trainAcc}")
        print (f"trainLoss = {trainLoss/(len(trainLoader) * batchSize * encodingLength)}")
        
        valLoss = 0.0
        bestAcc = 0.0
        valAcc = 0.0
        
        model.train(False)
        
        for x,y in valLoader:
            
            
            x,y = x.to(device), y.to(device)
            output = model (x,y)
            
            out = output.reshape(-1, output.shape[2])
            y = y.T.reshape(-1)
            
            loss = criterion(out, y.to(torch.long))
            valLoss += loss.item()
            
        correct = accuracy (model, valLoader, batchSize)
        print (correct)
        
        valLoss /= (len (valLoader) * batchSize * 35)
        
        scheduler.step(valLoss)
        
        if ((correct/ (len (valLoader)*batchSize))*100) < bestAcc + 1e-7:
            
            print ("stuck somewhere on the loss surface")
            patience += 1
            
        else:
            
            print("got out of valley...")
            patience = 0
            
        bestAcc = max(bestAcc, (correct/ (len (valLoader)*batchSize))*100)
        
        
        print (f"valAcc = {(correct/ (len (valLoader)*batchSize))*100}")
        
        print (f"valLoss = {valLoss/(len (valLoader) * batchSize * 35)}")
        
        if patience >= 10:
            return model
        
    return model
        

def breakItDown (predictedList, targetList):

    ''' Function to break tensor of tensors into list of lists.'''
    
    A = []
    B = []
    
    for element in predictedList:
        for word in element:
            A.append (word)
    for element in targetList:
        for word in element:
            B.append (word)
        
    return A, B


def plotConf (predictedString, targetedString):
    ''' initial deprecated function for plotting confusion matrix'''
    
    predictions = []
    targets = []
    
    for word in predictedString:
        for letter in word:
            predictions.append (letter)
            
    for word in targets:
        for letter in word:
            targets.append (letter)
            

def convertToString (predictedList, targetList, dataPrepper):
    ''' convert to String'''
    
    predictedString = []
    targetedString = []
    
    for element in range (len (predictedList)):
        
        
        x = dataPrepper.vectorToWord (predictedList[element])
        predictedString.append (x)
        
    for element in targetList:
        x = dataPrepper.vectorToWord (element)
        targetedString.append (x)
        
    
    plotConf (predictedString, targetedString)
    
    return predictedString, targetedString


def Test (model, testLoader, batchSize, dataPrepper):
    ''' Test function to go over the test data loader and output the test accuracy.'''
    
    
    batch = 0
    predictedList = []
    targetList = []
    
    global Testing
    
    
    for x,y in testLoader:
        
        
        if batch == 0:
            Testing = True
        else:
            Testing = False
            
        print (batch)
        print (Testing)
            
            
        x,y = x.to(device), y.to(device)
        
        output = model (x,y,0)
        
        predictions = torch.argmax (output, dim = 2)
        
        predictions = predictions.T
        
        predictedList.append (predictions)
        targetList.append (y)
        
        batch += 1
        
    correct = accuracy (model, testLoader, batchSize)
    
    print (f"accuracy: {(correct/ (len (testLoader)*batchSize))*100}")

    if plot == True:    
        plotConfusion (predictedList, targetList, dataPrepper)
    
    
    return predictedList, targetList



def wandbTrainer ():
    ''' Master Fucntion to set the hyper parameters, run traininig, testing and basically control all other functions.'''

    
    if bestConfig == True:


        batchSize = 256
        encoderEmbedding = 128
        decoderEmbedding = 128
        hiddenSize = 1024
        numLayers = 2
        encDropout = 0.3
        decDropout = 0.3
        num_epochs = 40
        learningRate = 0.001
        bidirectional = True
        varRNN = "GRU"
    
    else:
        
        # initialize the wandb run.
        wandb.init(project = "DLAssignment3", entity = "cs22m028")


        # define where the parameters come from
        parameters = wandb.config


        
        #define the parameters for this training.
        batchSize = parameters["batchSize"]
        encoderEmbedding = parameters["Embedding"]
        decoderEmbedding = parameters["Embedding"]
        hiddenSize = parameters["hiddenSize"]
        numLayers = parameters["numberOfLayers"]
        encDropout = parameters["EncoderDropout"]
        decDropout = parameters["DecoderDropout"]
        num_epochs = parameters["epochs"]
        learningRate = parameters["learningRate"]
        bidirectional = parameters["bidirectional"]
        teach = parameters["teacherForce"]
        duration = parameters["teacherDuration"]
        learningRate = parameters["learningRate"]
        varRNN = parameters["varRNN"]
        teach = 0.5
        duration = 0.5


        wandb.run.name = "config_batchSize_"+str(batchSize)+"_Embedding_"+str(encoderEmbedding)+"_hiddenSize_"+str(hiddenSize)+"_Layers_"+str(numLayers)+"_varRNN_"+str(varRNN)

        
    encodingLength = 35
    
    lenIn, lenOut, dataPrepper, trainLoader,valLoader, testLoader = createData (encodingLength, batchSize) 
    
    
    if loaded == False:

        model = fit (lenIn, lenOut, numLayers,encDropout,decDropout,num_epochs,learningRate,batchSize,encoderEmbedding,hiddenSize,varRNN, trainLoader, valLoader, testLoader, encodingLength)


        torch.save (model, "attentionModel")
        
    else: 

        model_path = input ("Enter path for model")
        
        model = torch.load (model_path)
    


    # obtain the dataLoader objects from the dataLoderCreator.
    
    if bestConfig == True:

        
        predictedList, targetList = Test (model, testLoader, batchSize, dataPrepper)
        
        #predictedList, targetList = makeHeatMaps ()


        predictedList, targetList = breakItDown (predictedList, targetList)    

        predictedList, targetList = convertToString (predictedList, targetList, dataPrepper)

        dumper = pd.DataFrame()
        dumper["predictions"]= predictedList
        dumper["target"] = targetList
        df = pd.read_csv ("/kaggle/input/aksharantar1/aksharantar_sampled/hin/hin_test.csv", names = ["eng", "hin"])        
        dumper["originals"] = df["eng"]
        dumper.to_csv('testSetPreds.csv', index=False)
        
    

def getLogging (key, projectName, entityName):
    ''' Start logging everything.'''


    # initialize the wandb.
    wandb.login(key=key)


    # set up sweep configuration method.
    sweep_config = {
        'method': 'bayes'
        }


    # set up sweep metric.
    metric = {
        'name': 'val_acc',
        'goal': 'maximize'   
        }


    # set sweep config.
    sweep_config['metric'] = metric



    # setup a parameters dictionary.
    parameters_dict = {


        'epochs' : {
            'values':[10,15,20]
        },

        'batchSize' : {
            'values' : [128, 256, 512]
        },

        'Embedding' : {
            'values' : [128, 256, 512]
        },

        'hiddenSize' : {
            'values' : [128, 256, 512, 1024]
        },

        'numberOfLayers' : {
            'values' : [2,4]
        },

        'EncoderDropout' : {
            'values' : [0.3, 0.5]
        },

        'DecoderDropout' : {
            'values' : [0.3, 0.5]
        },

        'learningRate' : {
            'values' : [0.001, 0.0001, 0.0005]
        },

        'bidirectional' : {
            'values' : [True, False]
        },

        'teacherForce' : {
            'values' : [0.5, 0.55, 0.6, 0.7]
        },

        'teacherDuration' : {
            'values' : [0.5, 0.55, 0.6, 0.7]
        },
        
        'varRNN' : {
            'values' : ["LSTM", "RNN", "GRU"]
        }
    }


    # set up the sweep configuration parameters.
    sweep_config['parameters'] = parameters_dict

    # create a sweep_id
    sweep_id = wandb.sweep(sweep_config, project= projectName)

    # wandb agent run.
    wandb.agent(sweep_id, project= projectName , function = wandbTrainer)



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import wandb
import os


def plotConfusion(predictedList, targetList, dataPrepper):
    '''' For the confusion matrix.'''


    font_path = input("Enter tff location so we can plot in devanagiri")  # Path to the uploaded TTF font file
    custom_font = FontProperties(fname=font_path)

    plt.rcParams['font.family'] = custom_font.get_name()

    # create characterwise matching of all 67 pairs.
    dictator = dataPrepper.getHinDict()

    characterNames = []
    ticks = []

    for key, value in dictator.items():
        characterNames.append(key)
        ticks.append(value)

    pred = []

    for element in predictedList:
        for tempEle in element:
            pred.append(np.array(tempEle.to("cpu")))

    del (predictedList)
    predArr = np.array(pred)
    del (pred)

    true = []
    for element in targetList:
        for tempEle in element:
            true.append(np.array(tempEle.to("cpu")))

    trueArr = np.array(true)
    del (true)

    trueNP = np.stack(trueArr)
    del (trueArr)

    predNP = np.stack(predArr)
    del (predArr)

    trueNPFlat = trueNP.flatten()
    predNPFlat = predNP.flatten()

    del (trueNP)
    del (predNP)

    print(trueNPFlat.shape)
    print(predNPFlat.shape)

    cm = confusion_matrix(trueNPFlat, predNPFlat)

    cm = cm[3:, 3:]

    fig, ax = plt.subplots(figsize=(24, 24))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)

    labels = characterNames
    plt.xticks(ticks, labels, rotation=90, fontproperties=custom_font)
    plt.yticks(ticks, labels, fontproperties=custom_font)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.savefig('confusion_matrix.png')

    plt.close()

    plt.tight_layout()
    plt.show()


''' main fucnction to start with the execution.'''

if __name__ == "__main__":
    
    print (bestConfig)


    if bestConfig == False:

        key = input ("Enter key for wandB access : ")
        getLogging (key, args.wandb_project, args.wandb_entity)    
    else:
        wandbTrainer()

