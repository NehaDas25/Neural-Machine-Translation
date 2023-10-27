# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- CIN: <ins> 401457144 </ins>
- GitHub: <ins> NehaDas25 </ins>
- Email: <ins> ndas@calstatela.edu </ins>


## Features

- Not Implemented:
  - what features have been implemented

<br><br>

- Implemented:
  - PART 2: Neural Machine Translation with Attention
  - PART 2.2: Helper Function
  - PART 2.2.1: Input Encoder
    - Exercise 1: input_encoder_fn
      - Implemented a function input_encoder_fn() that takes in input_vocab_size, d_model, n_encoder_layers.
      - The input encoder runs on the input tokens, creates its embeddings, and feeds it to an LSTM network. This outputs the activations that will be the keys and values for attention. 
      - It is a Serial network which uses:
        1. *tl.Embedding()*: Converts each token to its vector representation. In this case, it is the the size of the vocabulary by the dimension of the model: **tl.Embedding(vocab_size, d_model)**. vocab_size is the number of entries in the given vocabulary. d_model is the number of elements in the word embedding.
        2. *tl.LSTM()*: LSTM layer of size d_model. We want to be able to configure how many encoder layers we have so remember to create LSTM layers equal to the number of the n_encoder_layers parameter.
      - This passed all the unit-test cases.
  
  - PART 2.2.2: Pre-attention Decoder
    - Exercise 2: pre_attention_decoder_fn
      - The pre-attention decoder runs on the targets and creates activations that are used as queries in attention.
      - This is a Serial network which is composed of the following:
        1. *tl.ShiftRight()*: This pads a token to the beginning of your target tokens (e.g. [8, 34, 12] shifted right is [0, 8, 34, 12]). This will act like a start-of-sentence token that will be the first input to the decoder. During training, this shift also allows the target tokens to be passed as input to do teacher forcing.
        2. *tl.Embedding()*: Like in the previous function, this converts each token to its vector representation. In this case, it is the the size of the vocabulary by the dimension of the model: tl.Embedding(vocab_size, d_model). vocab_size is the number of entries in the given vocabulary. d_model is the number of elements in the word embedding.
        3. *tl.LSTM()*: LSTM layer of size d_model.
      - This passed all the unit-test cases as well.

  - PART 2.2.3: Preparing the Attention Input
    - Exercise 3 - prepare_attention_input
      - Implemented a function prepare_attention_input() that takes in encoder_activations, decoder_activations, inputs.
      - Set the keys and values to the encoder activations.
      - Set the queries to the decoder activations.
      - Generated the mask to distinguish real tokens from padding. 
      - Added axes to the mask for attention heads and decoder length.
      - Broadcasted so mask shape is [batch size, attention heads, decoder-len, encoder-len] and here attention heads is set to 1.
      - This passed all the test cases as well.

  - PART 2.3: Implementation Overview
    - Exercise 4 - NMTAttn
      - Implement the NMTAttn function below to define your machine translation model which uses attention that takes nput_vocab_size=33300,target_vocab_size=33300, d_model=1024, n_encoder_layers=2, n_decoder_layers=2, n_attention_heads=4, attention_dropout=0.0, mode='train'.
      - **Step 0**: Prepare the input encoder and pre-attention decoder branches.Already defined this earlier as helper functions so it's just a matter of calling those functions and assigning it to variables.
      - **Step 1**: Created a Serial network. This will stack the layers in the next steps one after the other using **tl.Serial()**.
      - **Step 2**: Made a copy of the input and target tokens. As seen in the diagram mentioned in the assignment, the input and target tokens will be fed into different layers of the model. We can use **tl.Select()** layer to create copies of these tokens. Arrange them as [input tokens, target tokens, input tokens, target tokens].
      - **Step 3**: Created a parallel branch to feed the input tokens to the input_encoder and the target tokens to the pre_attention_decoder. We can use **tl.Parallel()** to create these sublayers in parallel. We have to pass the variables defined in Step 0 as parameters to this layer.
      - **Step 4**: Next, call the prepare_attention_input function to convert the encoder and pre-attention decoder activations to a format that the attention layer will accept. You can use **tl.Fn()** to call this function. Passed the prepare_attention_input function as the f parameter in tl.Fn without any arguments or parenthesis.
      - **Step 5**: We will now feed the (queries, keys, values, and mask) to the **tl.AttentionQKV()** layer. This computes the scaled dot product attention and outputs the attention weights and mask. Take note that although it is a one liner, this layer is actually composed of a deep network made up of several branches. We'll show the implementation taken here to see the different layers used. Then nest it inside a Residual layer to add to the pre-attention decoder activations(i.e. queries) using **tl.Residual()**.
      - **Step 6**: We will not need the mask for the model we're building so we can safely drop it. At this point in the network, the signal stack currently has [attention activations, mask, target tokens] and you can use **tl.Select()** to output just [attention activations, target tokens].
      - **Step 7**: We can now feed the attention weighted output to the LSTM decoder. We can stack multiple **tl.LSTM()** layers to improve the output so remember to append LSTMs equal to the number defined by n_decoder_layers parameter to the model.
      - **Step 8**: We want to determine the probabilities of each subword in the vocabulary and you can set this up easily with a **tl.Dense()** layer by making its size equal to the size of our vocabulary.
      - **Step 9**: Normalize the output to log probabilities by passing the activations in Step 8 to a **tl.LogSoftmax()** layer.
      - This passed all the unit-test as well.
  
  - PART 3: Training
  - PART 3.1: TrainTask
    - Exercise 5: train_task_function
      - Implemented a function train_task_function() that takes in train_batch_stream.
      - Used the train batch stream as labeled data that is **labeled_data=train_batch_stream**.
      - Used the cross entropy loss that is **loss_layer= tl.CrossEntropyLoss()**.
      - Used the Adam optimizer with learning rate of 0.01,**optimizer= trax.optimizers.Adam(0.01)**.
      - Used the trax.lr.warmup_and_rsqrt_decay as the learning rate schedule that have 1000 warmup steps with a max value of 0.01 that is **lr_schedule= trax.lr.warmup_and_rsqrt_decay(1000,0.01)**.
      - Should have a checkpoint every 10 steps, that is **n_steps_per_checkpoint= 10**.
      - This passed all the unit-test cases as well.

  - PART 4: Testing
  - PART 4.1: Decoding
    - Exercise 6 - next_symbol
      - Implemented a function next_symbol() function that takes in the NMTAttn, input_tokens, cur_output_tokens, temperature.
      - Returns the index of the next word.
      - Set the length of the current output tokens.
      - Calculated next power of 2 for padding length, using the formula 2^log_2(token_length + 1). 
      - Pad cur_output_tokens up to the padded_length.
      - The Model expects the output to have an axis for the batch size in front so converted padded list to a numpy array with shape (1, <padded_length>).
      - Get the model prediction and log probabilities from the last token output.
      - Get the next symbol by getting a logsoftmax sample that is **symbol = int(tl.logsoftmax_sample(log_probs, temperature))**.
      - This passed all the unit-test cases ae well.

    - Exercise 7 - sampling_decode
      - Implement the function sampling_decode() that takes ininput_sentence, NMTAttn = None, temperature=0.0, vocab_file=None, vocab_dir=None, next_symbol=next_symbol, tokenize(), detokenize()
      -  Encoded the input sentence. 
      - Initialize an empty the list of output tokens and initialize an integer that represents the current output index. 
      - Set the encoding of the "end of sentence" as 1.
      - Checked that the current output is not the end of sentence token, then updated the current output token by getting the index of the next word and appended the current output token to the list of output tokens.
      - Detokenized the output tokens.
      - This passed all the unit-test cases as well.

    - Exercise 8 - rouge1_similarity
      - Implement the rouge1_similarity() function that takes in system, reference.
      - Make a frequency table of the system tokens.
      - Make a frequency table of the reference tokens.
      - Initialized overlap to 0 and run a for loop over the sys_counter object.
      - Get the precision and recall.
      - if precision + recall != 0, then compute rouge1_score = 2 * ((precision*recall)/(precision+recall)), else rouge1_score = 0.
      - Return rouge1_score.
      - This passed all the test cases as well.
  
    - Exercise 9 - average_overlap
      - Implemented the average_overlap() function that takes in similarity_fn, samples, *ignore_params.
      - Initialized dictionary. 
      - Run a double for loop for each sample.
      - Get the score for the candidate by computing the average.
      - Save the score in the dictionary and use index as the key.
      - Return Scores.
      - This passed all the unit-test cases.
    
    - Exercise 10 - mbr_decode
      - Implemented the mbr_decode() function that takes in sentence, n_samples, score_fn, similarity_fn, NMTAttn=None, temperature=0.6, vocab_file=None, vocab_dir=None, generate_samples=generate_samples, sampling_decode=sampling_decode, next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize.
      - Returns the translated sentence using Minimum Bayes Risk decoding.
      - Generated samples, using generate_samples(sentence, n_samples, NMTAttn, temperature, vocab_file, vocab_dir, sampling_decode, next_symbol, tokenize, detokenize).
      - Used the scoring function to get a dictionary of scores passed in the relevant parameters as shown in the function definition of the mean methods developed earlier that is scores = weighted_avg_overlap(jaccard_similarity, samples, log_probs).
      - Find the key with the highest score.
      - Detokenized the token list associated with the max_score_key.
      - This passed all the unit-test cases as well.


<br><br>

- Partly implemented:
  - w1_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().
<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Any thoughts you may have and would like to share.


## Output

### output:

<pre>
<br/><br/>
Out[3] - 

train data (en, de) tuple: (b'In the pregnant rat the AUC for calculated free drug at this dose was approximately 18 times the human AUC at a 20 mg dose.\n', b'Bei tr\xc3\xa4chtigen Ratten war die AUC f\xc3\xbcr die berechnete ungebundene Substanz bei dieser Dosis etwa 18-mal h\xc3\xb6her als die AUC beim Menschen bei einer 20 mg Dosis.\n')

eval data (en, de) tuple: (b'Subcutaneous use and intravenous use.\n', b'Subkutane Anwendung und intraven\xc3\xb6se Anwendung.\n')

Out[6] - 

Single tokenized example input: [ 8569  4094  2679 32826 22527     5 30650  4729   992     1]
Single tokenized example target: [12647 19749    70 32826 10008     5 30650  4729   992     1]

Out[8] -

Single detokenized example input: Decreased Appetite

Single detokenized example target: Verminderter Appetit


tokenize('hello'):  [[17332   140     1]]
detokenize([17332, 140, 1]):  hello

Out[10] -

input_batch data type:  <class 'numpy.ndarray'>
target_batch data type:  <class 'numpy.ndarray'>
input_batch shape:  (32, 64)
target_batch shape:  (32, 64)

Out[11] - 

THIS IS THE ENGLISH SENTENCE: 
 s categories: very common (≥ 1/ 10); common (≥ 1/ 100, < 1/ 10).
 

THIS IS THE TOKENIZED VERSION OF THE ENGLISH SENTENCE: 
  [   14  2060    64   173   568  5426 30650  4048  5701  3771   115   135
  6722   349  8076   568  5426 30650  4048  5701  3771   115   135  6722
   812  2294 33287   913   135  6722   349 33022 30650  4729   992     1
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0] 

THIS IS THE GERMAN TRANSLATION: 
 aufgeführt: sehr häufig (≥ 1/10); häufig (≥ 1/100, < 1/10).
 

THIS IS THE TOKENIZED VERSION OF THE GERMAN TRANSLATION: 
 [ 9675    64   200  2020  5426 30650  4048  5701  3771   115   135   123
   349  8076  2020  5426 30650  4048  5701  3771   115   135   123   812
  2294 33287   913   135   123   349 33022 30650  4729   992     1     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0]

Out[13] - All tests passed

Out[15] - All tests passed

Out[17] - All tests passed

Out[19] - 

Serial_in2_out2[
  Select[0,1,0,1]_in2_out4
  Parallel_in2_out2[
    Serial[
      Embedding_33300_1024
      LSTM_1024
      LSTM_1024
    ]
    Serial[
      Serial[
        ShiftRight(1)
      ]
      Embedding_33300_1024
      LSTM_1024
    ]
  ]
  PrepareAttentionInput_in3_out4
  Serial_in4_out2[
    Branch_in4_out3[
      None
      Serial_in4_out2[
        _in4_out4
        Serial_in4_out2[
          Parallel_in3_out3[
            Dense_1024
            Dense_1024
            Dense_1024
          ]
          PureAttention_in4_out2
          Dense_1024
        ]
        _in2_out2
      ]
    ]
    Add_in2
  ]
  Select[0,2]_in3_out2
  LSTM_1024
  LSTM_1024
  Dense_33300
  LogSoftmax
]
Expected Output:

Serial_in2_out2[
  Select[0,1,0,1]_in2_out4
  Parallel_in2_out2[
    Serial[
      Embedding_33300_1024
      LSTM_1024
      LSTM_1024
    ]
    Serial[
      Serial[
        ShiftRight(1)
      ]
      Embedding_33300_1024
      LSTM_1024
    ]
  ]
  PrepareAttentionInput_in3_out4
  Serial_in4_out2[
    Branch_in4_out3[
      None
      Serial_in4_out2[
        _in4_out4
        Serial_in4_out2[
          Parallel_in3_out3[
            Dense_1024
            Dense_1024
            Dense_1024
          ]
          PureAttention_in4_out2
          Dense_1024
        ]
        _in2_out2
      ]
    ]
    Add_in2
  ]
  Select[0,2]_in3_out2
  LSTM_1024
  LSTM_1024
  Dense_33300
  LogSoftmax
]

Out[20] - All tests passed

Out[23] - All tests passed

Out[26] - 

Step      1: Total number of trainable weights: 148492820
Step      1: Ran 1 train steps in 55.36 secs
Step      1: train CrossEntropyLoss |  10.42830944

Step      1: eval  CrossEntropyLoss |  10.40574932
Step      1: eval          Accuracy |  0.00000000

Step     10: Ran 9 train steps in 161.24 secs
Step     10: train CrossEntropyLoss |  10.24711132
Step     10: eval  CrossEntropyLoss |  9.95127201
Step     10: eval          Accuracy |  0.02429765

Out[29] - All tests passed

Out[31] -

([161, 12202, 12202, 5112, 5112, 3, 3, 1],
 -0.000102996826171875,
 'Ich liebe liebe Sprachen Sprachen..')

Out[32] - All tests passed

Out[34] -

English:  I am hungry
German:  Ich bin bin hungrhungrig ig..

Out[35] - 

English:  You are almost done with the assignment!
German:  Sie sind sind fast fast mit mit der der Aufgabe Aufgabe!!

Out[37] -

([[595, 75, 75, 67, 67, 352, 352, 102, 102, 1],
  [595, 119, 30166, 67, 705, 352, 352, 102, 102, 1],
  [595, 75, 75, 67, 67, 352, 352, 102, 102, 1],
  [595, 30166, 119, 705, 67, 352, 352, 102, 102, 1]],
 [-1.1444091796875e-05,
  -1.1444091796875e-05,
  -1.1444091796875e-05,
  -6.103515625e-05])

Out[39] - 0.75

Out[41] - 0.8571428571428571

Out[42] - All tests passed

Out[44] - {0: 0.45, 1: 0.625, 2: 0.575}

Out[45] - All tests passed

Out[47] - {0: 0.44255574831883415, 1: 0.631244796869735, 2: 0.5575581009406329}

Out[49] - All tests passed

<br/><br/>
</pre>
