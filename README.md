# deep-mlsa
Code for WWW 2017 conference paper "Leveraging large amounts of weakly supervised data for multi-language sentiment classification"

# Setup & Requirements
Python Version: The code is written for Python 3.5 with backwards compatibility to Python 2.7
DeepLearning Framework: Keras with both TensorFlow and Theano Backend supported.

1) Install [Anaconda](https://www.continuum.io/downloads)
2) Install requirements ```pip install -r requirements.txt```

The whloe porcess is configured via the configuration files you find in the [configurations](https://github.com/spinningbytes/deep-mlsa/tree/master/code/configurations) file. There you find the config files for both training and prediction.

To run the system:
```
$ python runner.py -c mlsa_evaluate/config_supervised_en.json
```

# Configuration Files
The configuration files are used to set the hyperparameters of the system as well as delcare the paths to the data, the embeddings, the pre-trained models, and the output path.
## Prediction
For the prediction part start with the [provided configuration file](https://github.com/spinningbytes/deep-mlsa/blob/master/code/configurations/mlsa_evaluate/config_supervised_en.json) and change the following fields:
* input_test_directories
    * direcory_name: path to the test file
    * file_names: list of file names to be tested
    * schema_directory: path to schema (usually the same as the directory name)
* pretrained_model_directory: path to the trained models, which can be downloaded [here](http://spinningbytes.com/resources/)
* pretrained_model_basename: name of the folder in which the model is stored
* embeddings_directory: path to the [embeddings](http://spinningbytes.com/resources/) (needed as thevocabulary is stored there as well)
* embeddings_basename: name of the folder in which the embeddings/ vocabulary is stored
* output_path: path to the outputs
* output_basename: name of the folder in which to store all the outputs

For example: if you put the embeddings into the folder: E:/embeddings/en_embeddings_200M_200d then you set:
* embeddings_directory: E:/embeddings
* embeddings_basename: en_embeddings_200M_200d

Then run the following command:
```
$ python runner.py -c mlsa_evaluate/config_supervised_en.json
```
The results are stored in results/[output_basename], the scores are stroed in results/results_log.tsv
To store the sentence embeddings put the flag: ```output_sentence_embeddings: True```

## Training
For training start from the [provided configurations](https://github.com/spinningbytes/deep-mlsa/tree/master/code/configurations/mlsa_train) and change the same fields as above. If you are running the supervised phase starting from a pre-trained model, make sure to set ```transfer_learning: True```.

Then run the following command:
```
$ python runner.py -c mlsa_train/config_supervised_en.json
```

## Corpora
The annotated German sentiment corpus of tweets is made available here:
[spinningbytes.com/resources/](http://spinningbytes.com/resources/)