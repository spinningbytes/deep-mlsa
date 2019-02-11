# deep-mlsa
Code for WWW 2017 conference paper "[Leveraging large amounts of weakly supervised data for multi-language sentiment classification](https://arxiv.org/pdf/1703.02504)"

We provide pre-trained models (CNNs) for sentiment classification in English, French, German and Italian, as well as the code to train new models if needed.

# Setup & Requirements
Python Version: The code is written for Python 3.5 (with backwards compatibility to Python 2.7)

DeepLearning Framework: Keras with both TensorFlow and Theano Backend supported.

   1. Install [Anaconda](https://www.continuum.io/downloads)
   2. Install requirements ```pip install -r requirements.txt```

The whole process for prediction and/or training is configured via the [configuration files available here](https://github.com/spinningbytes/deep-mlsa/tree/master/code/configurations) .

To run the system:
```
$ python runner.py -c mlsa_evaluate/config_supervised_en.json
```

# Running the Code
Configuration files are used to define all settings for prediction and training new models, including model hyperparameters, paths to the data, the word embeddings, the pre-trained models, and the output path.
## Prediction (Using Pre-Trained Models)
For the prediction part start with the [provided configuration file](https://github.com/spinningbytes/deep-mlsa/blob/master/code/configurations/mlsa_evaluate/config_supervised_en.json) and change the following fields:
* input_test_directories
    * direcory_name: path to the test files (one line per text to predict)
    * file_names: list of file names to be tested
    * schema_directory: path to schema (usually the same as the directory name)
* pretrained_model_directory: path to the trained models, which can be downloaded [here](http://spinningbytes.com/resources/)
* pretrained_model_basename: name of the folder in which the model is stored
* embeddings_directory: path to the [word embeddings](http://spinningbytes.com/resources/)
* embeddings_basename: name of the folder in which the word embeddings / vocabulary is stored
* output_path: path to the output directory
* output_basename: name of the folder in which to store all the outputs

For example: if you put the word embeddings into the folder: E:/embeddings/en_embeddings_200M_200d then you set:
* embeddings_directory: E:/embeddings
* embeddings_basename: en_embeddings_200M_200d

Then run the following command:
```
$ python runner.py -c mlsa_evaluate/config_supervised_en.json
```
The results are stored in results/[output_basename], the final prediction scores are stored in ```results/results_log.tsv```

To also output the produced sentence embeddings (last layer output) for each input text, add the flag: ```output_sentence_embeddings: True```

## Training a New Model
For training start from the [provided training configuration](https://github.com/spinningbytes/deep-mlsa/tree/master/code/configurations/mlsa_train) and change the same fields as above. If you want to run only the final supervised phase, starting from a pre-trained model, make sure to set ```transfer_learning: True```.

Then run the following command:
```
$ python runner.py -c mlsa_train/config_supervised_en.json
```

# Corpora
The annotated German sentiment corpus of tweets is made available here, see the website for more details:
[spinningbytes.com/resources/](http://spinningbytes.com/resources/)

# References
Please cite the following paper when using this code or pretrained models for your application.

  Jan Deriu, Aurelien Lucchi, Valeria De Luca, Aliaksei Severyn, Simon Müller, Mark Cieliebak, Thomas Hofmann, Martin Jaggi, [*Leveraging Large Amounts of Weakly Supervised Data for Multi-Language Sentiment Classification*](https://arxiv.org/pdf/1703.02504) WWW 2017 - International World Wide Web Conference

```
@inproceedings{deriu2017mlsent,
  title = {{Leveraging Large Amounts of Weakly Supervised Data for Multi-Language Sentiment Classification}},
  author = {Deriu, Jan and Lucchi, Aurelien and De Luca, Valeria and Severyn, Aliaksei and M{\"u}ller, Simon and Cieliebak, Mark and Hofmann, Thomas and Jaggi, Martin},
  booktitle = {WWW 2017 - International World Wide Web Conference},
  address = {Perth, Australia},
  year = {2017},
}
```
