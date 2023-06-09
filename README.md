# distributed-NLP-on-supercomputer-with-pytorch-lightning
Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) focusing on the interaction between computers and human language. It involves the development of algorithms and models that enable computers to understand, interpret, and generate natural language in a way that is similar to how humans do. NLP encompasses a wide range of tasks including sentiment analysis, text classification, machine translation, chatbots and virtual assistants, question answering and name entity recognition. 

This repo is intended to share best practices for fine-tuning pretrained models (e.g., BERT and GPT-2) using multiple GPU nodes on a supercomputer managed by SLURM. It encompasses five examples of fine tuning : (1) Document Classification (Sentiment Analysis), (2) Sentense Pair Classification (Natural Language Inference; NLI), (3) Sequence Labeling (Named Entity Recognition), (4) Question Answering, and (5) Sentence Generation. The initial four fine tunnings rely on a BERT pretrainded model, while the Sentence Generation example uses a GPT-2 pretrained model for its fine-tunning. 

Please note that all the code examples are sourced from the [Ratsgo's NLPBOOK](https://ratsgo.github.io/nlpbook/).
* You can access the original code on Google Colab through [Ratgo's Tutorial Link](https://ratsgo.github.io/nlpbook/docs/tutorial_links) 
* Detailed explanations of how the fine-tunning examples work can be found in [Ratsgo's NLPBOOK](https://ratsgo.github.io/nlpbook/)

