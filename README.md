![header](header.png)

# Understanding User Perceptions about Products on Tokopedia: A Predictive Modeling Approach

## About
This project aims to <b>automatically extract user perceptions, particularly binary sentiment classes (negative or positive), about products on an Indonesian e-commerce, namely Tokopedia</b>. To perform the automatic extraction, we used a predictive modeling approach, and in order to obtain the highest performing classifier, we carried out both a quantitative text analysis and machine learning model experiments. 
- **[Text Analysis](https://nbviewer.org/github/LingAdeu/sentiment-model-experiment/blob/main/notebook/01_analysis.ipynb)**: This analysis serves as a basis for exploration, seeking and addressing any potential issues before passing the dataset to the model. Based on our findings, the sentiments of product reviews, as expected, are highly associated with emotions (e.g., positive sentiment linked to joy and love) and customer satisfaction ratings (e.g., the higher the customer ratings, the more likely the sentiment is). In addition to this, different words seem to be associated with different sentiments, e.g., *bagus*, *sesuai*, and *cepat* for positive sentiment whereas *kecewa* and *rusak* for negative sentiment. A closer look to the patterns in frequent words and word combinations highlight product delivery and selling services. For the ML experiment, we did not find any resampling techniques necessary since the distribution of sentiment labels is roughly similar. Equally important, TF-IDF seems to be a good candidate for the vectorizer as different sentiments have unique word selections.
- **[Conventional ML Experiment](https://nbviewer.org/github/LingAdeu/sentiment-model-experiment/blob/main/notebook/02_tfidf-and-ml-models.ipynb)**: The experiments consist of different setups with an aim to obtain the best performing model in sentiment prediction. The first experiment (still ongoing) uses TF-IDF vectorizer and compares multiple machine learning algorithms, resulting in Support Vector Machine with a linear kernel function as a top candidate with F1 score of 0.95. Currently, we are still diagnosing the model's errors so that we can make an adjustment in the next experiment.
- **[Deep Learning Experiment](https://nbviewer.org/github/LingAdeu/sentiment-model-experiment/blob/main/notebook/03_tfidf-and-lstm.ipynb)**: This experiment uses the RNN LSTM model with several different settings that are expected to get better results than the Machine Learning model in the previous experiment. In the first experiment using LSTM we got an F1 value of 0.74 and in the second experiment by adding bidirectional to LSTM slightly improved the performance, with an F1 value of 0.75. This experiment shows that using complex models such as RNNs, particularly LSTMs, does not necessarily guarantee better prediction performance compared to simpler machine learning models. In our case, the SVM model achieved an impressive F1 score of 0.94, significantly outperforming the LSTM model which only achieved an F1 score of 0.75. This result highlights that the effectiveness of a model depends on the nature of the data and the problem at hand. While deep learning models such as LSTM excel at capturing sequential and temporal patterns, they are not always the optimal choice, especially if simpler models can effectively learn relevant patterns in the data set. This underscores the importance of choosing a model based on the characteristics of the data rather than assuming that greater complexity will always yield better results.
- **[Large Language Model Experiment](https://nbviewer.org/github/LingAdeu/sentiment-model-experiment/blob/main/notebook/04_embeddings-and-IndoBERT.ipynb)**: This experiment involves supervised fine-tuning for sentiment analysis (binary class) on base IndoBERT model based on BERT (Bidirectional Encoder Representations from Transformers) architecture with 111M params ([Koto et al, 2020](https://arxiv.org/pdf/2011.00677)). To understand if the fine-tuning process improved the classifier's predictions, we compared untuned and tuned models. The baseline model achieved only F1 score of 0.45 whereas the tuned model could significantly surpass the baseline model with the F1 score of 0.98. This means that the fine-tuning process successfully increased the model's performance in the sentiment task by 0.53.

## Folder Organization

    .
    ├── README.md                               : Top README file
    ├── data
    │   ├── PRDECT-ID Dataset.csv               : Raw dataset for building ML models
    │   └── cleaned_data.csv                    : Cleaned dataset
    ├── img                                     : Image assets used as illustrations
    ├── model
    │   ├── best_model_tfidf.pkl                : Best model from Exp 1 (SVM)
    │   └── calib_best_model_tfidf.pkl          : Calibrated best model from Exp 1 (SVM)
    └── notebook
        ├── 01_analysis.ipynb                   : Data analysis notebook
        ├── 02_tfidf-and-ml-models.ipynb        : Experiment 1 notebook
        └── 03_tfidf-and-lstm.ipynb             : Experiment 2 notebook
        └── 04_embeddings-and-indobert.ipynb    : Experiment 3 notebook

>[!important]
> Due to the size of the tuned LLM (1.33Gb), it is not possible to store the model on GitHub repository. The model is stored on HuggingFace repository. 

## Feedback
If there are any questions or suggestions for improvements, feel free to contact us here:
- [Adelia Januarto](mailto:januartoadelia@gmail.com)
- [Habib Ja'far Nur](mailto:habibjafar08@gmail.com)
