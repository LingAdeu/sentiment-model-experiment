![header](header.png)

# Understanding User Perceptions about Products on Tokopedia: A Predictive Modeling Approach

## About
This project aims to automatically extract user perceptions, particularly sentiment labels (negative or positive), about products on an Indonesian e-commerce, namely Tokopedia. To perform automatic extractions, we use a predictive modeling approach, and in order to obtain the highest performing classifier, we carried out both an analysis and a series of machine learning model experiments. 
- **Analysis**: This analysis serves as a basis for exploration, seeking and addressing any potential issues before passing the dataset to the model. Based on our findings, the sentiments of product reviews, as expected, are highly associated with emotions (e.g., positive sentiment linked to joy and love) and customer satisfaction ratings (e.g., the higher the customer ratings, the more likely the sentiment is). In addition to this, different words seem to be associated with different sentiments, e.g., *bagus*, *sesuai*, and *cepat* for positive sentiment whereas *kecewa* and *rusak* for negative sentiment. A closer look to the patterns in frequent words and word combinations highlight product delivery and selling services. For the ML experiment, we did not find any resampling techniques necessary since the distribution of sentiment labels is roughly similar. Equally important, TF-IDF seems to be a good candidate for the vectorizer as different sentiments have unique word selections.
- **ML Experiment**: The experiments consist of different setups with an aim to obtain the best performing model in sentiment prediction. The first experiment (still ongoing) uses TF-IDF vectorizer and compares multiple machine learning algorithms, resulting in Support Vector Machine with a linear kernel function as a top candidate with F1 score of 0.95. Currently, we are still diagnosing the model's errors so that we can make an adjustment in the next experiment.

## Folder Organization

    .
    ├── README.md                           : Top README file
    ├── data
    │   ├── PRDECT-ID Dataset.csv           : Raw dataset for building ML models
    │   └── cleaned_data.csv                : Cleaned dataset
    ├── img                                 : Image assets used as illustrations
    ├── model
    │   ├── best_model_tfidf.pkl            : Best model from Exp 1 (SVM)
    │   └── calib_best_model_tfidf.pkl      : Calibrated best model from Exp 1 (SVM)
    └── notebook
        ├── 01_analysis.ipynb               : Data analysis notebook
        ├── 02_tfidf-and-ml-models.ipynb    : Experiment 1 notebook
        └── 03_tfidf-and-lstm.ipynb         : Experiment 2 notebook
 
>[!important]
> To access the notebook files, storing information about the data analysis and experiments, kindly use these links:
> - [Data Analysis](https://nbviewer.org/github/LingAdeu/sentiment-model-experiment/blob/main/notebook/01_analysis.ipynb)
> - [Experiment 1](https://nbviewer.org/github/LingAdeu/sentiment-model-experiment/blob/main/notebook/02_tfidf-and-ml-models.ipynb): In Progress
> - Experiment 2: Upcoming

## Feedback
If there are any questions or suggestions for improvements, feel free to contact us here:
- [Adelia Januarto](mailto:januartoadelia@gmail.com)
- [Habib Ja'far Nur](habibjafar08@gmail.com)