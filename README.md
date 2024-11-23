![header](header.png)

# Understanding User Perceptions about Products on Tokopedia: A Predictive Modeling Approach

## About
This project aims to automatically extract user perceptions, particularly sentiment labels (negative or positive), about products on an Indonesian e-commerce, namely Tokopedia. To perform automatic extractions, we use a predictive modeling approach, and in order to obtain the highest performing classifier, we carried out both an analysis and a series of machine learning model experiments. 
- **Analysis**: This analysis serves as a basis for exploration, seeking and addressing any potential issues before passing the dataset to the model. Based on our findings, the sentiments of product reviews, as expected, are highly associated with emotions (e.g., positive sentiment linked to joy and love) and customer satisfaction ratings (e.g., the higher the customer ratings, the more likely the sentiment is). In addition to this, different words seem to be associated with different sentiments, e.g., *bagus*, *sesuai*, and *cepat* for positive sentiment whereas *kecewa* and *rusak* for negative sentiment. A closer look to the patterns in frequent words and word combinations highlight product delivery and selling services. For the ML experiment, we did not find any resampling techniques necessary since the distribution of sentiment labels is roughly similar. Equally important, TF-IDF seems to be a good candidate for the vectorizer as different sentiments have unique word selections.
- **ML Experiment**

## Folder Organization

    .
    ├── README.md
    ├── data
    │   ├── PRDECT-ID Dataset.csv
    │   └── cleaned_data.csv
    ├── img
    ├── model
    │   ├── best_model_tfidf.pkl
    │   └── calib_best_model_tfidf.pkl
    └── notebook
        ├── 01_analysis.ipynb
        ├── 02_tfidf-and-ml-models.ipynb
        └── 03_tfidf-and-lstm.ipynb
 

## Feedback
If there are any questions or suggestions for improvements, feel free to contact me here:

