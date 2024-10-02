from bertopic import BERTopic
import logging
import metrics
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    train_test_split,
)
from umap import UMAP


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_PATH = "./data/RM_data_science_technical_assessment_data.csv"


if __name__ == "__main__":

    log.info(f"Loading {DATA_PATH}.")
    data_transactions = pd.read_csv(DATA_PATH)

    # In the interest of berevity, use grid search cross validation to
    # test a handful of models that vary in the clustering and dimensionality
    # reduction methods used. 10% is held out for a final test set.
    Xtr, Xte = train_test_split(
        data_transactions[["original_description"]],
        test_size=0.1,
        shuffle=True,
    )

    # Guided topic modeling nudges BERT to converge on those topics.
    seed_topic_list = [
        "credit card",
        "subscription",
        "utilities",
        "rent",
        "dining",
        "entertainment",
        "investment",
    ]

    k_fold = KFold(
        n_splits=5,
        shuffle=True,
    )

    # Dictionary of the handful of BERTopic models to test.
    # BERTopic models are a sequential implementation of sentence-transformers,
    # UMAP, HDBSCAN, and c-TF-IDF.
    models = {
        "BERTopic.UMAP.01": BERTopic(
            language="english",
            seed_topic_list=seed_topic_list,
            umap_model=UMAP(
                n_neighbors=23,
                n_components=5,
                min_dist=0.01,
                metric="cosine",
            ),
            embedding_model="all-MiniLM-L6-v2",
            calculate_probabilities=True,
        ),
        "BERTopic.UMAP.1": BERTopic(
            language="english",
            seed_topic_list=seed_topic_list,
            umap_model=UMAP(
                n_neighbors=23,
                n_components=5,
                min_dist=0.1,
                metric="cosine",
            ),
            embedding_model="all-MiniLM-L6-v2",
            calculate_probabilities=True,
        ),
        "BERTopic.MPNET.01": BERTopic(
            language="english",
            seed_topic_list=seed_topic_list,
            umap_model=UMAP(
                n_neighbors=23,
                n_components=5,
                min_dist=0.01,
                metric="cosine",
            ),
            embedding_model="all-mpnet-base-v2",
            calculate_probabilities=True,
        ),
        "BERTopic.MPNET.1": BERTopic(
            language="english",
            seed_topic_list=seed_topic_list,
            umap_model=UMAP(
                n_neighbors=23,
                n_components=5,
                min_dist=0.1,
                metric="cosine",
            ),
            embedding_model="all-mpnet-base-v2",
            calculate_probabilities=True,
        ),
    }

    # Store the performance metrics for each model to select the best later.
    stats = dict()
    for model_name, base_model in models.items():
        log.info(f"Running k-fold cross validation for model: {model_name}.")

        # Store the training and validation purity score and unclassified rates.
        purity_tr = list()
        purity_te = list()
        unclass_tr = list()
        unclass_te = list()

        for train_indices, test_indices in k_fold.split(Xtr):
            # Refit the model from scratch on each fold.
            model = clone(base_model)
            # Pass "original_description" into the model pipeline.
            # Train and generate topics and probabilities for the training set.
            topics_tr, probs_tr = model.fit_transform(
                Xtr.iloc[
                    train_indices[0] : train_indices[-1]
                ].original_description.tolist()
            )
            # Generate topics and probabilities for a hold out validation fold.
            topics_te, probs_te = model.transform(
                Xtr.iloc[
                    test_indices[0] : test_indices[-1]
                ].original_description.tolist()
            )

            # Evaluate models using purity score and unclassified rate.
            purity_tr.append(metrics.purity(probs=probs_tr))
            purity_te.append(metrics.purity(probs=probs_te))
            unclass_tr.append(metrics.unclassified_rate(topics=topics_tr))
            unclass_te.append(metrics.unclassified_rate(topics=topics_te))

        stats[model_name] = dict()
        stats[model_name]["purity_tr"] = round(np.mean(purity_tr), 4)
        stats[model_name]["purity_te"] = round(np.mean(purity_te), 4)
        stats[model_name]["unclass_tr"] = round(np.mean(unclass_tr), 4)
        stats[model_name]["unclass_te"] = round(np.mean(unclass_te), 4)

        log.info(f"Training purity score: {stats[model_name]['purity_tr']}")
        log.info(f"Validation purity score: {stats[model_name]['purity_te']}")
        log.info(f"Training unclass rate: {stats[model_name]['unclass_tr']}")
        log.info(f"Validation unclass rate: {stats[model_name]['unclass_te']}")

    # TODO: using the selected model training/test evaluation

    # TODO: with that model print out the top topics.
    # model.get_topic_info() and log it in console.
