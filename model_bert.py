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
    # This ignores any temporal dependence in the data. If a time stamp exists,
    # it is better to use time series cross validation to avoid data leakage.
    Xtr, Xte = train_test_split(
        data_transactions[["original_description"]],
        test_size=0.1,
        shuffle=True,
    )

    # Guided topic modeling nudges BERT to converge on those topics. This hash
    # map is used to map classified transactions into meaningful categories.
    seed_topic_list = [
        ["investment", "invest"],
        ["credit card", "comenity"],
        ["subscription", "membership", "annual", "monthly", "recurring"],
        ["dining", "restaurant", "doordash", "food"],
        ["insurance", "geico", "state farm", "progressive insurance"],
    ]

    # Dictionary of the handful of BERTopic models to test.
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
            nr_topics=10,
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
            nr_topics=10,
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
            nr_topics=10,
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
            nr_topics=10,
        ),
    }

    k_fold = KFold(n_splits=5, shuffle=True)

    # Store the performance metrics for each model to select the best later.
    stats = dict()
    for model_name, base_model in models.items():
        log.info(f"Running k-fold cross validation for model: {model_name}.")

        # Training and validation coherence score and unclassified rates.
        c_tr = list()
        c_te = list()
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

            log.info(f"Topic info: {model.get_topic_info()}")

            # Evaluate models using coherence score and unclassified rate.
            c_tr.append(
                metrics.coherence(
                    bert_model=model,
                    documents=Xtr.iloc[
                        train_indices[0] : train_indices[-1]
                    ].original_description.tolist(),
                )
            )
            c_te.append(
                metrics.coherence(
                    bert_model=model,
                    documents=Xte.iloc[
                        test_indices[0] : test_indices[-1]
                    ].original_description.tolist(),
                )
            )
            unclass_tr.append(metrics.unclassified_rate(topics=topics_tr))
            unclass_te.append(metrics.unclassified_rate(topics=topics_te))

        stats[model_name] = dict()
        # Store training and validation stats in a dictionary for each model.
        stats[model_name]["c_tr"] = round(np.mean(c_tr), 4)
        stats[model_name]["c_te"] = round(np.mean(c_te), 4)
        stats[model_name]["unclass_tr"] = round(np.mean(unclass_tr), 4)
        stats[model_name]["unclass_te"] = round(np.mean(unclass_te), 4)

        log.info(f"Training coherence score: {stats[model_name]['c_tr']}")
        log.info(f"Validation coherence score: {stats[model_name]['c_te']}")
        log.info(f"Training unclass rate: {stats[model_name]['unclass_tr']}")
        log.info(f"Validation unclass rate: {stats[model_name]['unclass_te']}")

    # Select the model with the highest coherence score from the previous
    # 5-fold cross validation tests.
    best_coherence = 0.0
    for model_name, stat_dict in stats.items():
        if stat_dict["c_te"] > best_coherence:
            best_model_name = model_name
            best_purity = stat_dict["c_te"]

    log.info(f"Best model {best_model_name} has purity score {best_coherence}.")

    # Train the best model with the full training data set and run it on the 10%
    # of data that was held out for testing.
    best_model = clone(models[best_model_name])
    topics_tr, probs_tr = best_model.fit_transform(Xtr.original_description.tolist())
    topics_te, probs_te = best_model.transform(Xte.original_description.tolist())

    unclass_tr = round(metrics.unclassified_rate(topics=topics_tr), 4)
    unclass_te = round(metrics.unclassified_rate(topics=topics_te), 4)
    log.info(f"Training unclassified rate for best model {unclass_tr}")
    log.info(f"Testing unclassified rate for best model {unclass_te}")
    log.info(f"Topic info: {best_model.get_topic_info()}")

    # Show the breakdown of the top 5 topics to see if they align with seeding.
    log.info(f"Top 5 topics for best model {best_model_name}.")

    # Topic -1 is the unclassified topic, which does not represent anything.
    for i in range(5):
        log.info(f"Topic {i} : {best_model.get_topic(i)}")
