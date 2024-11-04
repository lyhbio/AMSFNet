import numpy as np
import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.decomposition import PCA

from utils.eval import evaluate_model
from utils.utils import precompute_similarity_features, wrap_for_algorithm_func, optimized_feature_extraction

###### Benchmark WorkFlow ######
def train_eval(Wrd, Wrr_list, Wdd_list, Trr, Tdd, algorithm_func=None, drug_embeddings=None, disease_embeddings=None, ml_benchmark=False, folds=10, seed=42):
    """
    Performs 10-fold cross-validation and calculates evaluation metrics for each fold.
    Implements a joint training framework that combines BMC, multi-view learning, and machine learning models.

    Args:
        algorithm_func: The joint training algorithm function.
        drug_embeddings: Matrix of drug embeddings.
        disease_embeddings: Matrix of disease embeddings.
        Wrd: Drug-disease interaction matrix.
        Wrr_list: List of drug similarity matrices.
        Wdd_list: List of disease similarity matrices.
        Trr: Training set of drug similarity matrices.
        Tdd: Training set of disease similarity matrices.
    """
    # Initialize variables
    original_Wrd = Wrd.copy() # Save the original drug-disease interaction matrix
    num_drugs, num_diseases = Wrd.shape

    if (algorithm_func is not None and algorithm_func.__name__ == 'AdaMVL') or ml_benchmark:
        # Precompute similarity features
        drug_sim_features, disease_sim_features = precompute_similarity_features(Trr, Tdd, num_drugs, num_diseases)

    # Get positive and negative sample indices
    print("Retrieving positive and negative sample indices...")
    positiveId = np.where(Wrd.ravel() != 0)[0]  # Indices of positive samples (flattened)
    negativeId = np.where(Wrd.ravel() == 0)[0]  # Indices of negative samples (flattened)

    # Flatten the interaction matrix into a label vector
    y = Wrd.flatten()

    # Create an array to mark which fold each positive sample belongs to
    crossval_id = np.zeros(len(positiveId), dtype=int)

    # Perform stratified K-fold cross-validation
    folds = folds
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    print(f"Assigning positive samples to {folds}-fold cross-validation...")

    for fold, (_, test_index) in enumerate(skf.split(np.zeros(len(positiveId)), np.zeros(len(positiveId)))):
        crossval_id[test_index] = fold

    # Lists to store evaluation metrics for each fold
    auc_scores, aupr_scores, average_f1_scores = [], [], []
    auc_scores_svc, aupr_scores_svc, average_f1_scores_svc = [], [], []
    auc_scores_rf, aupr_scores_rf, average_f1_scores_rf = [], [], []
    auc_scores_xgb, aupr_scores_xgb, average_f1_scores_xgb = [], [], []
    auc_scores_lgb, aupr_scores_lgb, average_f1_scores_lgb = [], [], []
    auc_scores_mlp, aupr_scores_mlp, average_f1_scores_mlp = [], [], []

    # Shuffle negative sample indices
    rng = np.random.default_rng(seed)
    rng.shuffle(negativeId)

    # Prepare for MLP
    if ml_benchmark:
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.fc3(out)
                out = self.sigmoid(out)
                return out

    # Start cross-validation
    for fold in range(folds):
        print(f"\n========== Starting Fold {fold + 1}/{folds} training ========== ")

        # Reset Wrd matrix
        Wrd = original_Wrd.copy()

        # Get positive test sample indices for the current fold
        PtestID = positiveId[crossval_id == fold]

        # Randomly select the same number of negative samples as positive ones for the test set
        start_idx = fold * len(PtestID)
        NtestID = negativeId[start_idx:start_idx + len(PtestID)]

        # Set positive test samples to zero in the Wrd matrix to prevent data leakage
        Wrd.ravel()[PtestID] = 0
        print("Test set prepared, positive interactions masked in Wrd matrix...")

        # Prepare train and test indices
        train_indices = np.setdiff1d(np.arange(len(y)), np.concatenate((PtestID, NtestID)))
        test_indices = np.concatenate((PtestID, NtestID))

        # Prepare labels for train and test sets
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Condition
        if ml_benchmark:
            # Balance the training data by under-sampling negative samples
            positive_train_indices = train_indices[y_train == 1]
            negative_train_indices = train_indices[y_train == 0]
            rng.shuffle(negative_train_indices)
            negative_train_indices = negative_train_indices[:len(positive_train_indices)]
            balanced_train_indices = np.concatenate((positive_train_indices, negative_train_indices))
            rng.shuffle(balanced_train_indices)

            # Prepare features for train and test sets
            X_train, X_test = [], []
            y_train_balanced = y[balanced_train_indices]
            
            print("Extracting features for the training set...")
            X_train = optimized_feature_extraction(balanced_train_indices, num_diseases, drug_sim_features, disease_sim_features, drug_embeddings, disease_embeddings)

            print("Extracting features for the test set...")
            X_test = optimized_feature_extraction(test_indices, num_diseases, drug_sim_features, disease_sim_features, drug_embeddings, disease_embeddings)

            # Feature normalization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print("Features normalized.")

            # Feature selection
            if max(num_diseases, num_drugs) > 1024 or X_train.shape[-1] > 1024:
                pca = PCA(n_components=64, random_state=seed)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                print("Features have been reduced in dimensionality.")
            print(f'The feature dimensions input into the machine learning model are X: {X_train.shape} and y: {y_train_balanced.shape}.')

            # Define the machine learning model (e.g., XGBoost)
            xgb_model = xgb.XGBClassifier(
                n_estimators=999,
                learning_rate=0.1,
                max_depth=0,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=seed,
                n_jobs=-1
            )

            lgb_model = lgb.LGBMClassifier(
                n_estimators=999,
                learning_rate=0.1,
                max_depth=-1,
                objective='binary',
                random_state=seed,
                n_jobs=-1,
                force_col_wise=False,
                verbose=-1
            )

            rf_model = RandomForestClassifier(
                n_estimators=999,
                max_depth=None,
                random_state=seed,
                n_jobs=-1,
            )

            svc_model = SVC(
                probability=True,
                random_state=seed
            )

            # Deep Learning Benchmark(MLP)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

            input_size = X_train.shape[-1]
            hidden_size = 128
            output_size = 1
            model = MLP(input_size, hidden_size, output_size)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            num_epochs = 64

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(X_train_tensor)
                outputs = outputs.squeeze()
                loss = criterion(outputs, y_train_tensor)
                
                loss.backward()
                optimizer.step()

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            model.eval()
            with torch.no_grad():
                y_pred_proba_mlp = model(X_test_tensor).squeeze().numpy()

            auc_mlp, aupr_mlp, avg_f1_mlp = evaluate_model(
                y_test, y_pred_proba_mlp, model_name=f"MLP - Fold {fold + 1}"
            )

            auc_scores_mlp.append(auc_mlp)
            aupr_scores_mlp.append(aupr_mlp)
            average_f1_scores_mlp.append(avg_f1_mlp)

            # Machine Learning Benchmark Evaluate
            # svc
            svc_model.fit(X_train, y_train_balanced)

            y_pred_proba_svm = svc_model.predict_proba(X_test)[:, 1]
            auc_svc, aupr_svc, avg_f1_svc = evaluate_model(
                y_test, y_pred_proba_svm, model_name=f"SVM - Fold {fold + 1}"
            )

            auc_scores_svc.append(auc_svc)
            aupr_scores_svc.append(aupr_svc)
            average_f1_scores_svc.append(avg_f1_svc)

            # rf
            rf_model.fit(X_train, y_train_balanced)
            
            y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
            auc_rf, aupr_rf, avg_f1_rf = evaluate_model(
                y_test, y_pred_proba_rf, model_name=f"RandomForest - Fold {fold + 1}"
            )

            auc_scores_rf.append(auc_rf)
            aupr_scores_rf.append(aupr_rf)
            average_f1_scores_rf.append(avg_f1_rf)

            # xgboost
            xgb_model.fit(X_train, y_train_balanced)
            
            y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
            auc_xgb, aupr_xgb, avg_f1_xgb = evaluate_model(
                y_test, y_pred_proba_xgb, model_name=f"XGBoost - Fold {fold + 1}"
            )

            auc_scores_xgb.append(auc_xgb)
            aupr_scores_xgb.append(aupr_xgb)
            average_f1_scores_xgb.append(avg_f1_xgb)

            # lightgbm
            lgb_model.fit(X_train, y_train_balanced)
            
            y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
            auc_lgb, aupr_lgb, avg_f1_lgb = evaluate_model(
                y_test, y_pred_proba_lgb, model_name=f"LightGBM - Fold {fold + 1}"
            )

            auc_scores_lgb.append(auc_lgb)
            aupr_scores_lgb.append(aupr_lgb)
            average_f1_scores_lgb.append(avg_f1_lgb)

            continue

        if algorithm_func.__name__ == 'AdaMVL':
            # Parameters for the AdaMVL algorithm
            params = {
                'alpha': 10,
                'beta': 10,
                'lamdaR': 0.1,
                'lamdaD': 0.1,
                'threshold': 0.8,
                'max_iter': 400,
                'tol1': 2e-3,
                'tol2': 1e-5,
                'gip_w': 0.2
            }
            F_final = algorithm_func(
                Wrd=Wrd,
                Wrr_list=Wrr_list,
                Wdd_list=Wdd_list,
                params=params
            )
        else:
            F_final = wrap_for_algorithm_func(algorithm_func, Wrd, Wrr_list=Wrr_list, Wdd_list=Wdd_list, Trr=Trr, Tdd=Tdd)

        # Make predictions
        F_final_flat = F_final.flatten()
        y_pred_proba = F_final_flat[test_indices]

        # Evaluate the model
        print(f"Evaluating model performance for Fold {fold + 1}...")
        auc, aupr, avg_f1 = evaluate_model(
            y_test, y_pred_proba, model_name=f"{algorithm_func.__name__} - Fold {fold + 1}"
        )

        # Store metrics for this fold
        auc_scores.append(auc)
        aupr_scores.append(aupr)
        average_f1_scores.append(avg_f1)

    # Final result show
    if ml_benchmark:
        # svc
        print("\n========== SVM Cross-Validation Results ==========")
        print(f"Average AUC: {np.mean(auc_scores_svc):.4f}")
        print(f"Average AUPR: {np.mean(aupr_scores_svc):.4f}")
        print(f"Average F1 (Weighted): {np.mean(average_f1_scores_svc):.4f}")
        # rf
        print("\n========== RandomForest Cross-Validation Results ==========")
        print(f"Average AUC: {np.mean(auc_scores_rf):.4f}")
        print(f"Average AUPR: {np.mean(aupr_scores_rf):.4f}")
        print(f"Average F1 (Weighted): {np.mean(average_f1_scores_rf):.4f}")
        # xgb
        print("\n========== XGBoost Cross-Validation Results ==========")
        print(f"Average AUC: {np.mean(auc_scores_xgb):.4f}")
        print(f"Average AUPR: {np.mean(aupr_scores_xgb):.4f}")
        print(f"Average F1 (Weighted): {np.mean(average_f1_scores_xgb):.4f}")
        # lgb
        print("\n========== LightGBM Cross-Validation Results ==========")
        print(f"Average AUC: {np.mean(auc_scores_lgb):.4f}")
        print(f"Average AUPR: {np.mean(aupr_scores_lgb):.4f}")
        print(f"Average F1 (Weighted): {np.mean(average_f1_scores_lgb):.4f}")
        # mlp
        print("\n========== MLP Cross-Validation Results ==========")
        print(f"Average AUC: {np.mean(auc_scores_mlp):.4f}")
        print(f"Average AUPR: {np.mean(aupr_scores_mlp):.4f}")
        print(f"Average F1 (Weighted): {np.mean(average_f1_scores_mlp):.4f}")
    else:
        print("\n========== Cross-Validation Results ==========")
        print(f"Average AUC: {np.mean(auc_scores):.4f}")
        print(f"Average AUPR: {np.mean(aupr_scores):.4f}")
        print(f"Average F1 (Weighted): {np.mean(average_f1_scores):.4f}")

###### Case Study ######
def case_study(Wrd, Wrr_list, Wdd_list, Trr, Tdd, drug_names, disease_names, algorithm_func=None, top=10):
    """
    Perform a case study to explore new indications for approved drugs using AdaMVL.
    Specifically, this study combines multi-source similarities to predict drug-disease associations on the gold standard dataset Fdataset.
    After predicting the drug-disease associations, it removes already validated associations and outputs the top 10 novel associations.

    Args:
        Wrd: Drug-disease interaction matrix.
        Wrr_list: List of drug similarity matrices.
        Wdd_list: List of disease similarity matrices.
        Trr: Training set of drug similarity matrices.
        Tdd: Training set of disease similarity matrices.
        drug_names: List of drug names corresponding to the rows of Wrd.
        disease_names: List of disease names corresponding to the columns of Wrd.
        algorithm_func: The joint training algorithm function.

    Returns:
        List of top 10 predicted novel drug-disease associations.
    """
    # Initialize variables
    _, num_diseases = Wrd.shape

    # Check if the algorithm function is provided
    if algorithm_func is None:
        raise ValueError("An algorithm function must be provided to perform predictions.")

    # Parameters for the algorithm (specific to AdaMVL)
    params = {
        'alpha': 10,
        'beta': 10,
        'lamdaR': 0.1,
        'lamdaD': 0.1,
        'threshold': 0.8,
        'max_iter': 400,
        'tol1': 2e-3,
        'tol2': 1e-5,
        'gip_w': 0.2
    }

    # Perform prediction using the provided algorithm function
    F_final = algorithm_func(Wrd=Wrd, Wrr_list=Wrr_list, Wdd_list=Wdd_list, params=params)

    # Flatten the interaction matrix and the prediction result matrix
    Wrd_flat = Wrd.flatten()
    F_final_flat = F_final.flatten()

    # Get indices of known drug-disease associations (where interaction exists)
    known_associations_indices = np.where(Wrd_flat != 0)[0]

    # Create a mask to filter out known associations from the predictions
    mask = np.ones(len(F_final_flat), dtype=bool)
    mask[known_associations_indices] = False

    # Apply the mask to get only novel predictions
    novel_predictions_indices = np.where(mask)[0]
    novel_predictions_scores = F_final_flat[novel_predictions_indices]

    # Get the top 10 novel predictions based on their scores
    top_10_indices = np.argsort(novel_predictions_scores)[-top:][::-1]
    top_10_scores = novel_predictions_scores[top_10_indices]
    top_10_pairs = [
        (novel_predictions_indices[idx] // num_diseases, novel_predictions_indices[idx] % num_diseases, top_10_scores[i])
        for i, idx in enumerate(top_10_indices)
    ]

    # Print the top 10 novel drug-disease associations
    print(f"\n========== Top {top} Predicted Novel Drug-Disease Associations ==========")
    for i, (drug_idx, disease_idx, score) in enumerate(top_10_pairs):
        drug_name = drug_names[drug_idx]
        disease_name = disease_names[disease_idx]
        print(f"Rank {i + 1}: Drug {drug_name} - Disease {disease_name} with Score: {score:.4f}")

    return top_10_pairs
