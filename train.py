import os
import json
import shutil
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA


baseline_svm = SVC(
    kernel="rbf",                       
    C=1.0,                 #regularization baseline basically smth smth penalty
    gamma="scale",          #low gamma - wider influence smoooth boundayr 
    probability=True,       # enables probablity estimates
    random_state=42         #sets seed for reproducibility of the rbf kernel
)


#train pipeline

def train():
    # Prefer richer filtered features if available, else fall back to base features
    if os.path.exists("filtered_features.csv"):
        feature_file = "filtered_features.csv"
    elif os.path.exists("features_small.csv"):
        feature_file = "features_small.csv"
    else:
        print("Error: No feature file found. Run 'datapreprocess.py' or 'filter_preprocess.py' first.")
        return

    print(f"Loading extracted features from '{feature_file}'...")
    df = pd.read_csv(feature_file)
    
    target_col = 'genre' if 'genre' in df.columns else df.columns[-1]

    drop_cols = [target_col]
    if 'track_id' in df.columns:
        drop_cols.append('track_id')
    if df.columns[0].startswith('Unnamed') or df.columns[0] == '0':
        drop_cols.append(df.columns[0])

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=49, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA Dimensionality Reduction (retains 95% of variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA: {X_train_scaled.shape[1]} original features -> {X_train_pca.shape[1]} principal components (95% variance)")

    # GridSearchCV: Hyperparameter search over C and gamma
    print("\nRunning GridSearchCV to find best SVM hyperparameters...")
    param_grid = {
        "C": [0.1, 1, 10, 40, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
    }
    grid = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_pca, y_train)
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_ * 100:.2f}%")

    # Support Vector Machine (SVM) configurations using best params from GridSearchCV
    models = {
        "SVM (RBF Baseline, C=1.0)": baseline_svm,
        "SVM (RBF GridSearch Best)": grid.best_estimator_
    }

    best_model = None
    best_acc = 0.0
    best_name = ""
    best_pred = None
    accuracies = {}

    print("\n--- Final Model Evaluation ---")
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        # Baseline uses original scaled features; GridSearch best uses PCA-reduced features
        if name == "SVM (RBF Baseline, C=1.0)":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test_pca)
        acc = accuracy_score(y_test, preds)
        print(f"-> {name} Test Accuracy: {acc * 100:.2f}%")
        accuracies[name] = acc

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            best_pred = preds

    print(f"\n==========================================")
    print(f" BEST MODEL: {best_name} ({best_acc * 100:.2f}% Accuracy)")
    print(f"==========================================\n")

    # Define directories
    train_model_dir = "train_model"
    latest_dir = os.path.join(train_model_dir, "latest_weight")
    best_dir = os.path.join(train_model_dir, "best_weight")

    # Clear and recreate latest directory
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    print("Generating and saving graphs, metrics, and weight files in 'latest_weight'...")
    save_plots_and_metrics(latest_dir, best_model, scaler, label_encoder, best_name, best_acc, best_pred, y_test, accuracies)
    print("Artifacts saved successfully!")

    # Check if latest weight is >= best weight
    best_metrics_path = os.path.join(best_dir, "metrics.json")
    update_best = False
    if not os.path.exists(best_metrics_path):
        update_best = True
    else:
        try:
            with open(best_metrics_path, "r") as f:
                best_metrics = json.load(f)
            if best_acc >= best_metrics.get("accuracy", 0.0):
                update_best = True
        except Exception as e:
            print(f"Warning: Could not read best metrics.json: {e}. Overwriting best weight.")
            update_best = True

    if update_best:
        print(f"\n---> Current accuracy ({best_acc * 100:.2f}%) is >= best saved accuracy. Updating 'best_weight'...")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(latest_dir, best_dir)
        print("Updated 'train_model/best_weight/' successfully with all graphs, metrics, and weights!")
    else:
        with open(best_metrics_path, "r") as f:
            best_metrics = json.load(f)
        print(f"\n---> Current accuracy ({best_acc * 100:.2f}%) did not exceed best saved accuracy ({best_metrics.get('accuracy', 0.0) * 100:.2f}%). 'best_weight' remains unchanged.")


# ==============================================================================
# 2. Plotting, Metrics & Presentation Layer
# ==============================================================================
def save_plots_and_metrics(latest_dir, best_model, scaler, label_encoder, best_name, best_acc, best_pred, y_test, accuracies):
    """
    Generates and saves visual evaluation artifacts, serialized weights, and metric reports.
    """
    # 1. Save model weights
    joblib.dump(best_model, os.path.join(latest_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(latest_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(latest_dir, "label_encoder.joblib"))
    
    # 2. Confusion Matrix Plot (using pure matplotlib)
    cm = confusion_matrix(y_test, best_pred)
    classes = label_encoder.classes_
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, 
        yticklabels=classes,
        title=f"Confusion Matrix\n{best_name} (Acc: {best_acc*100:.2f}%)",
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate confusion matrix values
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    fig.tight_layout()
    plt.savefig(os.path.join(latest_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 3. Model Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(accuracies.keys())
    model_vals = [accuracies[m] * 100 for m in model_names]
    
    bars = ax.barh(model_names, model_vals, color='#3182bd', edgecolor='#1c9099')
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Model Comparison - Test Accuracy')
    ax.set_xlim(0, 105)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                ha='left', va='center', fontweight='bold')
                
    fig.tight_layout()
    plt.savefig(os.path.join(latest_dir, "model_comparison.png"), dpi=150)
    plt.close()
    
    # 4. Save Text and Graphical Classification Report
    report_dict = classification_report(y_test, best_pred, target_names=classes, output_dict=True)
    report_str = classification_report(y_test, best_pred, target_names=classes)
    with open(os.path.join(latest_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)
        
    # Graphical Heatmap of Classification Report
    plot_rows = list(classes) + ['macro avg', 'weighted avg']
    metrics_cols = ['precision', 'recall', 'f1-score']
    matrix = np.zeros((len(plot_rows), len(metrics_cols)))
    
    for i, row_label in enumerate(plot_rows):
        for j, col_label in enumerate(metrics_cols):
            matrix[i, j] = report_dict[row_label][col_label]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics_cols)))
    ax.set_yticks(np.arange(len(plot_rows)))
    ax.set_xticklabels([m.capitalize() for m in metrics_cols], fontsize=11, fontweight='bold')
    ax.set_yticklabels(plot_rows, fontsize=11)
    ax.set_title(f"Classification Report Heatmap\n{best_name} (Accuracy: {best_acc*100:.2f}%)", fontsize=12, fontweight='bold', pad=12)

    for i in range(len(plot_rows)):
        for j in range(len(metrics_cols)):
            val = matrix[i, j]
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    ax.axhline(len(classes) - 0.5, color='gray', linestyle='--', linewidth=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Score (0.0 - 1.0)')
    fig.tight_layout()
    plt.savefig(os.path.join(latest_dir, "classification_report.png"), dpi=150)
    plt.close()

    # 5. Save metrics JSON
    metrics_data = {
        "accuracy": float(best_acc),
        "model_name": best_name,
    }
    with open(os.path.join(latest_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)


# ==============================================================================
# 3. Execution Entry Point
# ==============================================================================
if __name__ == "__main__":
    train()
