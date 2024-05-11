import matplotlib
matplotlib.use('agg')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def calculate_correlation(df, selected_features, target, graph_folder):
    selected_columns = selected_features + [target]
    df = df[selected_columns].copy()
    correlation_with_new_features = df.corrwith(df['Reached.on.Time_Y.N']).reset_index()
    correlation_with_new_features.columns = ['Features', 'Correlation with Reached.on.Time_Y.N']
    correlation_with_new_features = correlation_with_new_features.sort_values(by='Correlation with Reached.on.Time_Y.N', ascending=True)

    correlation_with_new_features = correlation_with_new_features[correlation_with_new_features['Features'] != 'Reached.on.Time_Y.N']

    # Reset the index after filtering so that we can get proper order for features
    correlation_with_new_features = correlation_with_new_features.reset_index(drop=True)

    # Plotting the bar chart for all features related to reached on time (vertical graph)
    plt.figure(figsize=(11, 8))
    plt.bar(correlation_with_new_features['Features'], correlation_with_new_features['Correlation with Reached.on.Time_Y.N'], color='lightblue')
    plt.axhline(y=0, color='r', linestyle='dashed', linewidth=1.5)
    plt.xlabel('Features')
    plt.ylabel('Correlation with Reached.on.Time_Y.N')
    plt.title('Correlation of Features with Reached.on.Time_Y.N (Analysis)')
    plt.xticks(rotation=90, ha='center')
    plt.grid(True, axis='y', linestyle='dotted')
    plt.tight_layout()

    # Annotating the correlation values on the plot to add title to every feature with their correlation number
    for i in range(len(correlation_with_new_features)):
        plt.annotate(f"{correlation_with_new_features['Correlation with Reached.on.Time_Y.N'][i]:.2f}",
                     (correlation_with_new_features['Features'][i], correlation_with_new_features['Correlation with Reached.on.Time_Y.N'][i]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')

    # Save the correlation graph
    plt.savefig(os.path.join(graph_folder, 'correlation_graph.png'))
    plt.close()




def run_machine_learning(model_name, df, selected_features, target, graph_folder):
    # Split the data into features (X) and target (y), here we are passing required features to train
    X = df[selected_features]
    y = df[target]

    # Initialize the selected model, here any one model will run at a time.
    if model_name == 'NaiveBayes':
        model = GaussianNB()
    elif model_name == 'LogisticRegression':
        model = LogisticRegression()
    elif model_name == 'K-NearestNeighbours':
        model = KNeighborsClassifier()
    elif model_name == 'SupportVectorMachine':
        model = SVC(probability=True)  # Set probability=True to enable ROC curve and to get rid of error: predict_proba not found
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier()
    elif model_name == 'XGBOOST':
        model = XGBClassifier()
    else:
        raise ValueError("Invalid model name. Please choose from the available models.")
    

    # Split the data into training and testing sets to build the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model to the training data using selected features
    model.fit(X_train, y_train)

    # Make predictions on the test data using selected features
    y_pred = model.predict(X_test)

    # Calculate the accuracy score of the specific model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {model_name} model is {accuracy*100:.2f}')

    print(f'Accuracy of {model_name} model is {accuracy*100:.2f}')
    model_str = ""
    if accuracy > 0.95:
        model_str = "Model might be overfitting."
    elif accuracy > 0.85:
        model_str = "Model is likely a good fit."
    else:
        model_str = "Model might be underfitting." 

    # Generate a confusion matrix of the specific model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix of {model_name}')
    plt.savefig(os.path.join(graph_folder, f'confusion_matrix_{model_name}.png'))
    plt.close()
    

    # Display the classification report of the specific model
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class (class 1) for ROC curve
        roc_auc = roc_auc_score(y_test, y_prob)

        # Plot the ROC curve for the specific model
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve of {model_name}')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(graph_folder, f'roc_curve_{model_name}.png'))
        plt.close()

    print("-------------------------------------------------------------------------------------------")
    print("\n")
    accuracy_percentage = round(accuracy * 100, 2)
    return accuracy_percentage, model_str
