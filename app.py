import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Set up upload folder and allowed file types.
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df, target_col):
    """
    Preprocess the dataframe:
      - Ensure the target column exists.
      - Separate the target variable (y) and features (X).
      - For numeric features, fill missing values with the median.
      - For categorical features, fill missing values with the mode.
      - For categorical features:
          * If unique values are 10 or less, apply one-hot encoding (drop_first=True).
          * Otherwise, apply label encoding.
      - For the target variable, if non-numeric, apply label encoding.
      - Standardize the features.
    Returns:
      X_scaled: Scaled feature array.
      y: Target variable.
      processed_df: Processed features as a DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataset.")
    
    data = df.copy()
    y = data[target_col]
    X = data.drop(target_col, axis=1)
    
    # Fill missing values for numeric features.
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col].fillna(X[col].median(), inplace=True)
    
    # Fill missing values for categorical features.
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].mode()[0], inplace=True)
    
    # Encode categorical features.
    for col in categorical_cols:
        if X[col].nunique() <= 10:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Process target variable.
    if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
    
    # Standardize the features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X

def evaluate_model(model, X, y):
    """
    Evaluate the given model using K-Fold cross-validation.
    Uses fewer folds if the dataset is large.
    Returns a dictionary with accuracy, precision, recall, and F1 score.
    """
    # Adaptive number of folds: use 5 folds for small datasets, else 3 folds.
    n_folds = 5 if X.shape[0] < 5000 else 3
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
    return {
        'accuracy': round(np.mean(acc_scores) * 100, 2),
        'precision': round(np.mean(prec_scores) * 100, 2),
        'recall': round(np.mean(rec_scores) * 100, 2),
        'f1_score': round(np.mean(f1_scores) * 100, 2)
    }

@app.route('/')
def home():
    """Render the home (upload) page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle the file upload and analysis process:
      1. Validate and save the uploaded CSV file.
      2. Determine the target column (using user input or defaulting to the last column).
      3. Preprocess the dataset.
      4. Optionally sample the dataset if too large.
      5. Evaluate multiple models using adaptive cross-validation.
      6. Compute feature importance for supported models.
      7. Build dictionaries for dataset info and chart data.
      8. Render the results page with the computed data.
    """
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    target_column = request.form.get('target_column', 'auto')
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        
        # Determine target column.
        if target_column == 'auto' or target_column not in df.columns:
            target_column = df.columns[-1]
        if target_column not in df.columns:
            return f"Invalid target column: {target_column}"
        
        # Preprocess the dataset.
        X_scaled, y, processed_df = preprocess_data(df, target_column)
        feature_names = processed_df.columns.tolist()
        
        # If dataset is very large, sample a subset (e.g. 10,000 rows).
        if X_scaled.shape[0] > 10000:
            indices = np.random.choice(X_scaled.shape[0], 10000, replace=False)
            X_scaled = X_scaled[indices]
            y = y[indices]
        
        # Define models to evaluate.
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(kernel='rbf'),
            'K-Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        results = []
        for name, model in models.items():
            metrics = evaluate_model(model, X_scaled, y)
            results.append({'model': name, **metrics})
        
        # Build chart data (bar chart for model accuracies).
        chart_data = {
            'models': [r['model'] for r in results],
            'accuracies': [r['accuracy'] for r in results]
        }
        
        # Compute feature importance for models that support it.
        feature_importance = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                model.fit(X_scaled, y)
                feature_importance[name] = model.feature_importances_.tolist()
        
        # Build dataset info dictionary.
        dataset_info = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.apply(str).to_dict(),
            'columns': df.columns.tolist(),
            'filename': file.filename
        }
        
        return render_template('results.html', 
                               results=results,
                               feature_names=feature_names,
                               target_column=target_column,
                               dataset_info=dataset_info,
                               chart_data=chart_data,
                               feature_importance=feature_importance,
                               enumerate=enumerate)
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
