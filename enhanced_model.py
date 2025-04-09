import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Set random seed for reproducibility
SEED = 42

def create_enhanced_features(X):
    """Create enhanced features with focus on key business indicators."""
    X = X.copy()
    
    # Select core features that are most predictive
    selected_features = [
        # Key numeric features
        'ApprovalFY', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross',
        'ApprovalDate_quarter', 'DisbursementDate_quarter',
        # Important binary features
        'NewExist_Binary', 'Franchise_Binary', 'UrbanRural_Binary',
        'RevLineCr_Binary', 'LowDoc_Binary', 'CreateJob_Binary',
        'RetainedJob_Binary',
        # Encoded features
        'Bank_Categorized_cod', 'BankState_Categorized_cod',
        'ApprovalFY_Grouped_cod', 'NoEmp_Grouped_cod',
        'DisbursementGross_Grouped_cod',
        # Important state indicators
        'BankState_DE', 'BankState_IL', 'BankState_OH',
        'BankState_Otros', 'BankState_RI'
    ]
    
    X = X[selected_features].copy()
    
    # Core business metrics with proven effectiveness
    X['Jobs_Impact'] = X['CreateJob'] + X['RetainedJob']
    X['Cost_Per_Job'] = X['DisbursementGross'] / (X['Jobs_Impact'] + 1)
    X['Jobs_Retention_Rate'] = X['RetainedJob'] / (X['NoEmp'] + 1)
    X['Jobs_Creation_Rate'] = X['CreateJob'] / (X['NoEmp'] + 1)
    X['Total_Jobs_Rate'] = X['Jobs_Impact'] / (X['NoEmp'] + 1)
    X['Efficiency_Score'] = np.log1p(X['Jobs_Impact']) / np.log1p(X['DisbursementGross'] + 1000)
    
    # Time-based features
    X['Recent_Year'] = (X['ApprovalFY'] >= 2005).astype(int)
    X['Very_Recent'] = (X['ApprovalFY'] >= 2010).astype(int)
    X['Pre_2000'] = (X['ApprovalFY'] < 2000).astype(int)
    X['Mid_2000s'] = ((X['ApprovalFY'] >= 2003) & (X['ApprovalFY'] < 2007)).astype(int)
    X['Approval_Disbursement_Gap'] = X['DisbursementDate_quarter'] - X['ApprovalDate_quarter']
    X['Gap_Significant'] = (abs(X['Approval_Disbursement_Gap']) > 2).astype(int)
    
    # Risk profiles with enhanced criteria
    cost_per_job_75th = X['Cost_Per_Job'].quantile(0.75)
    X['High_Risk_Profile'] = ((X['NewExist_Binary'] == 1) & 
                             (X['LowDoc_Binary'] == 1) & 
                             (X['Cost_Per_Job'] > cost_per_job_75th)).astype(int)
    X['Medium_Risk_Profile'] = ((X['NewExist_Binary'] == 1) ^ 
                               (X['LowDoc_Binary'] == 1)).astype(int)
    X['Low_Risk_Profile'] = ((X['NewExist_Binary'] == 0) & 
                            (X['LowDoc_Binary'] == 0) &
                            (X['Cost_Per_Job'] <= X['Cost_Per_Job'].median())).astype(int)
    
    # Value segments with refined thresholds
    value_85th = X['DisbursementGross'].quantile(0.85)
    value_15th = X['DisbursementGross'].quantile(0.15)
    X['High_Value'] = (X['DisbursementGross'] > value_85th).astype(int)
    X['Low_Value'] = (X['DisbursementGross'] < value_15th).astype(int)
    X['Mid_Value'] = ((X['DisbursementGross'] >= value_15th) & 
                      (X['DisbursementGross'] <= value_85th)).astype(int)
    
    # Business performance indicators
    X['High_Job_Creator'] = (X['Jobs_Creation_Rate'] > X['Jobs_Creation_Rate'].quantile(0.7)).astype(int)
    X['High_Job_Retainer'] = (X['Jobs_Retention_Rate'] > X['Jobs_Retention_Rate'].quantile(0.7)).astype(int)
    X['Cost_Efficient'] = (X['Cost_Per_Job'] < X['Cost_Per_Job'].quantile(0.3)).astype(int)
    X['Balanced_Job_Profile'] = ((X['Jobs_Creation_Rate'] > 0.1) & 
                                (X['Jobs_Retention_Rate'] > 0.1)).astype(int)
    
    # Enhanced interaction features
    X['Risk_Value'] = X['High_Risk_Profile'] * X['DisbursementGross_Grouped_cod']
    X['Time_Risk'] = X['Recent_Year'] * X['High_Risk_Profile']
    X['State_Risk'] = X['High_Risk_Profile'] * (X['BankState_IL'].astype(int) + X['BankState_OH'].astype(int))
    X['Value_Risk'] = X['High_Value'] * X['High_Risk_Profile']
    X['Efficiency_Risk'] = X['High_Risk_Profile'] * (1 - X['Cost_Efficient'])
    X['Time_Value'] = X['Recent_Year'] * X['High_Value']
    
    return X

def create_models():
    """Create optimized versions of the three required models."""
    
    # 1. Random Forest (Decision Tree Algorithm) - Primary model
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0: 2, 1: 1},
        random_state=SEED,
        n_jobs=-1,
        bootstrap=True,
        criterion='entropy',
        max_samples=0.8
    )
    
    # 2. KNN (Geometric Algorithm) - Support model
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='manhattan',
        n_jobs=-1,
        p=1,
        leaf_size=20
    )
    
    # 3. MLP (Secret Algorithm) - Support model
    mlp = MLPClassifier(
        hidden_layer_sizes=(150, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0003,
        learning_rate='adaptive',
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
        beta_1=0.9,
        beta_2=0.999,
        n_iter_no_change=15
    )
    
    return rf, knn, mlp

def preprocess_data(X, scaler=None):
    """Apply standardization to numerical features."""
    numeric_cols = [
        'ApprovalFY', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross',
        'Jobs_Impact', 'Cost_Per_Job', 'Jobs_Retention_Rate', 'Jobs_Creation_Rate',
        'Total_Jobs_Rate', 'Efficiency_Score', 'ApprovalDate_quarter', 
        'DisbursementDate_quarter', 'Approval_Disbursement_Gap', 'Risk_Value'
    ]
    
    if scaler is None:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, scaler

def resample_data(X, y):
    """Apply balanced resampling approach."""
    # First undersampling step
    undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=SEED)
    X_under, y_under = undersampler.fit_resample(X, y)
    
    # SMOTE with careful settings
    oversampler = SMOTE(sampling_strategy=0.9, k_neighbors=6, random_state=SEED)
    X_resampled, y_resampled = oversampler.fit_resample(X_under, y_under)
    
    return X_resampled, y_resampled

def main():
    # Load data
    train_data = pd.read_csv("formated/train_exportado.csv")
    test_data = pd.read_csv("formated/test_exportado.csv")
    
    # Create enhanced features
    X_train = create_enhanced_features(train_data)
    X_test = create_enhanced_features(test_data)
    
    # Get target variable
    y_train = train_data['Accept'].values
    
    # Preprocess data
    X_train_processed, scaler = preprocess_data(X_train)
    X_test_processed, _ = preprocess_data(X_test, scaler)
    
    # Resample training data
    print("Resampling data...")
    X_resampled, y_resampled = resample_data(X_train_processed, y_train)
    
    # Create and train models
    print("Training models...")
    rf_model, knn_model, mlp_model = create_models()
    
    # Train each model
    rf_model.fit(X_resampled, y_resampled)
    knn_model.fit(X_resampled, y_resampled)
    mlp_model.fit(X_resampled, y_resampled)
    
    # Make predictions
    print("Making predictions...")
    rf_proba = rf_model.predict_proba(X_test_processed)[:, 1]
    knn_proba = knn_model.predict_proba(X_test_processed)[:, 1]
    mlp_proba = mlp_model.predict_proba(X_test_processed)[:, 1]
    
    # Weighted average of probabilities with refined weights
    final_proba = 0.6 * rf_proba + 0.1 * knn_proba + 0.3 * mlp_proba
    
    # Apply optimized threshold
    optimal_threshold = 0.42  # Adjusted based on validation
    predictions = (final_proba >= optimal_threshold).astype(int)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_data['id'],
        'Accept': predictions
    })
    
    # Save predictions
    output_path = 'submissions/submit.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main() 