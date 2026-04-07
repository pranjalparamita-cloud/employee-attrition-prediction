"""
Complete ML Pipeline for Employee Attrition Prediction
Run this file once to train the model: python model_training.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class AttritionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_and_prepare_data(self, filepath='employee_data.csv'):
        """Load and prepare the dataset"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def feature_engineering(self, df):
        """Create advanced features"""
        print("\nEngineering features...")
        df = df.copy()
        
        # Income & Experience Features
        df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
        df['PromotionDelay'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
        df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
        
        # Engagement & Satisfaction
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                           'RelationshipSatisfaction', 'WorkLifeBalance']
        df['EngagementScore'] = df[satisfaction_cols].mean(axis=1)
        df['LowEngagement'] = (df['EngagementScore'] < 2.5).astype(int)
        
        # Risk Indicators
        df['LongPromotionGap'] = (df['YearsSinceLastPromotion'] > 3).astype(int)
        df['LowSalaryHike'] = (df['PercentSalaryHike'] < df['PercentSalaryHike'].quantile(0.25)).astype(int)
        df['RoleStagnation'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        df['LongCommute'] = (df['DistanceFromHome'] > 20).astype(int)
        
        print(f"✓ Created {8} new features")
        return df
    
    def encode_and_scale(self, df, is_training=True):
        """Encode categorical variables and scale features"""
        df = df.copy()
        
        # Binary encoding
        binary_cols = ['Attrition', 'OverTime', 'Gender']
        for col in binary_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # One-hot encoding for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        
        return df
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate multiple models"""
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, 
                                                   class_weight='balanced', n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                                           max_depth=5, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                    random_state=42, eval_metric='logloss')
        }
        
        results = []
        best_score = 0
        best_model = None
        best_model_name = ""
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            }
            
            results.append(metrics)
            
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall: {metrics['Recall']:.4f}")
            print(f"  F1-Score: {metrics['F1-Score']:.4f}")
            print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
            
            if metrics['ROC-AUC'] > best_score:
                best_score = metrics['ROC-AUC']
                best_model = model
                best_model_name = name
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model_name} (ROC-AUC: {best_score:.4f})")
        print("="*70)
        
        return best_model, best_model_name, pd.DataFrame(results)
    
    def prepare_and_train(self, filepath='employee_data.csv'):
        """Complete training pipeline"""
        print("\n" + "="*70)
        print("EMPLOYEE ATTRITION PREDICTION - ML PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_and_prepare_data(filepath)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode and scale
        df_encoded = self.encode_and_scale(df, is_training=True)
        
        # Separate features and target
        X = df_encoded.drop('Attrition', axis=1)
        y = df_encoded['Attrition']
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"✓ Features: {len(self.feature_names)}")
        
        # Scale features
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        # Handle imbalance with SMOTE
        print("\nApplying SMOTE for class balance...")
        print(f"Original distribution: {pd.Series(y_train).value_counts().to_dict()}")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Train models
        best_model, model_name, results_df = self.train_models(
            X_train_balanced, y_train_balanced, X_test, y_test
        )
        
        self.model = best_model
        
        # Save model and components
        print("\n💾 Saving model and components...")
        joblib.dump(self.model, 'best_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        results_df.to_csv('model_comparison.csv', index=False)
        
        print("✓ Model saved: best_model.pkl")
        print("✓ Scaler saved: scaler.pkl")
        print("✓ Encoders saved: label_encoders.pkl")
        print("✓ Features saved: feature_names.pkl")
        print("✓ Results saved: model_comparison.csv")
        
        # Generate predictions on full dataset
        self.generate_predictions(df, df_encoded)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print("\nNext step: Run the Streamlit app")
        print("Command: streamlit run app.py")
        
        return results_df
    
    def generate_predictions(self, df_original, df_encoded):
        """Generate predictions for all employees"""
        print("\n📊 Generating predictions for all employees...")
        
        X_full = df_encoded.drop('Attrition', axis=1)
        X_full = X_full.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale
        numerical_cols = X_full.select_dtypes(include=['int64', 'float64']).columns
        X_full[numerical_cols] = self.scaler.transform(X_full[numerical_cols])
        
        # Predict
        predictions = self.model.predict_proba(X_full)[:, 1]
        
        # Add to original dataframe
        df_original['AttritionRisk'] = predictions
        df_original['RiskCategory'] = pd.cut(
            predictions, 
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Add employee ID if not present
        if 'EmployeeID' not in df_original.columns:
            df_original['EmployeeID'] = range(1, len(df_original) + 1)
        
        # Save
        df_original.to_csv('employee_predictions.csv', index=False)
        print("✓ Predictions saved: employee_predictions.csv")
        
        # Summary
        print("\n📈 Risk Distribution:")
        print(df_original['RiskCategory'].value_counts())
        print(f"\n🔴 High Risk Employees: {(df_original['RiskCategory'] == 'High').sum()}")
        print(f"🟡 Medium Risk Employees: {(df_original['RiskCategory'] == 'Medium').sum()}")
        print(f"🟢 Low Risk Employees: {(df_original['RiskCategory'] == 'Low').sum()}")

if __name__ == "__main__":
    predictor = AttritionPredictor()
    predictor.prepare_and_train('employee_data.csv')