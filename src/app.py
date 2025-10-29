import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import SoccerPredictionModel
import torch
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Soccer Match Prediction",
    page_icon="⚽",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'mlp_encoders' not in st.session_state:
    st.session_state.mlp_encoders = None

def load_cleaned_data():
    """Load cleaned data for model training using the model's data loading method"""
    try:
        # Use the model's load_data method to ensure consistent preprocessing
        model = SoccerPredictionModel()
        X_train, X_test, y_train, y_test = model.load_data()
        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def load_raw_data():
    """Load raw data for match selection"""
    try:
        # Load raw data first to get Date column
        X_test_full = pd.read_csv("data/X_test.csv", index_col=0)
        y_test = pd.read_csv("data/y_test.csv", index_col=0).squeeze()
        
        # Create match information dataframe from raw data
        matches = X_test_full[['Date', 'HomeTeam', 'AwayTeam']].copy()
        matches['Date'] = pd.to_datetime(matches['Date'])
        matches['Match'] = matches['HomeTeam'] + ' vs ' + matches['AwayTeam']
        matches['Actual_Result'] = y_test.map({0: 'Away Win', 1: 'Draw', 2: 'Home Win'})
        
        # Load preprocessed data for predictions
        model = SoccerPredictionModel()
        X_train, X_test, y_train, y_test = model.load_data()
        
        return matches, X_test, y_test
    except Exception as e:
        st.error(f"Error loading match data: {str(e)}")
        return None, None, None

def load_mlp_model(X_train, X_test, y_test):
    """Load saved MLP model and prepare for predictions - reuses model preprocessing"""
    from base_trainer import BaseTrainer
    from model_configs import get_categorical_features
    
    # Use same feature set as other models
    feature_set = "odds_form_teams"
    feat_cats = get_categorical_features(feature_set)
    
    # Prepare features using the same method as other models
    model = SoccerPredictionModel()
    X_train_prep, X_test_prep = model.prepare_features(X_train, X_test, feature_set)
    
    # Encode categoricals using BaseTrainer (same as other models)
    trainer = BaseTrainer(categorical_features=feat_cats, random_state=42)
    X_train_prep, X_test_prep = trainer.convert_categorical_to_strings(X_train_prep, X_test_prep)
    X_train_prep, X_test_prep = trainer.encode_categorical_features(
        X_train_prep, X_test_prep, fit_on_combined=False 
    )
    encoders = trainer.label_encoders
    
    # Load model
    model_path = Path("models/mlp_model.ts")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.absolute()}")
    
    try:
        # Set torch to single-threaded mode to avoid potential segfaults in Streamlit
        torch.set_num_threads(1)
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load TorchScript model: {e}")
    
    # Wrapper for predictions
    class MLPWrapper:
        def __init__(self, model, encoders, feature_columns, feat_cats):
            self.model = model
            self.encoders = encoders
            self.feature_columns = feature_columns
            self.feat_cats = feat_cats
        
        def predict(self, X):
            X_prep = self._prepare_data(X)
            X_tensor = torch.tensor(X_prep.values, dtype=torch.float32)
            with torch.no_grad():
                preds = self.model(X_tensor).argmax(dim=1)
            return preds.numpy()
        
        def predict_proba(self, X):
            X_prep = self._prepare_data(X)
            X_tensor = torch.tensor(X_prep.values, dtype=torch.float32)
            with torch.no_grad():
                probs = torch.softmax(self.model(X_tensor), dim=1)
            return probs.numpy()
        
        def _prepare_data(self, X):
            """Prepare data for prediction - reuses model preprocessing"""
            if isinstance(X, pd.Series):
                X = X.to_frame().T
            
            # Use model's prepare_features to select and align features
            model_temp = SoccerPredictionModel()
            # Create dummy test data for prepare_features
            X_dummy_test = pd.DataFrame(columns=self.feature_columns)
            X_prep, _ = model_temp.prepare_features(X, X_dummy_test, "odds_form_teams")
            
            # Ensure exact column order
            X_prep = X_prep.reindex(columns=self.feature_columns, fill_value=0.0)
            
            # Encode categoricals using stored encoders (same as training)
            for c in self.feat_cats:
                if c in X_prep.columns and c in self.encoders:
                    le = self.encoders[c]
                    X_prep[c] = X_prep[c].astype(str)
                    # Handle unknown categories - map to first class
                    known = set(le.classes_)
                    X_prep.loc[~X_prep[c].isin(known), c] = le.classes_[0]
                    X_prep[c] = le.transform(X_prep[c])
            
            return X_prep
    
    mlp_wrapper = MLPWrapper(model, encoders, X_train_prep.columns.tolist(), feat_cats)
    
    # Evaluate model accuracy
    try:
        y_pred = mlp_wrapper.predict(X_test_prep)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        # If evaluation fails, set accuracy to None but still return the wrapper
        # The model can still be used for predictions even if evaluation fails
        accuracy = None
        import warnings
        warnings.warn(f"Could not evaluate model accuracy: {e}")
    
    return mlp_wrapper, {'accuracy': accuracy}

def train_model(model_type='catboost'):
    """Train the soccer prediction model (or load MLP)"""
    if model_type == 'mlp':
        with st.spinner("Loading MLP model..."):
            try:
                X_train, X_test, y_train, y_test = load_cleaned_data()
                if X_train is None:
                    return False
                
                model, metrics = load_mlp_model(X_train, X_test, y_test)
                
                st.session_state.model = model
                st.session_state.model_trained = True
                st.session_state.model_metrics = metrics
                st.session_state.model_type = model_type
                
                return True
            except Exception as e:
                st.error(f"Error loading MLP model: {str(e)}")
                return False
    else:
        with st.spinner(f"Training {model_type} model... This may take a few minutes."):
            try:
                X_train, X_test, y_train, y_test = load_cleaned_data()
                if X_train is None:
                    return False
                
                model = SoccerPredictionModel(model_type=model_type)
                model.train(X_train, y_train, X_test, y_test, 
                           feature_set="odds_form_teams", apply_smote=True)
                
                metrics = model.evaluate(X_test, y_test)
                
                st.session_state.model = model
                st.session_state.model_trained = True
                st.session_state.model_metrics = metrics
                st.session_state.model_type = model_type
                
                return True
            except Exception as e:
                st.error(f"Error training {model_type} model: {str(e)}")
                return False

def main():
    st.title("⚽ Soccer Match Prediction System")
    
    # Sidebar
    st.sidebar.title("Control Panel")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        options=['catboost', 'random_forest', 'gradient_boosting', 'xgboost', 'stacking', 'mlp'],
        index=0,
        help="Choose the machine learning model to train"
    )
    
    button_label = "Load Model" if model_type == 'mlp' else "Train Model"
    if st.sidebar.button(button_label, type="primary"):
        if train_model(model_type):
            action = "loaded" if model_type == 'mlp' else "trained"
            st.sidebar.success(f"{model_type.upper()} model {action} successfully!")
        else:
            action = "load" if model_type == 'mlp' else "train"
            st.sidebar.error(f"Failed to {action} {model_type} model")
    
    if st.session_state.model_trained:
        current_model = st.session_state.get('model_type', 'unknown')
        st.sidebar.success(f"{current_model.title()} model is ready!")
        if st.session_state.model_metrics and st.session_state.model_metrics.get('accuracy') is not None:
            acc = st.session_state.model_metrics['accuracy']
            if isinstance(acc, (int, float)):
                st.sidebar.metric("Accuracy", f"{acc:.2f}")
            else:
                st.sidebar.info(f"Accuracy: {acc}")
    else:
        st.sidebar.warning("Model not trained yet")
    
    # Main content
    tab1, tab2 = st.tabs(["Home", "⚽ Match Prediction"])

    with tab1:
        st.header("Welcome to Soccer Match Prediction System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("What This System Does")
            st.write("""
            This ML system predicts soccer match outcomes using multiple algorithms and advanced feature engineering.
            """)
        
        with col2:
            st.subheader("How to Use")
            st.write("""
            1. **Train/Load Model** (sidebar) - Choose your preferred ML algorithm (MLP is pre-trained)
            2. **Select & Predict** (Match Prediction tab) - Choose real matches to predict
            """)
        
        if st.session_state.model_trained:
            st.success("Model is ready for predictions!")
        else:
            st.info("Please train the model first")
    
    with tab2:
        st.header("⚽ Match Prediction")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first using the sidebar")
        else:
            # Load match data
            matches, X_test, y_test = load_raw_data()
            
            if matches is not None:
                st.subheader("Select a Match to Predict")
                
                # Match selection
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create a searchable selectbox
                    match_options = [f"{row['Match']} ({row['Date'].strftime('%Y-%m-%d')})" 
                                   for idx, row in matches.iterrows()]
                    selected_match = st.selectbox(
                        "Choose a match:",
                        options=match_options,
                        help="Select a real match from the test dataset"
                    )
                
                with col2:
                    # Show total matches available
                    st.metric("Available Matches", len(matches))
                
                if selected_match:
                    # Extract match index
                    match_idx = match_options.index(selected_match)
                    selected_row = matches.iloc[match_idx]
                    
                    # Display match details
                    st.subheader("Match Details")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info(f"**Home Team:** {selected_row['HomeTeam']}")
                    with col2:
                        st.info(f"**Away Team:** {selected_row['AwayTeam']}")
                    with col3:
                        st.info(f"**Date:** {selected_row['Date'].strftime('%Y-%m-%d')}")
                    
                    # Make prediction
                    if st.button("🔮 Predict Match Outcome", type="primary"):
                        try:
                            # Get the corresponding test data row
                            test_row = X_test.iloc[[match_idx]].copy()
                            
                            # Make prediction
                            prediction_result = st.session_state.model.predict(test_row)
                            probabilities_result = st.session_state.model.predict_proba(test_row)
                            
                            # Ensure we get scalar values
                            if isinstance(prediction_result, np.ndarray):
                                prediction = int(prediction_result.item())
                            else:
                                prediction = int(prediction_result)
                            
                            if isinstance(probabilities_result, np.ndarray):
                                probabilities = probabilities_result[0]
                            else:
                                probabilities = probabilities_result
                            
                            # Convert prediction to readable format
                            result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
                            predicted_result = result_map[prediction]
                            actual_result = selected_row['Actual_Result']
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Predicted Result", predicted_result)
                            with col2:
                                st.metric("Actual Result", actual_result)
                            with col3:
                                # Check if prediction is correct
                                is_correct = predicted_result == actual_result
                                st.metric("Correct?", "✅ Yes" if is_correct else "❌ No")
                            
                            # Display probabilities
                            st.subheader("Prediction Probabilities")
                            prob_col1, prob_col2, prob_col3 = st.columns(3)
                            
                            with prob_col1:
                                st.metric("Home Win", f"{probabilities[2]:.1%}")
                            with prob_col2:
                                st.metric("Draw", f"{probabilities[1]:.1%}")
                            with prob_col3:
                                st.metric("Away Win", f"{probabilities[0]:.1%}")
                            
                            # Visualize probabilities
                            prob_data = pd.DataFrame({
                                'Outcome': ['Home Win', 'Draw', 'Away Win'],
                                'Probability': [probabilities[2], probabilities[1], probabilities[0]]
                            })
                            
                            fig = px.bar(prob_data, x='Outcome', y='Probability', 
                                        title="Prediction Probabilities",
                                        color='Probability',
                                        color_continuous_scale='RdYlGn')
                            fig.update_layout(yaxis_tickformat='.1%')
                            st.plotly_chart(fig, width='content')
                            
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                
                # Show some sample matches
                st.subheader("Sample Available Matches")
                sample_matches = matches.head(10)[['Date', 'Match', 'Actual_Result']].copy()
                sample_matches['Date'] = sample_matches['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(sample_matches, use_container_width=True)
    
    

if __name__ == "__main__":
    main()