import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import SoccerPredictionModel

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

def train_model(model_type='catboost'):
    """Train the soccer prediction model"""
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
        options=['catboost', 'random_forest', 'gradient_boosting', 'xgboost', 'stacking'],
        index=0,
        help="Choose the machine learning model to train"
    )
    
    if st.sidebar.button("Train Model", type="primary"):
        if train_model(model_type):
            st.sidebar.success(f"{model_type.title()} model trained successfully!")
        else:
            st.sidebar.error(f"Failed to train {model_type} model")
    
    if st.session_state.model_trained:
        current_model = st.session_state.get('model_type', 'unknown')
        st.sidebar.success(f"{current_model.title()} model is ready!")
        if st.session_state.model_metrics:
            st.sidebar.metric("Accuracy", f"{st.session_state.model_metrics['accuracy']:.2f}")
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
                1. **Train the Model** (sidebar) - Choose your preferred ML algorithm
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
                st.set_page_config(layout="wide") 
                st.dataframe(sample_matches, use_container_width=True)
    
    

if __name__ == "__main__":
    main()