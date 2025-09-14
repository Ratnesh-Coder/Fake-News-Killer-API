import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the file paths for the two separate files
real_filepath = 'True.csv'
fake_filepath = 'Fake.csv'

try:
    # --- Step 2: Load, Label, and Combine Data ---
    print("--- Loading, Labeling, and Combining Data ---")
    real_df = pd.read_csv(real_filepath)
    fake_df = pd.read_csv(fake_filepath)
    real_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'
    combined_df = pd.concat([real_df, fake_df], ignore_index=True)
    df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Step 3: Text Preprocessing and Data Splitting ---
    print("--- Preprocessing and Splitting Data ---")
    df['full_text'] = df['title'] + ' ' + df['text']
    df['full_text'] = df['full_text'].fillna('')
    X = df['full_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the vectorizer ONCE on the training data
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # --- Step 4: Training the AI Model ---
    print("--- Training the AI Model ---")
    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    model.fit(tfidf_train, y_train)

    # --- Step 5: Evaluating the Model ---
    print("--- Evaluating the Model ---")
    y_pred = model.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print(conf_matrix)


    # --- Step 6: Making Real-Time Predictions ---
    print("\n--- Starting Step 6: Making Real-Time Predictions ---")

    def predict_news(text, vectorizer, trained_model):
        """
        Takes a string of text and predicts if it's FAKE or REAL.
        """
        # We need to transform the new text using the same vectorizer that was trained on our data
        text_vectorized = vectorizer.transform([text])
        
        # Predict using the trained model
        prediction = trained_model.predict(text_vectorized)
        
        return prediction[0]

    # Example 1: A clearly fake-sounding headline
    news_to_test_1 = "Scientists Discover Dragons Living in the Earth's Core, NASA Confirms. The creatures reportedly breathe fire and subsist on a diet of molten rock."
    
    # Example 2: A more realistic-sounding headline
    news_to_test_2 = "The Federal Reserve is expected to announce its decision on interest rates this Wednesday following a two-day policy meeting."

    # Make predictions
    prediction_1 = predict_news(news_to_test_1, tfidf_vectorizer, model)
    prediction_2 = predict_news(news_to_test_2, tfidf_vectorizer, model)

    print(f"\nNews 1: '{news_to_test_1[:70]}...'")
    print(f"Prediction: {prediction_1}")

    print(f"\nNews 2: '{news_to_test_2[:70]}...'")
    print(f"Prediction: {prediction_2}")


except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure 'True.csv' and 'Fake.csv' are in the same folder as your Python script.")

