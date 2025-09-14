import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("--- Starting Step 1: Loading All Datasets ---")

# --- Load Original Datasets (True.csv and Fake.csv) ---
try:
    true_df = pd.read_csv("True.csv")
    true_df['label'] = 'REAL'
    print(f"✅ Loaded {len(true_df)} articles from True.csv")

    fake_df = pd.read_csv("Fake.csv")
    fake_df['label'] = 'FAKE'
    print(f"✅ Loaded {len(fake_df)} articles from Fake.csv")

    # Combine original datasets
    original_df = pd.concat([true_df, fake_df], ignore_index=True)
    # The original data has 'title' and 'text'. We'll combine them.
    original_df['full_text'] = original_df['title'] + ' ' + original_df['text']
    print("✅ Combined original True and Fake datasets.")

except FileNotFoundError as e:
    print(f"🔴 Error: Make sure True.csv and Fake.csv are in the same folder as this script. {e}")
    exit()

# --- Load LIAR Dataset (train.tsv) ---
try:
    # The LIAR dataset is tab-separated (.tsv)
    liar_filepath = os.path.join("liar_dataset", "train.tsv")
    liar_df = pd.read_csv(liar_filepath, sep='\t', header=None)
    
    # Assign column names as per the dataset's description
    liar_df.columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts',
        'false_counts', 'half_true_counts', 'mostly_true_counts',
        'pants_on_fire_counts', 'context'
    ]
    print(f"✅ Loaded {len(liar_df)} statements from the LIAR dataset.")

    # --- Data Cleaning and Mapping for LIAR dataset ---
    # We only need the statement and the label for our purpose
    liar_df = liar_df[['statement', 'label']]
    
    # Map the detailed LIAR labels to our simple FAKE/REAL system
    def map_liar_labels(label):
        if label in ['true', 'mostly-true', 'half-true']:
            return 'REAL'
        else: # 'false', 'barely-true', 'pants-on-fire'
            return 'FAKE'
            
    liar_df['label'] = liar_df['label'].apply(map_liar_labels)
    
    # Rename 'statement' to 'full_text' to match our other dataframe
    liar_df.rename(columns={'statement': 'full_text'}, inplace=True)
    print("✅ Processed and mapped LIAR dataset labels.")

except FileNotFoundError as e:
    print(f"🔴 Error: Could not find the LIAR dataset. Make sure you have unzipped it and the 'liar_dataset' folder (containing 'train.tsv') is in your project folder. {e}")
    exit()

# --- Combine ALL datasets into one master DataFrame ---
final_df = pd.concat([original_df[['full_text', 'label']], liar_df], ignore_index=True)
final_df.dropna(subset=['full_text'], inplace=True) # Remove any rows where the text is empty
print(f"\n--- Master Dataset Created ---")
print(f"Total combined entries: {len(final_df)}")
print("Distribution of labels in the final dataset:")
print(final_df['label'].value_counts())

# Shuffle the dataset to mix the data sources thoroughly
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ Shuffled the master dataset.")

# --- Starting Step 2: Preprocessing and Splitting Data ---
print("\n--- Starting Step 2: Preprocessing and Splitting Data ---")
X = final_df['full_text']
y = final_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split into 80% training and 20% testing sets.")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform on training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# Only transform the test data
tfidf_test = tfidf_vectorizer.transform(X_test)
print("✅ Text data has been converted into numerical vectors.")

# --- Starting Step 3: Training the AI Model ---
print("\n--- Starting Step 3: Training YOUR AI Model ---")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)
print("✅ Model training complete!")

# --- Starting Step 4: Evaluating the New Model ---
print("\n--- Starting Step 4: Evaluating the New Model ---")
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ New Model Accuracy on unseen data: {accuracy:.4f}")

# --- Starting Step 5: Saving the Improved Model ---
print("\n--- Starting Step 5: Saving the Improved Model and Vectorizer ---")
joblib.dump(model, 'fake_news_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("\n✅ Your new, improved model has been saved!")
print("✅ You can now restart your 'api_server.py' to use it.")

