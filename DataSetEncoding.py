
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
# Load CSV

df = pd.read_csv("hajj_umrah_crowd_management_dataset.csv", parse_dates=["Timestamp"])

df.drop("Timestamp", axis=1, inplace=True)

# Define features and target (example)
X = df.drop("Satisfaction_Rating", axis=1)
y = df["Satisfaction_Rating"]

# List of columns
onehot_cols = ['Activity_Type', 'Weather_Conditions', 'AR_System_Interaction', 'Health_Condition', 'Emergency_Event',
               'Age_Group', 'Nationality', 'Transport_Mode', 'Incident_Type', 'Event_Type', 'Pilgrim_Experience', 'Crowd_Morale', 'AR_Navigation_Success']

ordinal_cols = ['Fatigue_Level', 'Stress_Level', 'Crowd_Density']

# ColumnTransformer setup
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
        ("ordinal", OrdinalEncoder(), ordinal_cols)
    ],
    remainder="passthrough"  # Keep the rest of the columns as they are (e.g., numeric)
)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformation
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)



# Get feature names from OneHotEncoder
onehot_feature_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(onehot_cols)

# Ordinal columns already retain their names
ordinal_feature_names = ordinal_cols

# Identify numeric passthrough columns
remainder_cols = [col for col in X.columns if col not in onehot_cols + ordinal_cols]

# Combine all feature names
all_feature_names = np.concatenate([onehot_feature_names, ordinal_feature_names, remainder_cols])

# Convert encoded training data to DataFrame
X_train_df = pd.DataFrame(X_train_encoded, columns=all_feature_names)

# Save to CSV
X_train_df.to_csv("encoded_dataset.csv", index=False)

print("✅ Encoded dataset saved as 'encoded_dataset.csv'")
