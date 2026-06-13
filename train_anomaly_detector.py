import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def train_and_score():
    print("Loading datasets...")
    if not os.path.exists('data/transactions.csv'):
        print("Error: data/transactions.csv not found. Please run generate_data.py first.")
        return
        
    df_tx = pd.read_csv('data/transactions.csv')
    df_tx['timestamp'] = pd.to_datetime(df_tx['timestamp'])
    
    print("Performing feature engineering...")
    # 1. Historical account baseline deviations
    user_stats = df_tx.groupby('sender_id')['amount'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['sender_id', 'user_mean_amount', 'user_std_amount']
    df_tx = df_tx.merge(user_stats, on='sender_id', how='left')
    
    # Amount deviation from user mean (fill NaNs with 0 in case user has only 1 transaction)
    df_tx['amount_deviation'] = df_tx['amount'] - df_tx['user_mean_amount']
    
    # 2. Time-based rolling windows (Velocity features)
    df_tx = df_tx.sort_values('timestamp').reset_index(drop=True)
    
    # Rolling counts and sums (merging on sender_id and timestamp to ensure alignment)
    df_indexed = df_tx.set_index('timestamp')
    
    # 10 minutes rolling count
    rolling_10m = df_indexed.groupby('sender_id')['amount'].rolling('10min').count().reset_index()
    rolling_10m.rename(columns={'amount': 'rolling_count_10m'}, inplace=True)
    
    # 1 hour rolling sum
    rolling_1h_sum = df_indexed.groupby('sender_id')['amount'].rolling('1h').sum().reset_index()
    rolling_1h_sum.rename(columns={'amount': 'rolling_sum_1h'}, inplace=True)
    
    # 1 hour rolling count
    rolling_1h_count = df_indexed.groupby('sender_id')['amount'].rolling('1h').count().reset_index()
    rolling_1h_count.rename(columns={'amount': 'rolling_count_1h'}, inplace=True)
    
    # Merge rolling features back
    df_tx = df_tx.merge(rolling_10m, on=['sender_id', 'timestamp'], how='left')
    df_tx = df_tx.merge(rolling_1h_sum, on=['sender_id', 'timestamp'], how='left')
    df_tx = df_tx.merge(rolling_1h_count, on=['sender_id', 'timestamp'], how='left')
    
    # Fill any NaNs resulting from merges
    df_tx['rolling_count_10m'] = df_tx['rolling_count_10m'].fillna(1)
    df_tx['rolling_sum_1h'] = df_tx['rolling_sum_1h'].fillna(df_tx['amount'])
    df_tx['rolling_count_1h'] = df_tx['rolling_count_1h'].fillna(1)
    
    # 3. Temporal & Categorical context
    df_tx['hour'] = df_tx['timestamp'].dt.hour
    # Night transactions (11 PM to 5 AM) often carry higher risk
    df_tx['is_night'] = ((df_tx['hour'] >= 23) | (df_tx['hour'] <= 5)).astype(int)
    
    # Location risk binary encoding (UNKNOWN/INT_OFFSHORE carry risk)
    df_tx['location_risk'] = df_tx['location'].apply(lambda x: 1 if x in ['UNKNOWN', 'INT_OFFSHORE'] else 0)
    
    # Define features for training
    feature_cols = [
        'amount', 
        'amount_deviation', 
        'rolling_count_10m', 
        'rolling_sum_1h', 
        'rolling_count_1h', 
        'is_night', 
        'location_risk'
    ]
    
    X = df_tx[feature_cols].copy()
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Isolation Forest Anomaly Detector...")
    # Set contamination to 1.5% to capture the injected fraud + slightly more
    model = IsolationForest(contamination=0.015, random_state=42)
    model.fit(X_scaled)
    
    # Predict (-1: anomaly, 1: normal)
    df_tx['is_anomaly'] = (model.predict(X_scaled) == -1).astype(int)
    
    # Calculate continuous risk score (0 to 100)
    decision_scores = model.decision_function(X_scaled)
    min_score, max_score = decision_scores.min(), decision_scores.max()
    risk_scores = 100 * (max_score - decision_scores) / (max_score - min_score)
    df_tx['risk_score'] = np.round(risk_scores, 1)
    
    print("Calculating Explainable AI feature contributions...")
    # Average profile of normal transactions
    normal_profile = X_scaled[df_tx['is_anomaly'] == 0].mean(axis=0)
    
    # For each transaction, calculate contribution based on feature deviation from normal
    contributions = []
    for i in range(len(df_tx)):
        deviations = np.abs(X_scaled[i] - normal_profile)
        total_dev = deviations.sum()
        if total_dev == 0:
            total_dev = 1.0
        contrib_dict = {}
        for idx, feat in enumerate(feature_cols):
            contrib_dict[f"contrib_{feat}"] = np.round(deviations[idx] / total_dev, 3)
        contributions.append(contrib_dict)
        
    df_contrib = pd.DataFrame(contributions)
    df_tx = pd.concat([df_tx, df_contrib], axis=1)
    
    # Format timestamp back to string for easier CSV saving
    df_tx['timestamp'] = df_tx['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 4. Save Artifacts
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/isolation_forest.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    df_tx.to_csv('data/scored_transactions.csv', index=False)
    
    print("\nModel Training and Scoring Complete!")
    print(f"Total Transactions Scored: {len(df_tx)}")
    print(f"Total Anomalies Flagged: {len(df_tx[df_tx['is_anomaly'] == 1])}")
    print(f"Recall on Injected Fraud: {df_tx[(df_tx['is_fraud'] == 1) & (df_tx['is_anomaly'] == 1)].shape[0]} / {df_tx[df_tx['is_fraud'] == 1].shape[0]} ({df_tx[(df_tx['is_fraud'] == 1) & (df_tx['is_anomaly'] == 1)].shape[0] / df_tx[df_tx['is_fraud'] == 1].shape[0] * 100:.1f}%)")
    print("Saved scored transactions to 'data/scored_transactions.csv'.")
    print("Saved model artifacts to 'models/' directory.")

if __name__ == "__main__":
    train_and_score()
