import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_fraud_dataset():
    print("Initializing synthetic data generation...")
    
    # 1. Generate Accounts
    num_accounts = 100
    account_ids = [f"ACC{i:03d}" for i in range(1, num_accounts + 1)]
    account_types = ['Savings', 'Checking']
    customer_names = [
        "James Smith", "Michael Smith", "Robert Smith", "Maria Garcia", "David Smith",
        "Maria Rodriguez", "Mary Smith", "Maria Hernandez", "Maria Martinez", "James Johnson",
        "John Smith", "David Johnson", "Robert Johnson", "Michael Johnson", "Richard Smith",
        "William Smith", "Mary Johnson", "Joseph Smith", "Thomas Smith", "Patricia Smith",
        "Charles Smith", "Christopher Smith", "Elizabeth Smith", "Matthew Smith", "Patricia Johnson",
        "Jennifer Smith", "Linda Smith", "Elizabeth Johnson", "Barbara Smith", "Richard Johnson",
        "William Johnson", "Susan Smith", "Joseph Johnson", "Thomas Johnson", "John Johnson",
        "Christopher Johnson", "Charles Johnson", "Jessica Smith", "Jennifer Johnson", "Sarah Smith",
        "Karen Smith", "Elizabeth Taylor", "Nancy Smith", "Lisa Smith", "Betty Smith",
        "Margaret Smith", "Sandra Smith", "Ashley Smith", "Dorothy Smith", "Kimberly Smith",
        "Emily Smith", "Donna Smith", "Michelle Smith", "Carol Smith", "Amanda Smith",
        "Melissa Smith", "Deborah Smith", "Stephanie Smith", "Rebecca Smith", "Laura Smith",
        "Sharon Smith", "Cynthia Smith", "Kathleen Smith", "Amy Smith", "Shirley Smith",
        "Angela Smith", "Helen Smith", "Anna Smith", "Brenda Smith", "Pamela Smith",
        "Nicole Smith", "Samantha Smith", "Katherine Smith", "Christine Smith", "Debra Smith",
        "Rachel Smith", "Carolyn Smith", "Janet Smith", "Catherine Smith", "Maria Smith",
        "Heather Smith", "Diane Smith", "Virginia Smith", "Julie Smith", "Joyce Smith",
        "Victoria Smith", "Olivia Smith", "Kelly Smith", "Christina Smith", "Lauren Smith",
        "Joan Smith", "Evelyn Smith", "Judith Smith", "Megan Smith", "Cheryl Smith",
        "Andrea Smith", "Hannah Smith", "Martha Smith", "Jacqueline Smith", "Frances Smith"
    ]
    
    accounts_data = []
    for acc_id, name in zip(account_ids, customer_names):
        accounts_data.append({
            'account_id': acc_id,
            'customer_name': name,
            'account_type': np.random.choice(account_types, p=[0.4, 0.6]),
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.75, 0.20, 0.05]),
            'balance': round(np.random.uniform(500, 50000), 2)
        })
    
    df_accounts = pd.DataFrame(accounts_data)
    
    # 2. Generate Normal Transactions
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 5, 31)
    date_range_seconds = int((end_date - start_date).total_seconds())
    
    num_normal_tx = 8000
    transactions = []
    
    print(f"Generating {num_normal_tx} normal transactions...")
    for i in range(num_normal_tx):
        # Pick random sender and receiver (must be different)
        sender, receiver = np.random.choice(account_ids, size=2, replace=False)
        
        # Amount: log-normal distribution for realistic transaction values
        amount = round(np.random.lognormal(mean=3.5, sigma=1.0) + 1.0, 2)
        # Cap amount at a reasonable level for normal behavior
        if amount > 1500:
            amount = round(np.random.uniform(100, 500), 2)
            
        # Timestamp
        random_seconds = np.random.randint(0, date_range_seconds)
        timestamp = start_date + timedelta(seconds=random_seconds)
        
        # Check sender's balance (simulate transaction type)
        txn_type = np.random.choice(['Transfer', 'Payment', 'Debit'], p=[0.5, 0.3, 0.2])
        
        # Location
        location = np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'])
        
        transactions.append({
            'transaction_id': f"TXN{i:05d}",
            'timestamp': timestamp,
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': amount,
            'transaction_type': txn_type,
            'location': location,
            'is_fraud': 0,
            'fraud_type': 'None'
        })
        
    # 3. Inject Fraud Scenario 1: Velocity Attack (Card Testing)
    # Target 3 accounts. Rapid succession of small transactions.
    print("Injecting Fraud Scenario 1: Velocity Attack...")
    velocity_accounts = ['ACC012', 'ACC045', 'ACC078']
    txn_id_counter = num_normal_tx
    
    for acc in velocity_accounts:
        # A burst of 15 transactions in 10 minutes
        burst_start = start_date + timedelta(seconds=np.random.randint(0, date_range_seconds - 3600))
        for j in range(15):
            receiver = np.random.choice([a for a in account_ids if a != acc])
            timestamp = burst_start + timedelta(seconds=np.random.randint(0, 600))  # within 10 mins
            amount = round(np.random.uniform(5.00, 15.00), 2)  # small amounts
            
            transactions.append({
                'transaction_id': f"TXN{txn_id_counter:05d}",
                'timestamp': timestamp,
                'sender_id': acc,
                'receiver_id': receiver,
                'amount': amount,
                'transaction_type': 'Payment',
                'location': 'UNKNOWN',  # Card testers often route through unusual gateways
                'is_fraud': 1,
                'fraud_type': 'velocity'
            })
            txn_id_counter += 1

    # 4. Inject Fraud Scenario 2: Large Value Deviation
    # Target 5 accounts. Sudden, massive amounts.
    print("Injecting Fraud Scenario 2: Large Value Deviation...")
    large_val_accounts = ['ACC003', 'ACC028', 'ACC056', 'ACC082', 'ACC095']
    for acc in large_val_accounts:
        receiver = np.random.choice([a for a in account_ids if a != acc])
        timestamp = start_date + timedelta(seconds=np.random.randint(0, date_range_seconds))
        # Massive amount compared to log-normal mean of ~$30
        amount = round(np.random.uniform(15000.00, 28000.00), 2)
        
        transactions.append({
            'transaction_id': f"TXN{txn_id_counter:05d}",
            'timestamp': timestamp,
            'sender_id': acc,
            'receiver_id': receiver,
            'amount': amount,
            'transaction_type': 'Transfer',
            'location': 'INT_OFFSHORE',  # Offshore transfer
            'is_fraud': 1,
            'fraud_type': 'large_deviation'
        })
        txn_id_counter += 1

    # 5. Inject Fraud Scenario 3: Structured Mule Ring
    # Hub: ACC100 (Mary Hernandez)
    # Spokes: ACC010, ACC020, ACC030, ACC040, ACC050, ACC060, ACC070, ACC080
    # Senders transfer $2,800 - $2,950 (below $3,000 threshold) to ACC100 within a 3-hour window
    # Then ACC100 sends the bulk out to an external offshore account or ACC099.
    print("Injecting Fraud Scenario 3: Structured Mule Ring...")
    hub_account = 'ACC100'
    spoke_accounts = ['ACC010', 'ACC020', 'ACC030', 'ACC040', 'ACC050', 'ACC060', 'ACC070', 'ACC080']
    outflow_target = 'ACC099'
    
    ring_start = start_date + timedelta(days=15, hours=10)  # Mid-month
    
    # Spokes transfer to Hub
    for idx, spoke in enumerate(spoke_accounts):
        # Staggered slightly
        timestamp = ring_start + timedelta(minutes=15 * idx + np.random.randint(0, 120))
        amount = round(np.random.uniform(2800.00, 2980.00), 2)  # Structured below $3k
        
        transactions.append({
            'transaction_id': f"TXN{txn_id_counter:05d}",
            'timestamp': timestamp,
            'sender_id': spoke,
            'receiver_id': hub_account,
            'amount': amount,
            'transaction_type': 'Transfer',
            'location': 'NY',
            'is_fraud': 1,
            'fraud_type': 'mule_ring'
        })
        txn_id_counter += 1
        
    # Large Outflow from Hub to target
    outflow_timestamp = ring_start + timedelta(hours=3, minutes=30)
    total_mule_volume = sum([t['amount'] for t in transactions if t['sender_id'] in spoke_accounts and t['receiver_id'] == hub_account and t['fraud_type'] == 'mule_ring'])
    outflow_amount = round(total_mule_volume * 0.98, 2)  # Keeping 2% cut for the hub mule
    
    transactions.append({
        'transaction_id': f"TXN{txn_id_counter:05d}",
        'timestamp': outflow_timestamp,
        'sender_id': hub_account,
        'receiver_id': outflow_target,
        'amount': outflow_amount,
        'transaction_type': 'Transfer',
        'location': 'INT_OFFSHORE',
        'is_fraud': 1,
        'fraud_type': 'mule_ring'
    })
    txn_id_counter += 1

    # Convert to DataFrame, Sort by timestamp to make it realistic
    df_tx = pd.DataFrame(transactions)
    df_tx = df_tx.sort_values(by='timestamp').reset_index(drop=True)
    
    # Format timestamps as string
    df_tx['timestamp'] = df_tx['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 6. Save Data
    os.makedirs('data', exist_ok=True)
    df_accounts.to_csv('data/accounts.csv', index=False)
    df_tx.to_csv('data/transactions.csv', index=False)
    
    print("\nDataset Generation Complete!")
    print(f"Total Accounts: {len(df_accounts)}")
    print(f"Total Transactions: {len(df_tx)}")
    print(f"Normal Transactions: {len(df_tx[df_tx['is_fraud'] == 0])}")
    print(f"Fraudulent Transactions: {len(df_tx[df_tx['is_fraud'] == 1])}")
    print(f"  - Velocity Attacks: {len(df_tx[df_tx['fraud_type'] == 'velocity'])}")
    print(f"  - Large Deviations: {len(df_tx[df_tx['fraud_type'] == 'large_deviation'])}")
    print(f"  - Mule Ring Transfers: {len(df_tx[df_tx['fraud_type'] == 'mule_ring'])}")
    print("Files saved in 'data/' directory.")

if __name__ == "__main__":
    generate_fraud_dataset()
