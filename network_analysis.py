import pandas as pd
import networkx as nx

def build_transaction_graph(df_tx):
    """
    Constructs a directed graph of transactions where nodes are accounts
    and edges represent the transfer of money.
    """
    G = nx.DiGraph()
    
    # Add nodes and edges with attributes
    for _, row in df_tx.iterrows():
        sender = row['sender_id']
        receiver = row['receiver_id']
        amount = row['amount']
        tx_id = row['transaction_id']
        risk = row.get('risk_score', 0)
        
        # Add nodes
        if not G.has_node(sender):
            G.add_node(sender)
        if not G.has_node(receiver):
            G.add_node(receiver)
            
        # Add edge with transaction metadata (using transaction_id as key for multi-edges if needed, 
        # but DiGraph allows one directed edge; we store aggregate stats or latest list of transactions)
        if G.has_edge(sender, receiver):
            # Update existing edge
            G[sender][receiver]['amount'] += amount
            G[sender][receiver]['tx_count'] += 1
            G[sender][receiver]['max_risk'] = max(G[sender][receiver]['max_risk'], risk)
            G[sender][receiver]['tx_ids'].append(tx_id)
        else:
            G.add_edge(sender, receiver, 
                       amount=amount, 
                       tx_count=1, 
                       max_risk=risk,
                       tx_ids=[tx_id])
            
    return G

def detect_mule_accounts(df_tx, min_unique_senders=3):
    """
    Identifies accounts showing potential money-mule activity.
    A mule hub is characterized by:
      1. High In-Degree: Receives money from many distinct senders.
      2. High Outflow: Transfers out a large percentage of received funds.
    """
    # Group to count unique senders per receiver
    receivers = df_tx.groupby('receiver_id').agg(
        unique_senders=('sender_id', 'nunique'),
        total_received=('amount', 'sum')
    ).reset_index()
    
    # Group to find total sent per sender
    senders = df_tx.groupby('sender_id').agg(
        total_sent=('amount', 'sum')
    ).reset_index()
    
    # Merge
    mule_candidates = receivers.merge(senders, left_on='receiver_id', right_on='sender_id', how='left')
    mule_candidates['total_sent'] = mule_candidates['total_sent'].fillna(0)
    
    # Calculate outflow ratio
    mule_candidates['outflow_ratio'] = mule_candidates['total_sent'] / (mule_candidates['total_received'] + 0.01)
    
    # Criteria: receives from multiple sources, and passes along > 80% of it
    potential_mules = mule_candidates[
        (mule_candidates['unique_senders'] >= min_unique_senders) & 
        (mule_candidates['outflow_ratio'] >= 0.8)
    ].copy()
    
    return potential_mules.rename(columns={'receiver_id': 'account_id'})

def get_local_subgraph_edges(df_tx, account_id, hops=1):
    """
    Extracts the transactions involving a specific account and its neighbors up to N hops.
    Returns a dataframe of transactions matching the local network.
    """
    connected_accounts = {account_id}
    
    for _ in range(hops):
        # Find all accounts that sent to or received from our current set of accounts
        current_neighbors = set()
        for acc in connected_accounts:
            senders = df_tx[df_tx['receiver_id'] == acc]['sender_id'].unique()
            receivers = df_tx[df_tx['sender_id'] == acc]['receiver_id'].unique()
            current_neighbors.update(senders)
            current_neighbors.update(receivers)
        connected_accounts.update(current_neighbors)
        
    # Return all transactions where both sender and receiver are in the local set
    df_local = df_tx[
        df_tx['sender_id'].isin(connected_accounts) & 
        df_tx['receiver_id'].isin(connected_accounts)
    ].copy()
    
    return df_local
