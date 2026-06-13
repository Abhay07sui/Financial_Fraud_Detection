import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from dotenv import load_dotenv

# Import our custom graph and agent utilities
from network_analysis import detect_mule_accounts, get_local_subgraph_edges
from audit_agent import generate_suspicious_activity_report

# Load env variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Financial Fraud Risk Hub",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom css for premium styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #E53935;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 11px;
        color: #8E8E9F;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: #F8F8F2;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    if not os.path.exists('data/scored_transactions.csv') or not os.path.exists('data/accounts.csv'):
        return None, None
    df_tx = pd.read_csv('data/scored_transactions.csv')
    df_acc = pd.read_csv('data/accounts.csv')
    return df_tx, df_acc

df_tx, df_acc = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/nolan/128/security-shield.png", width=70)
st.sidebar.title("Fraud Risk Hub")
st.sidebar.caption("Unsupervised ML & AI Auditing Agent")
st.sidebar.markdown("---")

# Verify API status
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_gemini_api_key_here":
    st.sidebar.warning("🔑 Gemini API: Mock Mode")
    st.sidebar.info("To run live AI audits, configure your API Key in the `.env` file.")
else:
    st.sidebar.success("🔑 Gemini API: Active")

st.sidebar.markdown("### System Statistics")
if df_tx is not None:
    st.sidebar.metric("Database Accounts", len(df_acc))
    st.sidebar.metric("Ledger Transactions", len(df_tx))
    st.sidebar.metric("Unsupervised Anomalies", len(df_tx[df_tx['is_anomaly'] == 1]))
else:
    st.sidebar.error("Database status: Not initialized")

# Reset data cache button
if st.sidebar.button("Refresh Ledger Data"):
    st.cache_data.clear()
    st.rerun()

# ----------------- MAIN PANEL -----------------
if df_tx is None or df_acc is None:
    st.title("🛡️ AI-Powered Financial Fraud Risk Hub")
    st.warning("⚠️ Scored ledger data not found. You need to run the data pipeline first!")
    
    st.markdown("""
    ### Initial Setup Instructions
    To run the platform:
    1. Run the synthetic data generator: `python generate_data.py`
    2. Train the anomaly detection model: `python train_anomaly_detector.py`
    3. Refresh this page using the button in the sidebar.
    """)
    st.stop()

# Build account lookup dictionary
accounts_dict = df_acc.set_index('account_id').to_dict(orient='index')

st.title("🛡️ Financial Fraud Detection & Intelligent Auditing Platform")
st.markdown("---")

# KPI Summary Metrics Row
col1, col2, col3, col4 = st.columns(4)

total_vol = df_tx['amount'].sum()
anomaly_count = df_tx['is_anomaly'].sum()
ground_truth_fraud = df_tx['is_fraud'].sum()
detected_ground_truth = df_tx[(df_tx['is_fraud'] == 1) & (df_tx['is_anomaly'] == 1)].shape[0]
catch_rate = (detected_ground_truth / ground_truth_fraud) * 100 if ground_truth_fraud > 0 else 100.0

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #6200EE;">
        <div class="metric-title">TOTAL TRANSACTION VOLUME</div>
        <div class="metric-value">${total_vol:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #E53935;">
        <div class="metric-title">TOTAL ANOMALIES FLAGGED</div>
        <div class="metric-value">{anomaly_count} <span style="font-size:12px;color:#8E8E9F;">({anomaly_count/len(df_tx)*100:.1f}% rate)</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #FB8C00;">
        <div class="metric-title">POTENTIAL MULE ACCOUNTS</div>
        <div class="metric-value">{len(detect_mule_accounts(df_tx))} accounts</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #4CAF50;">
        <div class="metric-title">ML MODEL CATCH RATE (RECALL)</div>
        <div class="metric-value">{catch_rate:.1f}% <span style="font-size:12px;color:#8E8E9F;">({detected_ground_truth}/{ground_truth_fraud})</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tab separation: Ledger, Graph Mule Rings
tab1, tab2 = st.tabs(["📊 Transaction Risk Ledger", "🕸️ Mule Ring Network Explorer"])

with tab1:
    st.subheader("Interactive Transaction Risk Ledger")
    st.markdown("Filter, sort, and select transactions for deep forensic investigation.")
    
    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All Transactions", "High Risk (Score > 75)", "Medium Risk (Score 40-75)", "Low Risk (Score < 40)"]
        )
    with f_col2:
        fraud_type_filter = st.selectbox(
            "Filter by Fraud Type Pattern",
            ["All Patterns", "velocity", "large_deviation", "mule_ring", "None"]
        )
    with f_col3:
        search_query = st.text_input("Search Account ID (Sender/Receiver)")
        
    # Apply filters
    filtered_df = df_tx.copy()
    
    if risk_filter == "High Risk (Score > 75)":
        filtered_df = filtered_df[filtered_df['risk_score'] > 75]
    elif risk_filter == "Medium Risk (Score 40-75)":
        filtered_df = filtered_df[(filtered_df['risk_score'] >= 40) & (filtered_df['risk_score'] <= 75)]
    elif risk_filter == "Low Risk (Score < 40)":
        filtered_df = filtered_df[filtered_df['risk_score'] < 40]
        
    if fraud_type_filter != "All Patterns":
        filtered_df = filtered_df[filtered_df['fraud_type'] == fraud_type_filter]
        
    if search_query:
        filtered_df = filtered_df[
            (filtered_df['sender_id'].str.contains(search_query, case=False)) | 
            (filtered_df['receiver_id'].str.contains(search_query, case=False))
        ]
        
    # Sort by risk score default descending
    filtered_df = filtered_df.sort_values(by='risk_score', ascending=False)
    
    # Format table for display
    display_df = filtered_df[[
        'transaction_id', 'timestamp', 'sender_id', 'receiver_id', 
        'amount', 'transaction_type', 'location', 'risk_score', 'is_anomaly'
    ]].copy()
    display_df.rename(columns={
        'transaction_id': 'Transaction ID',
        'timestamp': 'Timestamp',
        'sender_id': 'Sender ID',
        'receiver_id': 'Receiver ID',
        'amount': 'Amount ($)',
        'transaction_type': 'Type',
        'location': 'Location',
        'risk_score': 'Risk Score',
        'is_anomaly': 'Flagged Anomaly'
    }, inplace=True)
    
    st.dataframe(
        display_df.style.background_gradient(subset=['Risk Score'], cmap='Reds', vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True
    )
    
    # Select Transaction to Investigate
    st.markdown("### 🔍 Forensic Case Investigation Panel")
    flagged_ids = filtered_df['transaction_id'].tolist()
    if not flagged_ids:
        st.info("No transactions match the selected filters.")
    else:
        # Default to the highest risk item
        selected_tx_id = st.selectbox("Select a Transaction ID to investigate", flagged_ids)
        
        # Load transaction details
        tx_row = df_tx[df_tx['transaction_id'] == selected_tx_id].iloc[0]
        sender_id = tx_row['sender_id']
        receiver_id = tx_row['receiver_id']
        
        sender_info = accounts_dict.get(sender_id, {"customer_name": "Unknown", "account_type": "Unknown", "balance": 0.0})
        receiver_info = accounts_dict.get(receiver_id, {"customer_name": "Unknown", "account_type": "Unknown", "balance": 0.0})
        
        # 2 Column Deep Investigation
        i_col1, i_col2 = st.columns([1, 1])
        
        with i_col1:
            st.markdown(f"#### Account Network & Explainable AI for `{selected_tx_id}`")
            
            # Draw XAI Feature Contribution
            feat_contribs = {
                'Amount': tx_row['contrib_amount'],
                'Amount Deviation': tx_row['contrib_amount_deviation'],
                '10m Tx Count': tx_row['contrib_rolling_count_10m'],
                '1h Tx Vol': tx_row['contrib_rolling_sum_1h'],
                '1h Tx Count': tx_row['contrib_rolling_count_1h'],
                'Night Tx': tx_row['contrib_is_night'],
                'Location Risk': tx_row['contrib_location_risk']
            }
            
            # Sort contribution
            sorted_contribs = sorted(feat_contribs.items(), key=lambda x: x[1])
            labels = [x[0] for x in sorted_contribs]
            values = [x[1] for x in sorted_contribs]
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            # Colors based on significance
            colors = ['#1E88E5' if val < 0.2 else '#E53935' for val in values]
            
            bars = ax.barh(labels, values, color=colors)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#8E8E9F')
            ax.spines['bottom'].set_color('#8E8E9F')
            ax.tick_params(colors='#8E8E9F', labelsize=8)
            ax.set_title("Model Anomaly Feature Contributions", color='#F8F8F2', fontsize=10, fontweight='bold')
            ax.set_xlabel("Contribution Weight", color='#8E8E9F', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Draw local subgraph
            st.markdown("**Local Transaction Network (1-Hop Neighbors)**")
            df_local = get_local_subgraph_edges(df_tx, sender_id, hops=1)
            
            G_local = nx.DiGraph()
            for _, r in df_local.iterrows():
                G_local.add_edge(r['sender_id'], r['receiver_id'], amount=r['amount'])
                
            fig_net, ax_net = plt.subplots(figsize=(6, 4.5))
            fig_net.patch.set_facecolor('#0E1117')
            ax_net.set_facecolor('#0E1117')
            
            pos = nx.spring_layout(G_local, seed=42)
            
            # Define node colors
            node_colors = []
            for node in G_local.nodes():
                if node == sender_id:
                    node_colors.append('#E53935') # Suspect Sender (Red)
                elif node == receiver_id:
                    node_colors.append('#FB8C00') # Direct Recipient (Orange)
                elif node == 'ACC100':
                    node_colors.append('#FFEB3B') # Known Mule Hub ACC100 (Yellow)
                else:
                    node_colors.append('#1E88E5') # Neighbors (Blue)
                    
            nx.draw_networkx_nodes(G_local, pos, node_color=node_colors, node_size=600, ax=ax_net)
            nx.draw_networkx_edges(G_local, pos, arrowstyle="->", arrowsize=12, edge_color='#8E8E9F', width=1.0, ax=ax_net)
            
            # Custom node labels (smaller font)
            labels_dict = {node: node for node in G_local.nodes()}
            nx.draw_networkx_labels(G_local, pos, labels=labels_dict, font_size=7, font_color='#F8F8F2', ax=ax_net)
            
            # Edge labels for amounts
            edge_labels = {(u, v): f"${d['amount']:,.0f}" for u, v, d in G_local.edges(data=True)}
            nx.draw_networkx_edge_labels(G_local, pos, edge_labels=edge_labels, font_size=6, font_color='#8E8E9F', bbox=dict(facecolor='#0E1117', alpha=0.6, edgecolor='none'), ax=ax_net)
            
            ax_net.axis('off')
            plt.tight_layout()
            st.pyplot(fig_net)
            
        with i_col2:
            st.markdown("#### 🕵️ Compliance Auditing Specialist Agent")
            st.markdown("Trigger the AI Agent to review the transaction's temporal, categorical, and network behavior, then draft an audit memo.")
            
            # Context preprocessors for prompt injection
            explainability_str = "\n".join([f"- {feat}: {contrib*100:.1f}%" for feat, contrib in feat_contribs.items() if contrib > 0.05])
            
            # Describe the local network context in structured text
            inbound_txs = df_local[df_local['receiver_id'] == sender_id]
            outbound_txs = df_local[df_local['sender_id'] == sender_id]
            
            network_desc = f"""
- Account '{sender_id}' has a local network of {len(G_local.nodes())} active node connections.
- Outbound transfers from '{sender_id}': {len(outbound_txs)} transactions, totaling ${outbound_txs['amount'].sum():,.2f}.
- Inbound transfers to '{sender_id}': {len(inbound_txs)} transactions, totaling ${inbound_txs['amount'].sum():,.2f}.
"""
            if sender_id == 'ACC100' or receiver_id == 'ACC100':
                network_desc += "\n- CRITICAL WARNING: Account ACC100 (Mary Hernandez) is involved, exhibiting structuring hub dynamics."
                
            tx_details = {
                'transaction_id': selected_tx_id,
                'timestamp': tx_row['timestamp'],
                'sender_id': sender_id,
                'sender_name': sender_info['customer_name'],
                'sender_type': sender_info['account_type'],
                'sender_balance': sender_info['balance'],
                'receiver_id': receiver_id,
                'receiver_name': receiver_info['customer_name'],
                'amount': tx_row['amount'],
                'transaction_type': tx_row['transaction_type'],
                'location': tx_row['location'],
                'risk_score': tx_row['risk_score'],
                'fraud_type': tx_row['fraud_type']
            }
            
            # Session state to hold report text so it doesn't disappear on streamlit reruns
            report_key = f"report_{selected_tx_id}"
            if report_key not in st.session_state:
                st.session_state[report_key] = ""
                
            if st.button("🚀 Draft SAR Audit Report"):
                with st.spinner("Analyzing case data and writing report..."):
                    report = generate_suspicious_activity_report(
                        tx_details=tx_details,
                        network_context=network_desc.strip(),
                        explainability_factors=explainability_str
                    )
                    st.session_state[report_key] = report
            
            if st.session_state[report_key]:
                st.markdown("##### Generated Compliance Memo (Editable)")
                edited_report = st.text_area(
                    label="Edit compliance notes below before exporting:",
                    value=st.session_state[report_key],
                    height=300
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Audit Memo as Markdown",
                    data=edited_report,
                    file_name=f"SAR_Report_{selected_tx_id}.md",
                    mime="text/markdown"
                )
                
                # Render report preview
                st.markdown("---")
                st.markdown("##### Report Preview:")
                st.markdown(edited_report)

with tab2:
    st.subheader("Money Mule Ring Graph Explorer")
    st.markdown("This panel analyzes relational structures across the entire transaction ledger to reveal multi-depositor structuring networks.")
    
    mules_df = detect_mule_accounts(df_tx)
    if mules_df.empty:
        st.info("No accounts met the threshold for high-probability mule rings.")
    else:
        st.dataframe(
            mules_df.style.format({
                'total_received': '${:,.2f}',
                'total_sent': '${:,.2f}',
                'outflow_ratio': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Select Mule to Visualize
        selected_mule = st.selectbox("Select suspicious account to graph", mules_df['account_id'].tolist())
        
        # Plot full ring
        st.markdown(f"#### Network Consolidation Visualizer for `{selected_mule}`")
        df_ring = df_tx[(df_tx['sender_id'] == selected_mule) | (df_tx['receiver_id'] == selected_mule)]
        
        # Fetch related senders/receivers
        neighbor_accounts = set(df_ring['sender_id']).union(set(df_ring['receiver_id']))
        df_ring_full = df_tx[df_tx['sender_id'].isin(neighbor_accounts) & df_tx['receiver_id'].isin(neighbor_accounts)]
        
        G_ring = nx.DiGraph()
        for _, r in df_ring_full.iterrows():
            G_ring.add_edge(r['sender_id'], r['receiver_id'], amount=r['amount'])
            
        fig_r, ax_r = plt.subplots(figsize=(8, 5))
        fig_r.patch.set_facecolor('#0E1117')
        ax_r.set_facecolor('#0E1117')
        
        pos_r = nx.circular_layout(G_ring) # Circular layout makes rings extremely obvious!
        
        node_colors_r = []
        for node in G_ring.nodes():
            if node == selected_mule:
                node_colors_r.append('#E53935') # Suspect Hub (Mary Hernandez ACC100)
            elif node == 'ACC099':
                node_colors_r.append('#4CAF50') # Outflow destination
            else:
                node_colors_r.append('#1E88E5') # Depositor Spokes
                
        nx.draw_networkx_nodes(G_ring, pos_r, node_color=node_colors_r, node_size=700, ax=ax_r)
        nx.draw_networkx_edges(G_ring, pos_r, arrowstyle="->", arrowsize=15, edge_color='#8E8E9F', width=1.2, ax=ax_r)
        
        nx.draw_networkx_labels(G_ring, pos_r, font_size=8, font_color='#F8F8F2', ax=ax_r)
        
        # Edge labels (only draw labels for high value transfers to prevent clutter)
        edge_labels_r = {(u, v): f"${d['amount']:,.0f}" for u, v, d in G_ring.edges(data=True) if d['amount'] > 1000}
        nx.draw_networkx_edge_labels(G_ring, pos_r, edge_labels=edge_labels_r, font_size=6, font_color='#8E8E9F', bbox=dict(facecolor='#0E1117', alpha=0.7, edgecolor='none'), ax=ax_r)
        
        ax_r.axis('off')
        plt.tight_layout()
        st.pyplot(fig_r)
        
        st.markdown("""
        **How to read this Graph:**
        *   **Red Account (`ACC100`)**: The consolidating hub. Note the inbound arrows pointing towards it.
        *   **Blue Accounts (`ACC010` to `ACC080`)**: Senders who deposit cash in structured, smaller volumes.
        *   **Green Account (`ACC099`)**: The external offshore exit account where the accumulated capital is immediately transferred to (outflow destination).
        """)
