import os
from datetime import datetime
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def generate_suspicious_activity_report(tx_details, network_context, explainability_factors):
    """
    Generates a formal SAR Memo using the Gemini API, 
    with a realistic mock fallback if the API key is not configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Check if API key is valid or placeholder
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Gemini API key not configured. Generating mock compliance report...")
        return generate_mock_report(tx_details, network_context, explainability_factors)
        
    try:
        from google import genai
        # Initialize client (looks for GEMINI_API_KEY in environment)
        client = genai.Client()
        
        prompt = f"""
You are a Certified Anti-Money Laundering Specialist (CAMS) and senior compliance auditor.
Analyze the following flagged financial transaction and generate a professional, audit-ready **Suspicious Activity Report (SAR) Memo**.

--- TARGET TRANSACTION DETAILS ---
Transaction ID: {tx_details['transaction_id']}
Timestamp: {tx_details['timestamp']}
Sender Account: {tx_details['sender_id']} (Name: {tx_details['sender_name']}, Account Type: {tx_details.get('sender_type', 'Checking')}, Balance: ${tx_details.get('sender_balance', 0.0):,.2f})
Receiver Account: {tx_details['receiver_id']} (Name: {tx_details['receiver_name']})
Amount: ${tx_details['amount']:,.2f}
Type: {tx_details['transaction_type']}
Location: {tx_details['location']}
ML Risk Score: {tx_details['risk_score']}/100

--- ML ANOMALY EXPLANATION (XAI) ---
The unsupervised anomaly detector flagged this based on the following feature deviations:
{explainability_factors}

--- TRANSACTION NETWORK CONTEXT ---
{network_context}

--- MEMO FORMAT REQUIREMENTS ---
Write the report in a highly professional, objective regulatory tone. Use the following Markdown template structure:

# SUSPICIOUS ACTIVITY REPORT (SAR) COMPLIANCE MEMO
**Case ID:** SAR-{tx_details['transaction_id']}  
**Date of Investigation:** {datetime.now().strftime('%Y-%m-%d')}  
**Investigator Status:** Automated Fraud Engine & AI Agent  

## 1. EXECUTIVE SUMMARY
Provide a concise 2-3 sentence summary of the flagged risk, the primary actor, and the recommended action.

## 2. CLIENT & TRANSACTION PROFILE
Detail the sender, receiver, and transaction parameters. Discuss historical anomalies or changes in behavior.

## 3. ANOMALY ANALYSIS & MODEL EXPLANATIONS
Explain why the Isolation Forest model flagged this. Refer to the specific feature deviations (e.g., rapid velocity, large amounts relative to history, night transfers, location risk).

## 4. NETWORK & GRAPH RELATIONS
Discuss the relationships between the accounts. Highlight if this transaction belongs to a larger structured loop, multi-depositor hub, or money mule layout.

## 5. COMPLIANCE RECOMMENDATIONS & NEXT STEPS
Provide actionable steps for the compliance team (e.g., temporary lock, request source of funds documentation, submit SAR to FinCEN, clear false positive).

Keep the writing dense, realistic, and audit-ready. Do not use conversational filler or meta-remarks.
"""
        
        # Use gemini-2.5-flash as the standard model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}. Falling back to mock report...")
        return generate_mock_report(tx_details, network_context, explainability_factors, error_msg=str(e))

def generate_mock_report(tx_details, network_context, explainability_factors, error_msg=None):
    """
    Generates highly realistic, domain-specific mock reports matching the fraud types
    to ensure the dashboard behaves beautifully out-of-the-box.
    """
    fraud_type = tx_details.get('fraud_type', 'None').lower()
    tx_id = tx_details['transaction_id']
    sender = tx_details['sender_id']
    sender_name = tx_details['sender_name']
    receiver = tx_details['receiver_id']
    receiver_name = tx_details['receiver_name']
    amount = tx_details['amount']
    timestamp = tx_details['timestamp']
    risk_score = tx_details['risk_score']
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    header = f"""# SUSPICIOUS ACTIVITY REPORT (SAR) COMPLIANCE MEMO
**Case ID:** SAR-{tx_id}  
**Date of Investigation:** {current_date}  
**Investigator Status:** Automated Fraud Engine (MOCK FALLBACK MODE)  

"""
    if error_msg:
        header = f"> [!NOTE]\n> The Gemini API call failed or is not configured. Displaying a simulated compliance report based on the transaction metadata.\n\n" + header
    else:
        header = f"> [!TIP]\n> Displaying simulated compliance report. To use live AI generation, configure `GEMINI_API_KEY` in your `.env` file.\n\n" + header

    if 'velocity' in fraud_type:
        return header + f"""## 1. EXECUTIVE SUMMARY
Subject account `{sender}` ({sender_name}) was flagged for extreme transaction frequency (velocity attack). Within a 10-minute window, the account initiated 15 consecutive low-value outbound transfers. Immediate card suspension and account locking are recommended to prevent further exposure.

## 2. CLIENT & TRANSACTION PROFILE
*   **Sender:** {sender_name} (`{sender}`)
*   **Target Transaction:** Transfer of ${amount:.2f} to {receiver_name} (`{receiver}`) at {timestamp}.
*   **Historical Profile:** The client typically conducts 1-2 transactions daily, with a mean transaction value of ~$30.00. The sudden spike to 15 transactions represents a severe deviation.

## 3. ANOMALY ANALYSIS & MODEL EXPLANATIONS
The Isolation Forest model outputted a high risk score of **{risk_score}/100** due to:
*   **rolling_count_10m:** Scaled deviation shows frequency is 15x normal user threshold.
*   **location_risk:** Transactions routed through UNKNOWN location gateways, a common marker for automated card-testing bots.
*   **amount:** The individual transaction values ($5.00 - $15.00) are small, typical of card validation behavior by malicious actors prior to larger scale cash-outs.

## 4. NETWORK & GRAPH RELATIONS
Graph analysis shows the sender account (`{sender}`) attempting to broadcast funds to multiple separate destinations (`{receiver}` and others) in parallel. This is not a standard hub-and-spoke relationship but rather a scatter-shot testing pattern. No incoming structured links were observed.

## 5. COMPLIANCE RECOMMENDATIONS & NEXT STEPS
1.  **Restrict Account:** Place a temporary debit-freeze on account `{sender}` immediately.
2.  **Contact Client:** Reach out to {sender_name} to verify if their card details or login credentials have been compromised.
3.  **Flag Recipient:** Add the recipient accounts to the bank's internal watchlist.
"""

    elif 'large_deviation' in fraud_type or amount > 10000:
        return header + f"""## 1. EXECUTIVE SUMMARY
Subject account `{sender}` ({sender_name}) flagged for a single high-value transfer of ${amount:,.2f} to `{receiver}` ({receiver_name}), representing an unprecedented spike in amount. The funds were routed to an offshore entity. Immediate transaction hold and source-of-wealth validation are recommended.

## 2. CLIENT & TRANSACTION PROFILE
*   **Sender:** {sender_name} (`{sender}`)
*   **Target Transaction:** Outbound wire of ${amount:,.2f} to {receiver_name} (`{receiver}`) at {timestamp}.
*   **Historical Profile:** The account maintains an average balance of ~${tx_details.get('sender_balance', 0.0):,.2f} and typically routes small checking payments. This outbound transfer of ${amount:,.2f} exceeds historical averages by multiple standard deviations.

## 3. ANOMALY ANALYSIS & MODEL EXPLANATIONS
The anomaly model calculated a risk score of **{risk_score}/100** driven by:
*   **amount_deviation:** The amount represents a extreme deviation from the user's mean spending profile.
*   **location_risk:** The transaction destination is categorized as `INT_OFFSHORE`, indicating cross-border risk.
*   **is_night:** The transaction occurred during off-hours, reducing the likelihood of typical corporate or pre-planned personal wires.

## 4. NETWORK & GRAPH RELATIONS
The transaction graph shows a direct, high-value outbound connection from `{sender}` to the offshore node `{receiver}`. Prior to this transfer, `{sender}` had zero historical interactions with `{receiver}`, representing a cold-start relation with a high-risk jurisdiction.

## 5. COMPLIANCE RECOMMENDATIONS & NEXT STEPS
1.  **Hold Transfer:** Place the transaction on administrative hold under AML Safe Harbor provisions.
2.  **Request Documentation:** Contact {sender_name} and request a "Proof of Source of Funds" (e.g., inheritance, property sale, business invoice).
3.  **Regulatory Filing:** Prepare a standard Form 111 (Suspicious Activity Report) for FinCEN filing if documentation is not provided within 48 hours.
"""

    elif 'mule_ring' in fraud_type:
        is_hub = (sender == 'ACC100' or receiver == 'ACC100')
        role = "Hub (Consolidator)" if is_hub else "Spoke (Depositor)"
        
        return header + f"""## 1. EXECUTIVE SUMMARY
Subject account was flagged as part of a structured money-laundering network (Mule Ring). The network utilizes a "smurfing" structure where multiple accounts (`ACC010` through `ACC080`) deposit structured amounts under the $3,000 regulatory reporting threshold into account `ACC100` (Mary Hernandez), which then consolidates and routes the capital offshore.

## 2. CLIENT & TRANSACTION PROFILE
*   **Account Inspected:** {sender_name} (`{sender}`) acting as **{role}**
*   **Transaction:** Transfer of ${amount:,.2f} between `{sender}` and `{receiver}` at {timestamp}.
*   **Context:** Transactions occur in rapid succession within a 3-hour window, totaling over $23,000 in aggregated structured transfers.

## 3. ANOMALY ANALYSIS & MODEL EXPLANATIONS
The transaction was flagged with a risk score of **{risk_score}/100** due to:
*   **structuring_pattern:** The transaction amount (${amount:,.2f}) is intentionally kept just below the standard threshold.
*   **rolling_count_1h:** The sender account shows a spike in transaction velocity.
*   **location_risk:** The consolidation is followed by a transfer to a high-risk offshore destination (`INT_OFFSHORE`).

## 4. NETWORK & GRAPH RELATIONS
Graph analytics reveals a classic **Money Mule Hub-and-Spoke structure**:
*   **Spokes:** 8 separate accounts transfer similar volumes (~$2,900) to the Hub (`ACC100`) within 180 minutes.
*   **Hub:** Account `ACC100` consolidates the funds and immediately executing a 98% outflow transfer to an offshore target (`ACC099`), leaving a 2% commission. This is a high-confidence graph signature of layering.

## 5. COMPLIANCE RECOMMENDATIONS & NEXT STEPS
1.  **Block Entire Ring:** Freeze all participating accounts (`ACC010` to `ACC080`, `ACC100`, and the receiving account `ACC099`).
2.  **Legal Escalation:** Submit an expedited SAR report to federal financial regulators detailing the structuring ring coordinates.
3.  **Police Report:** Coordinated fraud department outreach to law enforcement regarding potential identity theft on the spoke account holders.
"""

    else:
        return header + f"""## 1. EXECUTIVE SUMMARY
Subject transaction was flagged as a potential outlier by the unsupervised ML model. However, network context and historical analysis suggest this may be a **false positive** due to an occasional high-value personal transaction. Standard review is recommended.

## 2. CLIENT & TRANSACTION PROFILE
*   **Sender:** {sender_name} (`{sender}`)
*   **Transaction:** Outbound wire of ${amount:,.2f} to {receiver_name} (`{receiver}`) at {timestamp}.
*   **ML Risk Score:** {risk_score}/100

## 3. ANOMALY ANALYSIS & MODEL EXPLANATIONS
The Isolation Forest model flagged this transaction due to the amount being higher than average, but other indicators (night hours, transaction velocity, and location risk) remain low/normal.

## 4. NETWORK & GRAPH RELATIONS
The transaction graph shows a standard connection between two long-standing accounts. No cyclic loops or structuring indicators were observed.

## 5. COMPLIANCE RECOMMENDATIONS & NEXT STEPS
1.  **Verification:** Perform a quick review of the client's past checking accounts.
2.  **Resolve Flag:** If the transaction aligns with regular commercial activities, mark the case as "Resolved - False Positive" and clear the flag.
"""
