# langfuse_data_fetcher.py
import requests
import base64
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='langfuse_data_fetcher.log',
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

# Custom JSON encoder to handle NumPy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def fetch_langfuse_data(days_back=30):
    """
    Main function to fetch Langfuse data and save it to JSON files
    """
    logging.info(f"Starting Langfuse data fetch for the last {days_back} days...")
    print(f"Starting Langfuse data fetch for the last {days_back} days...")
    
    # Set Langfuse host
    base_url = os.getenv("LANGFUSE_HOST", "https://langfuse.zoomrx.ai")
    logging.info(f"Base URL: {base_url}")
    print(f"Base URL: {base_url}")

    # Define all agent configurations
    agent_configs = [
        {
            "name": "HCP_P",
            "public_key": os.getenv("HCP_P_LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("HCP_P_LANGFUSE_SECRET_KEY"),
        },
        {
            "name": "SCOPING",
            "public_key": os.getenv("SCOPING_LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("SCOPING_LANGFUSE_SECRET_KEY"),
        },
        {
            "name": "SYNAPSE",
            "public_key": os.getenv("SYNAPSE_LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("SYNAPSE_LANGFUSE_SECRET_KEY"),
        },
        {
            "name": "HASHTAG",
            "public_key": os.getenv("HASHTAG_LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("HASHTAG_LANGFUSE_SECRET_KEY"),
        },
        {
            "name": "SURVEY_CODING",
            "public_key": os.getenv("SURVEY_CODING_LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("SURVEY_CODING_LANGFUSE_SECRET_KEY"),
        }
    ]

    # Create data directory if it doesn't exist
    os.makedirs("dashboard_data", exist_ok=True)

    # Get date range for specified days back
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Initialize combined data structures
    all_traces = []
    all_sessions = []
    all_observations = []
    all_daily_metrics = []

    # Process data for each agent
    for agent_config in agent_configs:
        agent_name = agent_config["name"]
        public_key = agent_config["public_key"]
        secret_key = agent_config["secret_key"]
        
        if not public_key or not secret_key:
            logging.warning(f"Skipping {agent_name} due to missing API keys")
            print(f"Skipping {agent_name} due to missing API keys")
            continue
            
        logging.info(f"=== Processing {agent_name} Agent ===")
        print(f"\n=== Processing {agent_name} Agent ===")
        
        # Create Basic Auth string
        auth_string = f"{public_key}:{secret_key}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

        # Headers with Basic Auth
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/json"
        }
        
        # 1. Get daily metrics for token and cost data
        try:
            metrics_response = requests.get(
                f"{base_url}/api/public/metrics/daily",
                headers=headers,
                params={
                    "startDate": start_date_str,
                    "endDate": end_date_str
                }
            )
            
            logging.info(f"Daily Metrics Status Code: {metrics_response.status_code}")
            
            if metrics_response.status_code == 200:
                metrics_data = metrics_response.json()
                logging.info(f"Fetched daily metrics for {agent_name}")
                
                # Process metrics data
                for day_data in metrics_data.get('data', []):
                    day_metrics = {
                        'date': day_data.get('date'),
                        'agent': agent_name,
                        'countTraces': day_data.get('countTraces', 0),
                        'countObservations': day_data.get('countObservations', 0),
                        'totalCost': day_data.get('totalCost', 0),
                        'totalTokens': 0  # Initialize token sum
                    }
                    
                    # Sum up tokens from usage data
                    for usage in day_data.get('usage', []):
                        # Add tokens from input and output usage
                        day_metrics['totalTokens'] += usage.get('inputUsage', 0) + usage.get('outputUsage', 0)
                        
                        # Extract model information if available
                        if usage.get('model'):
                            model_name = usage.get('model', '').split('/')[-1] if usage.get('model') else 'Unknown'
                            day_metrics[f'model_{model_name}_tokens'] = usage.get('inputUsage', 0) + usage.get('outputUsage', 0)
                            day_metrics[f'model_{model_name}_cost'] = usage.get('totalCost', 0)
                    
                    all_daily_metrics.append(day_metrics)
                
                # Save agent-specific metrics data
                with open(f"dashboard_data/{agent_name.lower()}_daily_metrics.json", "w") as f:
                    json.dump(metrics_data, f, indent=2, cls=NpEncoder)
                logging.info(f"Saved daily metrics for {agent_name}")
                print(f"Saved daily metrics for {agent_name}")
            else:
                logging.error(f"Failed to fetch daily metrics for {agent_name}: {metrics_response.text}")
                print(f"Failed to fetch daily metrics for {agent_name}: {metrics_response.text}")
        except Exception as e:
            logging.error(f"Error processing metrics for {agent_name}: {e}")
            print(f"Error processing metrics for {agent_name}: {e}")

        # 2. Get traces (conversation level data) with pagination
        try:
            page = 1
            while True:
                traces_response = requests.get(
                    f"{base_url}/api/public/traces",
                    headers=headers,
                    params={
                        "limit": 100,
                        "page": page,
                        "dateFrom": start_date_str,
                        "dateTo": end_date_str
                    }
                )
                logging.info(f"Traces Status Code (page {page}): {traces_response.status_code}")
                if traces_response.status_code == 200:
                    traces_data = traces_response.json()
                    batch = traces_data.get('data', [])
                    if not batch:
                        break
                    for trace in batch:
                        trace_item = {
                            'id': trace.get('id', ''),
                            'name': trace.get('name', ''),
                            'timestamp': trace.get('timestamp', ''),
                            'date': trace.get('timestamp', '').split('T')[0] if trace.get('timestamp') else '',
                            'userId': trace.get('userId', 'unknown'),
                            'status': trace.get('status', ''),
                            'agent': agent_name
                        }
                        # Extract metadata if available
                        metadata = trace.get('metadata', {})
                        if metadata:
                            if isinstance(metadata, dict):
                                trace_item['model'] = metadata.get('model', 'Unknown')
                                if 'sessionId' in metadata:
                                    trace_item['sessionId'] = metadata.get('sessionId')
                                if 'conversationId' in metadata:
                                    trace_item['conversationId'] = metadata.get('conversationId')
                        all_traces.append(trace_item)
                    if len(batch) < 100:
                        break
                    page += 1
                else:
                    logging.error(f"Failed to fetch traces for {agent_name} (page {page}): {traces_response.text}")
                    break
                
            # Save agent-specific traces data (first page only, for legacy)
            # with open(f"dashboard_data/{agent_name.lower()}_traces.json", "w") as f:
            #     json.dump(traces_data, f, indent=2, cls=NpEncoder)
            logging.info(f"Saved traces data for {agent_name}")
            print(f"Saved traces data for {agent_name}")
        except Exception as e:
            logging.error(f"Error processing traces for {agent_name}: {e}")
            print(f"Error processing traces for {agent_name}: {e}")

        # 3. Get sessions data with pagination
        try:
            page = 1
            while True:
                sessions_response = requests.get(
                    f"{base_url}/api/public/sessions",
                    headers=headers,
                    params={
                        "limit": 100,
                        "page": page,
                        "dateFrom": start_date_str,
                        "dateTo": end_date_str
                    }
                )
                logging.info(f"Sessions Status Code (page {page}): {sessions_response.status_code}")
                if sessions_response.status_code == 200:
                    sessions_data = sessions_response.json()
                    batch = sessions_data.get('data', [])
                    if not batch:
                        break
                    for session in batch:
                        created_at = session.get('createdAt', '')
                        session_item = {
                            'id': session.get('id', ''),
                            'name': session.get('name', ''),
                            'timestamp': created_at,
                            'date': created_at.split('T')[0] if created_at else '',
                            'userId': session.get('userId', 'unknown'),
                            'agent': agent_name
                        }
                        all_sessions.append(session_item)
                    if len(batch) < 100:
                        break
                    page += 1
                else:
                    logging.error(f"Failed to fetch sessions for {agent_name} (page {page}): {sessions_response.text}")
                    break
            logging.info(f"Saved sessions data for {agent_name}")
            print(f"Saved sessions data for {agent_name}")
        except Exception as e:
            logging.error(f"Error processing sessions for {agent_name}: {e}")
            print(f"Error processing sessions for {agent_name}: {e}")

        # 4. Get observations data
        try:
            observations_response = requests.get(
                f"{base_url}/api/public/observations",
                headers=headers,
                params={
                    "limit": 100,
                    "page": 1,
                    "dateFrom": start_date_str,
                    "dateTo": end_date_str
                }
            )
            
            logging.info(f"Observations Status Code: {observations_response.status_code}")
            
            if observations_response.status_code == 200:
                observations_data = observations_response.json()
                
                # Process observations data
                for observation in observations_data.get('data', []):
                    observation_item = {
                        'id': observation.get('id', ''),
                        'name': observation.get('name', ''),
                        'timestamp': observation.get('timestamp', ''),
                        'date': observation.get('timestamp', '').split('T')[0] if observation.get('timestamp') else '',
                        'traceId': observation.get('traceId', ''),
                        'type': observation.get('type', ''),
                        'level': observation.get('level', ''),
                        'agent': agent_name,
                        'inputTokens': 0,
                        'outputTokens': 0,
                        'totalTokens': 0,
                        'totalCost': 0
                    }
                    
                    # Extract token and cost information if available
                    # Check if input is a dictionary before accessing attributes
                    input_data = observation.get('input')
                    if input_data and isinstance(input_data, dict):
                        observation_item['inputTokens'] = input_data.get('totalTokens', 0)
                    
                    # Check if output is a dictionary before accessing attributes
                    output_data = observation.get('output')
                    if output_data and isinstance(output_data, dict):
                        observation_item['outputTokens'] = output_data.get('totalTokens', 0)
                    
                    # Check if usage is a dictionary before accessing attributes
                    usage_data = observation.get('usage')
                    if usage_data and isinstance(usage_data, dict):
                        observation_item['totalTokens'] = usage_data.get('totalTokens', 0)
                        observation_item['promptTokens'] = usage_data.get('promptTokens', 0)
                        observation_item['completionTokens'] = usage_data.get('completionTokens', 0)
                        observation_item['totalCost'] = usage_data.get('totalCost', 0)
                    
                    # Extract model information if available (as a direct field)
                    if 'model' in observation and observation['model']:
                        observation_item['model'] = observation['model']
                    
                    all_observations.append(observation_item)
                
                # Save agent-specific observations data
                with open(f"dashboard_data/{agent_name.lower()}_observations.json", "w") as f:
                    json.dump(observations_data, f, indent=2, cls=NpEncoder)
                    
                logging.info(f"Saved observations data for {agent_name}")
                print(f"Saved observations data for {agent_name}")
            else:
                logging.error(f"Failed to fetch observations for {agent_name}: {observations_response.text}")
                print(f"Failed to fetch observations for {agent_name}: {observations_response.text}")
        except Exception as e:
            logging.error(f"Error processing observations for {agent_name}: {e}")
            print(f"Error processing observations for {agent_name}: {e}")

    # Save combined data for all agents
    logging.info("Writing combined daily metrics, traces, sessions, and observations files.")
    with open("dashboard_data/all_daily_metrics.json", "w") as f:
        json.dump(all_daily_metrics, f, indent=2, cls=NpEncoder)
    with open("dashboard_data/all_traces.json", "w") as f:
        json.dump(all_traces, f, indent=2, cls=NpEncoder)
    with open("dashboard_data/all_sessions.json", "w") as f:
        json.dump(all_sessions, f, indent=2, cls=NpEncoder)
    with open("dashboard_data/all_observations.json", "w") as f:
        json.dump(all_observations, f, indent=2, cls=NpEncoder)
    logging.info("Finished writing combined files.")

    # --- Restore summary files for dashboard compatibility ---
    # 1. agent_comparison.json
    agent_comparison = []
    agent_names = set(row['agent'] for row in all_daily_metrics)
    for agent in agent_names:
        count_traces = sum(row['countTraces'] for row in all_daily_metrics if row['agent'] == agent)
        agent_comparison.append({
            'agent': agent,
            'countTraces': count_traces
        })
    with open("dashboard_data/agent_comparison.json", "w") as f:
        json.dump(agent_comparison, f, indent=2, cls=NpEncoder)

    # 2. daily_tokens_by_agent.json
    # Structure: [{date: 'YYYY-MM-DD', AGENT1: tokens, AGENT2: tokens, ...}, ...]
    daily_tokens = {}
    for row in all_daily_metrics:
        date = row['date']
        agent = row['agent']
        tokens = row.get('totalTokens', 0)
        if date not in daily_tokens:
            daily_tokens[date] = {}
        daily_tokens[date][agent] = tokens
    daily_tokens_list = []
    for date in sorted(daily_tokens.keys()):
        entry = {'date': date}
        entry.update(daily_tokens[date])
        daily_tokens_list.append(entry)
    with open("dashboard_data/daily_tokens_by_agent.json", "w") as f:
        json.dump(daily_tokens_list, f, indent=2, cls=NpEncoder)

    # 3. daily_cost_by_agent.json
    daily_cost = {}
    for row in all_daily_metrics:
        date = row['date']
        agent = row['agent']
        cost = row.get('totalCost', 0)
        if date not in daily_cost:
            daily_cost[date] = {}
        daily_cost[date][agent] = cost
    daily_cost_list = []
    for date in sorted(daily_cost.keys()):
        entry = {'date': date}
        entry.update(daily_cost[date])
        daily_cost_list.append(entry)
    with open("dashboard_data/daily_cost_by_agent.json", "w") as f:
        json.dump(daily_cost_list, f, indent=2, cls=NpEncoder)

    # 4. user_metrics.json
    # Structure: [{userId, totalTraces, agents, firstSeen, lastSeen}]
    user_dict = {}
    for trace in all_traces:
        user = trace.get('userId', 'unknown')
        agent = trace.get('agent', 'unknown')
        ts = trace.get('timestamp')
        if not user:
            continue
        if user not in user_dict:
            user_dict[user] = {
                'userId': user,
                'totalTraces': 0,
                'agents': set(),
                'firstSeen': ts,
                'lastSeen': ts
            }
        user_dict[user]['totalTraces'] += 1
        user_dict[user]['agents'].add(agent)
        if ts and (not user_dict[user]['firstSeen'] or ts < user_dict[user]['firstSeen']):
            user_dict[user]['firstSeen'] = ts
        if ts and (not user_dict[user]['lastSeen'] or ts > user_dict[user]['lastSeen']):
            user_dict[user]['lastSeen'] = ts
    user_metrics = []
    for user, info in user_dict.items():
        user_metrics.append({
            'userId': user,
            'totalTraces': info['totalTraces'],
            'agents': list(info['agents']),
            'firstSeen': info['firstSeen'],
            'lastSeen': info['lastSeen']
        })
    with open("dashboard_data/user_metrics.json", "w") as f:
        json.dump(user_metrics, f, indent=2, cls=NpEncoder)

    # 5. agent_distribution.json
    # Structure: [{agent, count}]
    agent_dist = {}
    for trace in all_traces:
        agent = trace.get('agent', 'unknown')
        agent_dist[agent] = agent_dist.get(agent, 0) + 1
    agent_distribution = [{'agent': agent, 'count': count} for agent, count in agent_dist.items()]
    with open("dashboard_data/agent_distribution.json", "w") as f:
        json.dump(agent_distribution, f, indent=2, cls=NpEncoder)

    # 6. conversation_lengths.json
    # Structure: [{conversationId, length}]
    # We'll use traceId and count observations per trace
    conv_lengths = {}
    for obs in all_observations:
        trace_id = obs.get('traceId')
        if not trace_id:
            continue
        conv_lengths[trace_id] = conv_lengths.get(trace_id, 0) + 1
    conversation_lengths = [{'conversationId': cid, 'length': length} for cid, length in conv_lengths.items()]
    with open("dashboard_data/conversation_lengths.json", "w") as f:
        json.dump(conversation_lengths, f, indent=2, cls=NpEncoder)

    # 7. model_token_usage.json
    # Structure: [{model, totalTokens, totalCost}]
    model_usage = {}
    for obs in all_observations:
        model = obs.get('model', 'unknown')
        tokens = obs.get('totalTokens', 0)
        cost = obs.get('totalCost', 0)
        if model not in model_usage:
            model_usage[model] = {'model': model, 'totalTokens': 0, 'totalCost': 0}
        model_usage[model]['totalTokens'] += tokens
        model_usage[model]['totalCost'] += cost
    model_token_usage = list(model_usage.values())
    with open("dashboard_data/model_token_usage.json", "w") as f:
        json.dump(model_token_usage, f, indent=2, cls=NpEncoder)

    logging.info("Restored all summary files for dashboard compatibility.")
    print("Restored all summary files for dashboard compatibility.")
    print("\n=== Data Processing Complete ===")
    print("All data has been processed and saved to the dashboard_data directory")
    logging.info("Data Processing Complete. All data has been processed and saved.")
    return True

if __name__ == "__main__":
    # Default to 30 days if not specified
    fetch_langfuse_data(days_back=30)