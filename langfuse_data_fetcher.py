import requests
import base64
import os
import json
from datetime import datetime, timedelta, timezone
import numpy as np
from dotenv import load_dotenv
import logging
from collections import defaultdict
import re # Import regex for cleaning endpoint names

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='langfuse_data_fetcher.log',
    filemode='a', # Append to the log file
    format='%(asctime)s - %(levelname)s - %(message)s', # Added levelname
    level=logging.INFO # Log INFO level and above (WARNING, ERROR, CRITICAL)
)
# Optional: Add console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# logging.getLogger().addHandler(console_handler)


# Custom JSON encoder to handle NumPy types and datetime
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

def parse_datetime(date_string):
    """Safely parse ISO date strings, returning None if invalid."""
    if not date_string:
        logging.warning("Attempted to parse an empty or null date string.")
        return None
    try:
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse datetime string '{date_string}': {e}")
        return None

def fetch_paginated_data(base_url, endpoint, headers, params):
    """Fetches all pages for a given Langfuse endpoint."""
    all_raw_data = [] # Store raw data from batches
    page = 1
    time_range = f"from {params.get('fromTimestamp') or params.get('fromCreatedAt') or 'start'} to {params.get('toTimestamp') or params.get('toCreatedAt') or 'end'}"
    print(f"   Fetching paginated data for endpoint: {endpoint} ({time_range})...")
    logging.info(f"Initiating paginated fetch for {endpoint} ({time_range})")
    
    # Ensure we have date filtering params to avoid fetching all data
    if not (params.get('fromTimestamp') or params.get('fromCreatedAt')):
        logging.error(f"Missing date filter for {endpoint}. This would fetch ALL data. Aborting.")
        print(f"   ERROR: Missing date filter for {endpoint}. Aborting to prevent fetching all data.")
        return []

    while True:
        params['page'] = page
        logging.debug(f"Fetching page {page} for {endpoint} with params: {params}")
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
            logging.info(f"Fetching {endpoint} page {page} - Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                     logging.error(f"JSON decode error on page {page} for {endpoint}: {e} - Response text: {response.text[:500]}...")
                     print(f"   ERROR: JSON decode error fetching {endpoint} page {page}. Check log.")
                     break

                batch = data.get('data', [])
                meta = data.get('meta', {})
                total_items = meta.get('totalItems')
                limit = meta.get('limit', params.get('limit', 100))
                total_pages = meta.get('totalPages')

                if batch:
                    all_raw_data.extend(batch)
                    logging.info(f"Fetched {len(batch)} items on page {page} for {endpoint}. Total fetched: {len(all_raw_data)}")
                    print(f"   Page {page}: Fetched {len(batch)} items. Total so far: {len(all_raw_data)}")
                else:
                    logging.info(f"No more data found for {endpoint} on page {page}.")
                    print(f"   Page {page}: No more items found. Fetch complete for {endpoint}.")
                    break

                current_total = len(all_raw_data)
                if total_pages is not None and page >= total_pages: break
                if total_items is not None and current_total >= total_items: break
                if limit is not None and limit > 0 and len(batch) < limit: break

                page += 1

            elif response.status_code == 401:
                 logging.error(f"Authorization error (401) fetching {endpoint} page {page}. Check API keys.")
                 print(f"   ERROR: Authorization failed (401) for {endpoint}. Check API keys.")
                 break
            elif response.status_code == 404:
                 logging.error(f"Endpoint not found (404) at {base_url}{endpoint}.")
                 print(f"   ERROR: Endpoint not found (404): {endpoint}")
                 break
            else:
                logging.error(f"Failed to fetch {endpoint} page {page}: Status {response.status_code} - {response.text[:500]}...")
                print(f"   ERROR: Failed to fetch {endpoint} page {page} (Status: {response.status_code}). Check log.")
                break

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching {endpoint} page {page}: {e}", exc_info=True)
            print(f"   ERROR: Network or request error fetching {endpoint} page {page}. Check log.")
            break

    logging.info(f"Finished paginated fetch for {endpoint}. Total items retrieved: {len(all_raw_data)}")
    print(f"   Finished fetching for {endpoint}. Total items: {len(all_raw_data)}")
    return all_raw_data

def save_raw_data(data_list, agent_name, endpoint_name, base_dir="dashboard_data/raw_api"):
    """Saves the raw fetched data list to a JSON file."""
    if not data_list:
        logging.warning(f"No raw data provided for {agent_name} - {endpoint_name}. Skipping save.")
        return

    cleaned_endpoint = endpoint_name.replace('/api/public/', '').replace('/', '_')
    filename = f"{agent_name}_{cleaned_endpoint}_raw.json"
    filepath = os.path.join(base_dir, filename)

    try:
        os.makedirs(base_dir, exist_ok=True) # Ensure directory exists
        print(f"      Saving raw data to {filepath}...")
        logging.info(f"Saving {len(data_list)} raw items for {agent_name} - {endpoint_name} to {filepath}")
        with open(filepath, "w") as f:
            json.dump(data_list, f, indent=2, cls=NpEncoder)
        logging.info(f"Successfully saved raw data to {filepath}")
        print(f"         Done.")
    except Exception as e:
        logging.error(f"Failed to save raw data to {filepath}: {e}", exc_info=True)
        print(f"         ERROR: Failed to save raw data to {filepath}. Check logs.")


def fetch_langfuse_data(start_date):
    """
    Main function to fetch and process Langfuse data.
    Fetches data from the specified start_date to the current time.
    
    Args:
        start_date: The start date as a datetime object with timezone info.
    """
    start_time_script = datetime.now()
    
    # Add very clear start markers for both console and log
    print("\n" + "="*80)
    print("SCRIPT_MARKER: LANGFUSE_DATA_FETCHER_STARTING")
    print(f"Starting Langfuse data refresh at: {start_time_script}")
    print("="*80 + "\n")
    
    logging.info("="*50)
    logging.info("SCRIPT_MARKER: LANGFUSE_DATA_FETCHER_STARTING")
    logging.info(f"--- Langfuse Data Fetch Script Started: {start_time_script} ---")
    logging.info("="*50)
    
    # Date range calculation
    end_date = datetime.now(timezone.utc)
    
    # Ensure the provided date has timezone info
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    
    # Ensure we have an exact midnight timestamp for consistent filtering
    start_date = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=timezone.utc)
    
    logging.info(f"Using start date: {start_date.isoformat()}")
    print(f"Using start date: {start_date.isoformat()}")
    
    start_date_str_iso = start_date.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    end_date_str_iso = end_date.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    start_date_str_day = start_date.strftime("%Y-%m-%d")
    end_date_str_day = end_date.strftime("%Y-%m-%d")
    logging.info(f"Date range (ISO UTC): {start_date_str_iso} to {end_date_str_iso}")
    logging.info(f"Date range (YYYY-MM-DD): {start_date_str_day} to {end_date_str_day}")
    print(f"Date range: {start_date_str_iso} to {end_date_str_iso}")

    base_url = os.getenv("LANGFUSE_HOST", "https://your-langfuse-host.com")
    if not base_url.startswith("http"): base_url = "https://" + base_url
    logging.info(f"Using Langfuse Host: {base_url}")
    print(f"Using Langfuse Host: {base_url}")

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
    logging.info(f"Found {len(agent_configs)} agent configurations.")
    print(f"Processing {len(agent_configs)} agent configurations...")

    data_dir = "dashboard_data"
    raw_data_dir = os.path.join(data_dir, "raw_api") # Define raw data subdirectory
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True) # Ensure raw data directory exists
    logging.info(f"Ensured data directory exists: {data_dir}")
    logging.info(f"Ensured raw data directory exists: {raw_data_dir}")

    # Initialize lists for *processed* data across all agents
    all_traces_processed = []
    all_sessions_processed = []
    all_observations_processed = []
    all_daily_metrics_processed = []

    # --- Loop Through Agents ---
    for agent_config in agent_configs:
        agent_name = agent_config.get("name", "UnknownAgent")
        public_key = agent_config.get("public_key")
        secret_key = agent_config.get("secret_key")
        print(f"\n=== Processing Agent: {agent_name} ===")
        logging.info(f"--- Processing Agent: {agent_name} ---")

        if not public_key or not secret_key:
            logging.warning(f"API keys missing for {agent_name}. Skipping.")
            print(f"   WARNING: API keys missing for {agent_name}. Skipping.")
            continue

        auth_string = f"{public_key}:{secret_key}"
        auth_b64 = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
        headers = { "Authorization": f"Basic {auth_b64}", "Content-Type": "application/json" }
        logging.debug(f"Prepared headers for {agent_name}")

        # --- Fetch Data and Save Raw Responses ---
        common_params_trace_obs = {"limit": 100, "fromTimestamp": start_date_str_iso, "toTimestamp": end_date_str_iso}
        # Ensure sessions use the same exact timestamp for consistency
        session_params = {"limit": 100, "fromCreatedAt": start_date_str_iso, "toCreatedAt": end_date_str_iso}

        # 1. Daily Metrics
        print(f"   Fetching Daily Metrics for {agent_name}...")
        metrics_endpoint = "/api/public/metrics/daily"
        metrics_params = { "startDate": start_date_str_day, "endDate": end_date_str_day }
        raw_metrics_data = []
        try:
            metrics_response = requests.get(f"{base_url}{metrics_endpoint}", headers=headers, params=metrics_params)
            logging.info(f"Daily Metrics API Status Code: {metrics_response.status_code}")
            print(f"   Daily Metrics API Status: {metrics_response.status_code}")
            if metrics_response.status_code == 200:
                raw_metrics_data = metrics_response.json().get('data', [])
                save_raw_data(raw_metrics_data, agent_name, metrics_endpoint, base_dir=raw_data_dir)
                logging.info(f"Processing {len(raw_metrics_data)} raw daily metric records")
                for day_data in raw_metrics_data:
                     day_metrics = {
                        'date': day_data.get('date'), 'agent': agent_name,
                        'countTraces': day_data.get('countTraces', 0),
                        'countObservations': day_data.get('countObservations', 0),
                        'totalCost': day_data.get('totalCost', 0.0) or 0.0,
                        'totalTokens': 0, 'modelUsage': [] }
                     total_day_tokens = 0
                     for usage in day_data.get('usage', []):
                        input_usage = usage.get('inputUsage', 0) or 0
                        output_usage = usage.get('outputUsage', 0) or 0
                        tokens = usage.get('totalUsage')
                        if tokens is None: tokens = input_usage + output_usage
                        else: tokens = tokens or 0
                        cost = usage.get('totalCost', 0.0) or 0.0
                        model_name = usage.get('model', 'Unknown')
                        total_day_tokens += tokens
                        day_metrics['modelUsage'].append({ 'model': model_name, 'inputUsage': input_usage, 'outputUsage': output_usage, 'totalUsage': tokens, 'countTraces': usage.get('countTraces', 0), 'countObservations': usage.get('countObservations', 0), 'totalCost': cost })
                     day_metrics['totalTokens'] = total_day_tokens
                     all_daily_metrics_processed.append(day_metrics)
            else:
                 logging.error(f"Failed to fetch daily metrics: {metrics_response.status_code}")
                 print(f"   ERROR: Failed to fetch daily metrics (Status: {metrics_response.status_code})")
        except Exception as e:
            logging.error(f"Exception processing daily metrics: {e}", exc_info=True)
            print(f"   ERROR: Exception during daily metrics processing.")

        # 2. Traces
        print(f"   Fetching Traces for {agent_name}...")
        traces_endpoint = "/api/public/traces"
        raw_agent_traces = fetch_paginated_data(base_url, traces_endpoint, headers, common_params_trace_obs.copy())
        save_raw_data(raw_agent_traces, agent_name, traces_endpoint, base_dir=raw_data_dir)
        print(f"   Processing {len(raw_agent_traces)} fetched traces...")
        
        # Double-check that all items have timestamps on or after our filter date
        trace_count_before_filter = len(raw_agent_traces)
        raw_agent_traces = [trace for trace in raw_agent_traces if 
                           parse_datetime(trace.get('timestamp')) and 
                           parse_datetime(trace.get('timestamp')) >= start_date]
        if len(raw_agent_traces) < trace_count_before_filter:
            filtered_count = trace_count_before_filter - len(raw_agent_traces)
            logging.warning(f"Filtered out {filtered_count} traces with timestamps before {start_date.isoformat()}")
            print(f"   Warning: Filtered out {filtered_count} traces with timestamps before {start_date.isoformat()}")
        
        # Check trace API for token and cost data
        trace_with_cost = sum(1 for t in raw_agent_traces if t.get('totalCost') is not None and t.get('totalCost') > 0)
        trace_with_tokens = sum(1 for t in raw_agent_traces if t.get('totalTokens') is not None and t.get('totalTokens') > 0)
        logging.info(f"Trace API data check: {trace_with_cost}/{len(raw_agent_traces)} have cost data, {trace_with_tokens}/{len(raw_agent_traces)} have token data")
        print(f"   Trace API data check: {trace_with_cost}/{len(raw_agent_traces)} have cost data, {trace_with_tokens}/{len(raw_agent_traces)} have token data")
        
        # Save a sample trace to help debug
        if len(raw_agent_traces) > 0:
            sample_trace = raw_agent_traces[0]
            logging.info(f"Sample trace data - ID: {sample_trace.get('id')}, Cost: {sample_trace.get('totalCost')}, Tokens: {sample_trace.get('totalTokens')}, Latency: {sample_trace.get('latency')}")
            
            # Log the first 5 traces with latency data for debugging
            latency_traces = [t for t in raw_agent_traces if t.get('latency') is not None][:5]
            if latency_traces:
                logging.info(f"Found {len([t for t in raw_agent_traces if t.get('latency') is not None])} traces with latency data")
                for idx, trace in enumerate(latency_traces):
                    logging.info(f"Raw trace {idx+1} latency example - ID: {trace.get('id')}, Latency: {trace.get('latency')}")
        
        for trace in raw_agent_traces:
            timestamp = parse_datetime(trace.get('timestamp'))
            all_traces_processed.append({
                'id': trace.get('id'), 'timestamp': timestamp, 'name': trace.get('name'),
                'userId': trace.get('userId'), 'sessionId': trace.get('sessionId'),
                'metadata': trace.get('metadata'), 'tags': trace.get('tags', []),
                'agent': agent_name,
                # Get cost from trace API (works correctly)
                'totalCost': trace.get('totalCost') if trace.get('totalCost') is not None else 0.0,
                # Initialize tokens to 0, we'll populate from observations later
                'totalTokens': 0,
                # Use raw trace latency value if available, otherwise initialize to 0.0
                'latency': trace.get('latency', 0.0),
                # Placeholders for other values derived from observations
                'messageCount': 0,
                'startTime': timestamp, 'endTime': timestamp
            })
        print(f"   Finished processing traces.")

        # 3. Sessions
        print(f"   Fetching Sessions for {agent_name}...")
        sessions_endpoint = "/api/public/sessions"
        raw_agent_sessions = fetch_paginated_data(base_url, sessions_endpoint, headers, session_params.copy())
        save_raw_data(raw_agent_sessions, agent_name, sessions_endpoint, base_dir=raw_data_dir)
        print(f"   Processing {len(raw_agent_sessions)} fetched sessions...")
        
        # Double-check that all sessions have creation dates on or after our filter date
        session_count_before_filter = len(raw_agent_sessions)
        raw_agent_sessions = [session for session in raw_agent_sessions if 
                             parse_datetime(session.get('createdAt')) and 
                             parse_datetime(session.get('createdAt')) >= start_date]
        if len(raw_agent_sessions) < session_count_before_filter:
            filtered_count = session_count_before_filter - len(raw_agent_sessions)
            logging.warning(f"Filtered out {filtered_count} sessions with creation dates before {start_date.isoformat()}")
            print(f"   Warning: Filtered out {filtered_count} sessions with creation dates before {start_date.isoformat()}")
            
        for session in raw_agent_sessions:
            created_at = parse_datetime(session.get('createdAt'))
            # Only include sessions at or after the start date
            all_sessions_processed.append({
                'id': session.get('id'), 'createdAt': created_at, 'agent': agent_name,
                'userId': None, 'durationSeconds': 0, 'totalLatency': 0.0,
                'traceCount': 0, 'totalTokens': 0, 'totalCost': 0.0,
                'startTime': created_at, 'endTime': created_at })
        print(f"   Finished processing sessions.")

        # 4. Observations
        fetch_obs = True
        if fetch_obs:
            print(f"   Fetching Observations for {agent_name}...")
            obs_endpoint = "/api/public/observations"
            raw_agent_observations = fetch_paginated_data(base_url, obs_endpoint, headers, common_params_trace_obs.copy())
            save_raw_data(raw_agent_observations, agent_name, obs_endpoint, base_dir=raw_data_dir)
            print(f"   Processing {len(raw_agent_observations)} fetched observations...")
            
            # Double-check that all observations have start times on or after our filter date
            obs_count_before_filter = len(raw_agent_observations)
            raw_agent_observations = [obs for obs in raw_agent_observations if 
                                     parse_datetime(obs.get('startTime')) and 
                                     parse_datetime(obs.get('startTime')) >= start_date]
            if len(raw_agent_observations) < obs_count_before_filter:
                filtered_count = obs_count_before_filter - len(raw_agent_observations)
                logging.warning(f"Filtered out {filtered_count} observations with start times before {start_date.isoformat()}")
                print(f"   Warning: Filtered out {filtered_count} observations with start times before {start_date.isoformat()}")
            
            # Check observations for token data
            obs_with_tokens = sum(1 for o in raw_agent_observations if o.get('usage', {}) and (o.get('usage', {}).get('total', 0) > 0 or 
                                                             o.get('usage', {}).get('input', 0) > 0 or 
                                                             o.get('usage', {}).get('output', 0) > 0))
            logging.info(f"Observation API data check: {obs_with_tokens}/{len(raw_agent_observations)} have token data")
            print(f"   Observation API data check: {obs_with_tokens}/{len(raw_agent_observations)} have token data")
            
            # Save a sample observation to help debug
            if len(raw_agent_observations) > 0:
                sample_obs = raw_agent_observations[0]
                logging.info(f"Sample observation data - ID: {sample_obs.get('id')}, Usage: {sample_obs.get('usage')}")
            
            for obs in raw_agent_observations:
                start_time = parse_datetime(obs.get('startTime'))
                end_time = parse_datetime(obs.get('endTime'))
                
                # Skip observations that don't meet the time criteria
                
                usage_data = obs.get('usage', {}) or {}
                input_tokens = usage_data.get('input', 0) or 0
                output_tokens = usage_data.get('output', 0) or 0
                total_tokens = usage_data.get('total')
                if total_tokens is None: total_tokens = input_tokens + output_tokens
                else: total_tokens = total_tokens or 0
                input_cost = usage_data.get('inputCost', 0.0) or 0.0
                output_cost = usage_data.get('outputCost', 0.0) or 0.0
                total_cost = usage_data.get('totalCost')
                if total_cost is None: total_cost = input_cost + output_cost
                else: total_cost = total_cost or 0.0
                
                # Make sure the cost is extracted properly
                if total_cost == 0.0 and 'cost' in usage_data:
                    total_cost = float(usage_data.get('cost', 0.0))
                
                latency = 0.0
                if start_time and end_time and isinstance(start_time, datetime) and isinstance(end_time, datetime):
                    latency = max(0, (end_time - start_time).total_seconds())
                all_observations_processed.append({
                    'id': obs.get('id'), 'traceId': obs.get('traceId'), 'type': obs.get('type'),
                    'name': obs.get('name'), 'startTime': start_time, 'endTime': end_time,
                    'latency': latency, 'model': obs.get('model'),
                    'inputTokens': input_tokens, 'outputTokens': output_tokens,
                    'totalTokens': total_tokens, 'totalCost': total_cost,
                    'level': obs.get('level'), 'statusMessage': obs.get('statusMessage'),
                    'parentObservationId': obs.get('parentObservationId'),
                    'metadata': obs.get('metadata'), 'agent': agent_name })
            print(f"   Finished processing observations.")
        else:
            print("   Skipping Observation fetch.")
            logging.info("Skipping Observation fetch.")

        logging.info(f"--- Finished fetching data for agent: {agent_name} ---")
        print(f"=== Finished fetching data for {agent_name} ===")

    # --- Data Aggregation (Across All Agents) ---
    total_obs_count = len(all_observations_processed)
    total_trace_count = len(all_traces_processed)
    total_session_count = len(all_sessions_processed)
    print(f"\n=== Starting Data Aggregation Across All Agents ===")
    print(f"   Total Observations Processed: {total_obs_count}")
    print(f"   Total Traces Processed: {total_trace_count}")
    print(f"   Total Sessions Processed: {total_session_count}")
    logging.info(f"Aggregation Start. Obs: {total_obs_count}, Traces: {total_trace_count}, Sess: {total_session_count}")

    # 1. Aggregate Observation Data per Trace for token data
    print("   Aggregating token data from observations to traces...")
    logging.info("Aggregating token data from observations to traces...")
    trace_metrics = defaultdict(lambda: {'messageCount': 0, 'totalLatency': 0.0, 'totalTokens': 0, 'startTime': None, 'endTime': None})
    model_usage_agg = defaultdict(lambda: {'totalTokens': 0, 'totalCost': 0.0, 'observationCount': 0})

    for i, obs in enumerate(all_observations_processed):
        trace_id = obs.get('traceId')
        if (i + 1) % 10000 == 0:
             logging.info(f"Aggregating observation {i+1}/{total_obs_count}...")
             print(f"      Aggregating observation {i+1}/{total_obs_count}...")
        if not trace_id: continue

        trace_metrics[trace_id]['messageCount'] += 1
        trace_metrics[trace_id]['totalLatency'] += obs.get('latency', 0.0)
        # Critical fix: Capture token data from observations
        trace_metrics[trace_id]['totalTokens'] += obs.get('totalTokens', 0)

        # Track min start and max end times
        obs_start_time = obs.get('startTime'); obs_end_time = obs.get('endTime')
        if obs_end_time is None and obs_start_time is not None: obs_end_time = obs_start_time
        current_trace_start = trace_metrics[trace_id]['startTime']
        current_trace_end = trace_metrics[trace_id]['endTime']
        if obs_start_time and isinstance(obs_start_time, datetime):
            if current_trace_start is None or obs_start_time < current_trace_start:
                trace_metrics[trace_id]['startTime'] = obs_start_time
        if obs_end_time and isinstance(obs_end_time, datetime):
             if current_trace_end is None or obs_end_time > current_trace_end:
                 trace_metrics[trace_id]['endTime'] = obs_end_time

        # Aggregate model usage globally
        model = obs.get('model')
        if model:
             model_key = model or "Unknown"
             model_usage_agg[model_key]['totalTokens'] += obs.get('totalTokens', 0)
             model_usage_agg[model_key]['totalCost'] += obs.get('totalCost', 0.0)
             model_usage_agg[model_key]['observationCount'] += 1

    # Update traces with aggregated observation data
    updated_trace_count = 0
    traces_with_tokens_updated = 0
    total_tokens_from_observations = 0
    for trace in all_traces_processed:
        trace_id = trace.get('id')
        if trace_id in trace_metrics:
            metrics = trace_metrics[trace_id]
            # Update message count and times
            trace['messageCount'] = metrics['messageCount']
            
            # Only update latency if there is observation data and the trace doesn't already have latency
            # This way we prioritize the original trace latency over calculated observation latency
            if trace.get('latency', 0.0) == 0.0 and metrics['totalLatency'] > 0.0:
                trace['latency'] = metrics['totalLatency']
                logging.info(f"Updated trace {trace_id} latency from observations: {metrics['totalLatency']}")
            
            # Key fix: Update token data from observations
            obs_tokens = metrics['totalTokens']
            if obs_tokens > 0:
                trace['totalTokens'] = obs_tokens
                total_tokens_from_observations += obs_tokens
                traces_with_tokens_updated += 1
                
                # Log the first few updates for verification
                if traces_with_tokens_updated <= 5:
                    logging.info(f"Updated trace {trace_id} with {obs_tokens} tokens from observations")
            
            # Update time data
            trace_start_updated = False
            if metrics['startTime'] is not None:
                trace['startTime'] = metrics['startTime']; trace_start_updated = True
            if metrics['endTime'] is not None and (trace['startTime'] is None or metrics['endTime'] >= trace['startTime']):
                 trace['endTime'] = metrics['endTime']
            elif trace['startTime'] is not None: trace['endTime'] = trace['startTime']
            
            updated_trace_count += 1
        else:
            if trace['startTime'] is None: trace['startTime'] = trace['timestamp']
            if trace['endTime'] is None: trace['endTime'] = trace['timestamp']

    logging.info(f"Trace aggregation complete: Updated {updated_trace_count}/{total_trace_count} traces")
    logging.info(f"Token data updated for {traces_with_tokens_updated} traces. Total tokens from observations: {total_tokens_from_observations}")
    print(f"   Finished aggregating trace data. Updated {updated_trace_count} traces.")
    print(f"   Token data updated for {traces_with_tokens_updated} traces. Total tokens: {total_tokens_from_observations}")

    # After processing traces, log some statistics about latency values
    trace_latency_stats = {
        "with_raw_latency": 0,
        "with_observation_latency": 0,
        "with_no_latency": 0,
        "total_traces": len(all_traces_processed)
    }

    for trace in all_traces_processed:
        trace_id = trace.get('id')
        if trace.get('latency', 0) > 0:
            if trace_id in trace_metrics and trace_metrics[trace_id]['totalLatency'] > 0:
                trace_latency_stats["with_observation_latency"] += 1
            else:
                trace_latency_stats["with_raw_latency"] += 1
        else:
            trace_latency_stats["with_no_latency"] += 1

    logging.info(f"Trace latency statistics: {trace_latency_stats}")
    print(f"   Trace latency statistics: Raw: {trace_latency_stats['with_raw_latency']}, Obs: {trace_latency_stats['with_observation_latency']}, None: {trace_latency_stats['with_no_latency']}, Total: {trace_latency_stats['total_traces']}")

    # 2. Aggregate Trace Data per Session (using trace-level token/cost)
    print("   Aggregating trace data per session...")
    logging.info("Aggregating trace data per session...")
    session_metrics = defaultdict(lambda: {'traceCount': 0, 'totalTokens': 0, 'totalCost': 0.0, 'userIds': set(), 'agentNames': set(), 'totalLatency': 0.0, 'startTime': None, 'endTime': None})
    traces_grouped_by_session = defaultdict(list)
    for trace in all_traces_processed:
         session_id = trace.get('sessionId')
         if session_id: traces_grouped_by_session[session_id].append(trace)
    sessions_found_in_traces = len(traces_grouped_by_session)
    logging.info(f"Found {sessions_found_in_traces} unique session IDs referenced.")
    print(f"   Found {sessions_found_in_traces} unique session IDs in traces.")

    for session_id, traces_in_session in traces_grouped_by_session.items():
        session_start, session_end, session_tokens, session_cost, session_total_latency = None, None, 0, 0.0, 0.0
        
        # Log for debugging - detailed session trace info
        if len(traces_in_session) >= 3:  # Only log sessions with a meaningful number of traces
            logging.info(f"Detailed analysis of session {session_id} with {len(traces_in_session)} traces:")
            
        for trace in traces_in_session:
            session_metrics[session_id]['traceCount'] += 1
            # ** Fix: Use trace-level token/cost directly from processed trace **
            session_tokens += trace.get('totalTokens', 0)  # Already defaulted to 0 if null
            session_cost += trace.get('totalCost', 0.0)    # Already defaulted to 0.0 if null
            
            # Use trace latency directly - this now preserves the original trace latency
            trace_latency = trace.get('latency', 0.0)
            session_total_latency += trace_latency
            
            # Detailed logging for sample sessions
            if len(traces_in_session) >= 3 and session_metrics[session_id]['traceCount'] <= 5:  # Log first 5 traces
                logging.info(f"  Trace {trace.get('id')}: latency={trace_latency:.2f}s, start={trace.get('startTime')}, end={trace.get('endTime')}")
                time_diff = 0
                if trace.get('startTime') and trace.get('endTime'):
                    time_diff = (trace.get('endTime') - trace.get('startTime')).total_seconds()
                    if abs(time_diff - trace_latency) > 1.0:  # If there's a significant difference
                        logging.info(f"    NOTE: Trace time diff ({time_diff:.2f}s) differs from latency ({trace_latency:.2f}s)")
            
            if trace.get('userId'): session_metrics[session_id]['userIds'].add(trace.get('userId'))
            if trace.get('agent'): session_metrics[session_id]['agentNames'].add(trace.get('agent'))
            trace_start_time = trace.get('startTime');
            trace_end_time = trace.get('endTime');  # Add this line to fix the missing variable
            if trace_start_time and isinstance(trace_start_time, datetime):
                if session_start is None or trace_start_time < session_start: session_start = trace_start_time
            if trace_end_time and isinstance(trace_end_time, datetime):
                 if session_end is None or trace_end_time > session_end: session_end = trace_end_time
        
        # If we have start/end times but no latency, calculate an estimated latency
        if session_total_latency == 0.0 and session_start and session_end and session_start != session_end:
            estimated_latency = max(0, (session_end - session_start).total_seconds())
            session_metrics[session_id]['totalLatency'] = estimated_latency
            logging.info(f"Estimated session latency for {session_id}: {estimated_latency:.2f}s (no observation latency data)")
            print(f"      Estimated session latency for {session_id}: {estimated_latency:.2f}s (based on timestamps)")
        else:
            session_metrics[session_id]['totalLatency'] = session_total_latency
            
        # Log the final session metrics for sample sessions
        if len(traces_in_session) >= 3:
            session_time_diff = 0
            if session_start and session_end:
                session_time_diff = (session_end - session_start).total_seconds()
            logging.info(f"Session {session_id} final stats: latency_sum={session_total_latency:.2f}s, time_diff={session_time_diff:.2f}s")
            logging.info(f"  Start: {session_start}, End: {session_end}")
            
        session_metrics[session_id]['totalTokens'] = session_tokens
        session_metrics[session_id]['totalCost'] = session_cost
        session_metrics[session_id]['startTime'] = session_start
        session_metrics[session_id]['endTime'] = session_end

    # Update the main session list
    updated_session_count, sessions_without_trace_times, sessions_without_traces = 0, 0, 0
    for session in all_sessions_processed:
        session_id = session.get('id')
        if session_id in session_metrics:
            metrics = session_metrics[session_id]
            # Update session totals using the correctly summed trace values
            session.update({ 'traceCount': metrics['traceCount'], 'totalTokens': metrics['totalTokens'], 'totalCost': metrics['totalCost'], 'totalLatency': metrics['totalLatency'] })
            session_start_updated = False
            if metrics['startTime'] is not None: session['startTime'] = metrics['startTime']; session_start_updated = True
            elif session['createdAt']: session['startTime'] = session['createdAt']; session_start_updated = True
            if metrics['endTime'] is not None and session['startTime'] and metrics['endTime'] >= session['startTime']: session['endTime'] = metrics['endTime']
            elif session['startTime'] is not None: session['endTime'] = session['startTime']
            elif session['createdAt']: session['endTime'] = session['createdAt']
            
            # IMPORTANT FIX: Use totalLatency for durationSeconds instead of time difference
            session['durationSeconds'] = metrics['totalLatency']
            
            if metrics['userIds']: session['userId'] = sorted(list(metrics['userIds']))[0]
            else: session['userId'] = None
            if metrics['agentNames']: session['agent'] = sorted(list(metrics['agentNames']))[0]
            else: session['agent'] = session.get('agent', 'Unknown')
            updated_session_count += 1
            if not session_start_updated: sessions_without_trace_times += 1
        else:
            sessions_without_traces += 1
            session['traceCount'], session['totalTokens'], session['totalCost'], session['totalLatency'], session['durationSeconds'], session['userId'] = 0, 0, 0.0, 0.0, 0, None

    # Log session metrics statistics
    avg_duration_by_latency = sum(s['durationSeconds'] for s in all_sessions_processed) / max(1, len(all_sessions_processed))
    logging.info(f"Average session duration (by latency): {avg_duration_by_latency:.2f} seconds")
    print(f"   Average session duration: {avg_duration_by_latency:.2f} seconds")
    
    # Log detailed session duration metrics for the top 5 longest duration sessions
    top_sessions = sorted(all_sessions_processed, key=lambda s: s.get('durationSeconds', 0), reverse=True)[:5]
    logging.info(f"Top 5 longest duration sessions:")
    for idx, session in enumerate(top_sessions):
        session_id = session.get('id')
        duration = session.get('durationSeconds', 0)
        time_diff = 0
        if session.get('startTime') and session.get('endTime'):
            time_diff = (session.get('endTime') - session.get('startTime')).total_seconds()
        
        logging.info(f"{idx+1}. Session {session_id}: duration={duration:.2f}s, start={session.get('startTime')}, end={session.get('endTime')}")
        logging.info(f"   Time diff between start/end: {time_diff:.2f}s")
        logging.info(f"   Trace count: {session.get('traceCount')}")
        
    logging.info(f"Finished session aggregation. Updated {updated_session_count}/{total_session_count} sessions.")
    if sessions_without_traces > 0: logging.warning(f"{sessions_without_traces} sessions had no traces.")
    if sessions_without_trace_times > 0: logging.warning(f"{sessions_without_trace_times} sessions lacked trace times.")
    print(f"   Finished aggregating session metrics. Updated {updated_session_count} sessions.")

    # Additional diagnostic logging for observations with unusual latency values
    unusual_latency_observations = []
    zero_latency_observations = 0
    negative_latency_observations = 0
    extremely_long_latency_observations = 0
    
    for obs in all_observations_processed:
        latency = obs.get('latency', 0)
        if latency == 0:
            zero_latency_observations += 1
        elif latency < 0:  # This shouldn't happen but let's check
            negative_latency_observations += 1
            unusual_latency_observations.append(obs.get('id'))
        elif latency > 300:  # More than 5 minutes
            extremely_long_latency_observations += 1
            unusual_latency_observations.append(obs.get('id'))
            
    if unusual_latency_observations or zero_latency_observations > total_obs_count * 0.25:
        logging.warning(f"Observation latency analysis: {zero_latency_observations} zero latency, " +
                      f"{negative_latency_observations} negative latency, " +
                      f"{extremely_long_latency_observations} extremely long latency (>5min)")
        if unusual_latency_observations:
            sample_ids = unusual_latency_observations[:5]  # Log up to 5 unusual observations
            logging.warning(f"Sample unusual latency observation IDs: {sample_ids}")
    
    # --- Write Processed Data to JSON Files (Main Files) ---
    print("\n=== Writing Processed Data to JSON Files ===")
    logging.info("Starting to write processed data files.")
    processed_output_files = {
        "all_daily_metrics.json": all_daily_metrics_processed,
        "all_traces.json": all_traces_processed,
        "all_sessions.json": all_sessions_processed,
    }
    write_processed_observations = False
    if write_processed_observations and all_observations_processed:
         processed_output_files["all_observations_processed.json"] = all_observations_processed

    for filename, data_list in processed_output_files.items():
        filepath = os.path.join(data_dir, filename)
        print(f"   Writing {len(data_list)} items to {filepath}...")
        logging.info(f"Writing {len(data_list)} items to {filepath}")
        try:
            with open(filepath, "w") as f: json.dump(data_list, f, indent=2, cls=NpEncoder)
            logging.info(f"Successfully wrote {filepath}"); print(f"      Done.")
        except Exception as e: logging.error(f"Failed to write {filepath}: {e}", exc_info=True); print(f"ERROR writing {filepath}")


    # --- Generate and Write Summary Files ---
    print("\n=== Generating and Writing Summary Files ===")
    logging.info("Starting generation of summary JSON files.")
    summary_files_to_generate = {}

    # 1. agent_comparison.json
    agent_summary_data = defaultdict(lambda: {'agent': None, 'countTraces': 0, 'totalUsers': 0, 'totalMessages': 0, 'totalTokens': 0, 'totalCost': 0.0, '_userIds': set()})
    for trace in all_traces_processed:
         agent = trace.get('agent', 'Unknown')
         agent_summary_data[agent]['agent'] = agent; agent_summary_data[agent]['countTraces'] += 1
         agent_summary_data[agent]['totalMessages'] += trace.get('messageCount', 0)
         # Use trace-level tokens/cost for summary
         agent_summary_data[agent]['totalTokens'] += trace.get('totalTokens', 0)
         agent_summary_data[agent]['totalCost'] += trace.get('totalCost', 0.0)
         if trace.get('userId'): agent_summary_data[agent]['_userIds'].add(trace.get('userId'))
    agent_comparison_list = []
    for agent, data in agent_summary_data.items():
        trace_count = data['countTraces']; user_count = len(data['_userIds'])
        avg_msgs = (data['totalMessages'] / trace_count) if trace_count > 0 else 0
        agent_comparison_list.append({'agent': agent, 'countTraces': trace_count, 'totalUsers': user_count, 'avgMessagesPerConv': avg_msgs, 'totalTokens': data['totalTokens'], 'totalCost': data['totalCost']})
    agent_comparison_list.sort(key=lambda x: x['countTraces'], reverse=True)
    summary_files_to_generate['agent_comparison.json'] = agent_comparison_list

    # 2. user_metrics.json
    user_dict = defaultdict(lambda: {'totalTraces': 0, 'agents': set(), 'firstSeen': None, 'lastSeen': None, 'totalTokens': 0, 'totalCost': 0.0})
    for trace in all_traces_processed:
        user = trace.get('userId'); agent = trace.get('agent')
        if not user: continue
        ts = trace.get('endTime') or trace.get('timestamp')
        user_dict[user]['totalTraces'] += 1
        if agent: user_dict[user]['agents'].add(agent)
        # Use trace-level tokens/cost for summary
        user_dict[user]['totalTokens'] += trace.get('totalTokens', 0)
        user_dict[user]['totalCost'] += trace.get('totalCost', 0.0)
        first_seen, last_seen = user_dict[user]['firstSeen'], user_dict[user]['lastSeen']
        trace_start = trace.get('startTime') or trace.get('timestamp')
        if trace_start and isinstance(trace_start, datetime):
            if first_seen is None or trace_start < first_seen: user_dict[user]['firstSeen'] = trace_start
        if ts and isinstance(ts, datetime):
            if last_seen is None or ts > last_seen: user_dict[user]['lastSeen'] = ts
    user_metrics = [{'userId': user, 'totalTraces': info['totalTraces'], 'agents': sorted(list(info['agents'])), 'firstSeen': info['firstSeen'], 'lastSeen': info['lastSeen'], 'totalTokens': info['totalTokens'], 'totalCost': info['totalCost']} for user, info in user_dict.items()]
    user_metrics.sort(key=lambda x: x['lastSeen'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    summary_files_to_generate['user_metrics.json'] = user_metrics

    # 3. model_token_usage.json (Still aggregated from observations)
    model_token_usage = [{'model': model, 'totalTokens': stats['totalTokens'], 'totalCost': stats['totalCost'], 'observationCount': stats['observationCount']} for model, stats in model_usage_agg.items()]
    
    # Calculate total trace costs and total token counts for scaling
    total_trace_cost = sum(trace.get('totalCost', 0.0) for trace in all_traces_processed)
    total_model_cost_before = sum(model_data['totalCost'] for model_data in model_token_usage)
    
    logging.info(f"Total trace cost from API: ${total_trace_cost:.4f}")
    logging.info(f"Total model cost before adjustment: ${total_model_cost_before:.4f}")
    
    # If model costs are still 0, estimate based on model and token count
    estimated_cost_models = 0
    for model_data in model_token_usage:
        # Only estimate if cost is 0 but tokens exist
        if model_data['totalCost'] == 0.0 and model_data['totalTokens'] > 0:
            model_name = model_data['model'].lower()
            original_cost = model_data['totalCost']
            # Claude models (approximate pricing)
            if 'claude' in model_name and '3' in model_name:
                if 'sonnet' in model_name:
                    # Claude 3 Sonnet: ~$3.00/1M tokens (avg of input/output)
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 3.0
                elif 'opus' in model_name:
                    # Claude 3 Opus: ~$15.00/1M tokens (avg of input/output)
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 15.0
                elif 'haiku' in model_name:
                    # Claude 3 Haiku: ~$0.25/1M tokens (avg of input/output)
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 0.25
                else:
                    # Generic Claude 3: ~$6.00/1M tokens
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 6.0
            # Gemini models (approximate pricing)
            elif 'gemini' in model_name:
                if '1.5' in model_name and 'flash' in model_name:
                    # Gemini 1.5 Flash: ~$0.35/1M tokens (avg of input/output)
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 0.35
                elif '1.5' in model_name and 'pro' in model_name:
                    # Gemini 1.5 Pro: ~$3.50/1M tokens (avg of input/output)
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 3.50
                elif '2.0' in model_name:
                    # Gemini 2.0: ~$7.00/1M tokens (avg of input/output) - approximate
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 7.00
                else:
                    # Generic Gemini: ~$1.00/1M tokens
                    model_data['totalCost'] = (model_data['totalTokens'] / 1000000) * 1.00
            
            # Log the cost estimation
            if model_data['totalCost'] > 0:
                estimated_cost_models += 1
                logging.info(f"Estimated cost for {model_data['model']}: {original_cost} → {model_data['totalCost']:.4f} USD ({model_data['totalTokens']:,} tokens)")
    
    # Calculate total model cost after initial estimates
    total_model_cost_after = sum(model_data['totalCost'] for model_data in model_token_usage)
    
    # Apply scaling to match the actual API costs
    if total_model_cost_after > 0 and total_trace_cost > 0:
        scale_factor = total_trace_cost / total_model_cost_after
        logging.info(f"Scaling model costs by factor: {scale_factor:.4f} to match API trace costs")
        print(f"   Scaling model costs to match API trace costs (factor: {scale_factor:.4f})")
        
        for model_data in model_token_usage:
            original_cost = model_data['totalCost']
            model_data['totalCost'] = original_cost * scale_factor
            logging.info(f"Adjusted cost for {model_data['model']}: {original_cost:.4f} → {model_data['totalCost']:.4f} USD (scaled)")
    else:
        logging.warning(f"Could not scale model costs: total_model_cost={total_model_cost_after}, total_trace_cost={total_trace_cost}")
    
    logging.info(f"Applied cost estimates to {estimated_cost_models}/{len(model_token_usage)} models")
    print(f"   Applied cost estimates to {estimated_cost_models} models with missing cost data")
    
    model_token_usage.sort(key=lambda x: x['totalCost'], reverse=True)
    summary_files_to_generate['model_token_usage.json'] = model_token_usage

    # Write summary files
    for filename, data_content in summary_files_to_generate.items():
        filepath = os.path.join(data_dir, filename)
        print(f"   Writing {len(data_content)} summary items to {filepath}...")
        logging.info(f"Writing summary file {filepath} with {len(data_content)} items")
        try:
            with open(filepath, "w") as f: json.dump(data_content, f, indent=2, cls=NpEncoder)
            logging.info(f"Successfully wrote {filepath}"); print(f"      Done.")
        except Exception as e: logging.error(f"Failed to write summary {filepath}: {e}", exc_info=True); print(f"ERROR writing {filepath}")

    # --- Completion ---
    end_time_script = datetime.now()
    duration = end_time_script - start_time_script
    
    # Add very clear end markers for both console and log
    print("\n" + "="*80)
    print("SCRIPT_MARKER: LANGFUSE_DATA_FETCHER_COMPLETED")
    print(f"Langfuse data refresh COMPLETED at: {end_time_script}")
    print(f"Total execution time: {duration}")
    print(f"Raw API data saved to '{raw_data_dir}'. Processed data saved to '{data_dir}'.")
    print("="*80 + "\n")
    
    logging.info("="*50)
    logging.info("SCRIPT_MARKER: LANGFUSE_DATA_FETCHER_COMPLETED")
    logging.info(f"--- Langfuse Data Fetch Script Finished: {end_time_script} (Duration: {duration}) ---")
    logging.info(f"Raw API data saved to '{raw_data_dir}'. Processed data saved to '{data_dir}'.")
    logging.info("="*50)
    
    return True

if __name__ == "__main__":
    # Use May 1st, 2025 at 00:00 UTC as the start date
    may_1st_2025_utc = datetime(2025, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    print(f"Fetching data from {may_1st_2025_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    fetch_langfuse_data(may_1st_2025_utc)