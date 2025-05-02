import requests
import base64
import os
import json
from datetime import datetime, timedelta, timezone
# Removed pandas import as it's not used after refactor
# import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Set up logging
# Consider adding levelname to see INFO vs ERROR easily
logging.basicConfig(
    filename='langfuse_data_fetcher.log',
    filemode='a', # Append to the log file
    format='%(asctime)s - %(levelname)s - %(message)s', # Added levelname
    level=logging.INFO # Log INFO level and above (WARNING, ERROR, CRITICAL)
)
# Add a handler to also print log messages to console (optional)
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
        # Ensure datetime objects are converted to ISO format strings
        if isinstance(obj, datetime):
            # Make sure datetime is timezone-aware (assuming UTC if naive)
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        # Let the base class default method raise the TypeError
        return super(NpEncoder, self).default(obj)

def parse_datetime(date_string):
    """Safely parse ISO date strings, returning None if invalid."""
    if not date_string:
        logging.warning("Attempted to parse an empty or null date string.")
        return None
    try:
        # Handle 'Z' suffix for UTC
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        # If the datetime object is naive (no timezone), assume UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        # Otherwise, convert to UTC
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse datetime string '{date_string}': {e}")
        return None

def fetch_paginated_data(base_url, endpoint, headers, params):
    """Fetches all pages for a given Langfuse endpoint."""
    all_data = []
    page = 1
    # Extract time range for logging clarity
    time_range = f"from {params.get('fromTimestamp') or params.get('fromCreatedAt') or 'start'} to {params.get('toTimestamp') or params.get('toCreatedAt') or 'end'}"
    print(f"   Fetching paginated data for endpoint: {endpoint} ({time_range})...")
    logging.info(f"Initiating paginated fetch for {endpoint} ({time_range})")

    while True:
        params['page'] = page
        logging.debug(f"Fetching page {page} for {endpoint} with params: {params}")
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
            # Log status code for every request
            logging.info(f"Fetching {endpoint} page {page} - Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                     logging.error(f"JSON decode error on page {page} for {endpoint}: {e} - Response text: {response.text[:500]}...") # Log beginning of text
                     print(f"   ERROR: JSON decode error fetching {endpoint} page {page}. Check log.")
                     break # Stop fetching this endpoint for this agent

                batch = data.get('data', [])
                meta = data.get('meta', {})
                total_items = meta.get('totalItems')
                limit = meta.get('limit', params.get('limit', 100)) # Use limit from params as fallback
                total_pages = meta.get('totalPages')

                if batch:
                    all_data.extend(batch)
                    logging.info(f"Fetched {len(batch)} items on page {page} for {endpoint}. Total fetched: {len(all_data)}")
                    print(f"   Page {page}: Fetched {len(batch)} items. Total so far: {len(all_data)}")
                else:
                    logging.info(f"No more data found for {endpoint} on page {page}. Expected based on previous batch or meta.")
                    print(f"   Page {page}: No more items found. Fetch complete for {endpoint}.")
                    break # No data in this batch, assume end

                # More robust break conditions
                current_total = len(all_data)
                # 1. If total_pages is known and we've reached it
                if total_pages is not None and page >= total_pages:
                    logging.info(f"Reached total expected pages ({total_pages}) for {endpoint}. Stopping.")
                    print(f"   Reached expected total pages ({total_pages}). Fetch complete.")
                    break
                # 2. If total_items is known and we've fetched at least that many
                #    Use >= because totalItems might be an estimate
                if total_items is not None and current_total >= total_items:
                    logging.info(f"Fetched {current_total} items, reaching or exceeding expected totalItems ({total_items}) for {endpoint}. Stopping.")
                    print(f"   Fetched {current_total} items >= expected {total_items}. Fetch complete.")
                    break
                 # 3. If the last batch was smaller than the limit (indicates the last page)
                #    Only apply if limit is known and > 0
                if limit is not None and limit > 0 and len(batch) < limit:
                     logging.info(f"Fetched batch size {len(batch)} < limit {limit}. Assuming last page for {endpoint}. Stopping.")
                     print(f"   Last batch ({len(batch)}) was smaller than limit ({limit}). Fetch complete.")
                     break

                page += 1 # Prepare for the next page

            # Handle non-200 status codes
            elif response.status_code == 401:
                 logging.error(f"Authorization error (401) fetching {endpoint} page {page}. Check API keys for this agent.")
                 print(f"   ERROR: Authorization failed (401) for {endpoint}. Check API keys.")
                 break # Stop fetching for this agent
            elif response.status_code == 404:
                 logging.error(f"Endpoint not found (404) at {base_url}{endpoint}.")
                 print(f"   ERROR: Endpoint not found (404): {endpoint}")
                 break # Stop fetching, endpoint likely wrong
            else:
                logging.error(f"Failed to fetch {endpoint} page {page}: Status {response.status_code} - {response.text[:500]}...")
                print(f"   ERROR: Failed to fetch {endpoint} page {page} (Status: {response.status_code}). Check log.")
                # Optional: implement retries for transient errors (e.g., 5xx)
                break # Stop fetching this endpoint for this agent

        # Handle request exceptions (network issues, etc.)
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching {endpoint} page {page}: {e}", exc_info=True)
            print(f"   ERROR: Network or request error fetching {endpoint} page {page}. Check log.")
            break # Stop fetching this endpoint for this agent

    logging.info(f"Finished paginated fetch for {endpoint}. Total items retrieved: {len(all_data)}")
    print(f"   Finished fetching for {endpoint}. Total items: {len(all_data)}")
    return all_data


def fetch_langfuse_data(days_back=30):
    """
    Main function to fetch and process Langfuse data.
    """
    # --- Setup ---
    start_time_script = datetime.now()
    logging.info(f"--- Langfuse Data Fetch Script Started: {start_time_script} ---")
    logging.info(f"Fetching data for the last {days_back} days.")
    print(f"--- Langfuse Data Fetch Script Started ---")
    print(f"Fetching data for the last {days_back} days.")

    base_url = os.getenv("LANGFUSE_HOST", "https://langfuse.zoomrx.ai") # Ensure this is correct
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
        # Add other agents if needed
    ]
    logging.info(f"Found {len(agent_configs)} agent configurations.")
    print(f"Processing {len(agent_configs)} agent configurations...")


    data_dir = "dashboard_data"
    os.makedirs(data_dir, exist_ok=True)
    logging.info(f"Ensured data directory exists: {data_dir}")

    # Calculate date range in UTC and ISO format
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    # Ensure ISO format with 'Z' for UTC, required by Langfuse API timestamp filters
    start_date_str_iso = start_date.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    end_date_str_iso = end_date.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    # YYYY-MM-DD format for daily metrics API
    start_date_str_day = start_date.strftime("%Y-%m-%d")
    end_date_str_day = end_date.strftime("%Y-%m-%d")
    logging.info(f"Date range (ISO UTC): {start_date_str_iso} to {end_date_str_iso}")
    logging.info(f"Date range (YYYY-MM-DD): {start_date_str_day} to {end_date_str_day}")
    print(f"Date range: {start_date_str_iso} to {end_date_str_iso}")


    # Initialize lists to hold data across all agents
    all_traces_processed = []
    all_sessions_processed = []
    all_observations_raw = []
    all_daily_metrics_processed = []

    # --- Loop Through Agents ---
    for agent_config in agent_configs:
        agent_name = agent_config["name"]
        public_key = agent_config["public_key"]
        secret_key = agent_config["secret_key"]
        print(f"\n=== Processing Agent: {agent_name} ===")
        logging.info(f"--- Processing Agent: {agent_name} ---")


        if not public_key or not secret_key:
            logging.warning(f"API keys missing for {agent_name}. Skipping.")
            print(f"   WARNING: API keys missing for {agent_name}. Skipping.")
            continue

        # Prepare authentication headers
        auth_string = f"{public_key}:{secret_key}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/json"
        }
        logging.debug(f"Prepared headers for {agent_name}")

        # --- Fetch Data for Current Agent ---
        # Parameters using timestamps for most endpoints
        common_params_trace_obs = {
            "limit": 100,
            "fromTimestamp": start_date_str_iso, # Use ISO format with Z
            "toTimestamp": end_date_str_iso,     # Use ISO format with Z
        }

        # 1. Daily Metrics (uses date strings)
        print(f"   Fetching Daily Metrics for {agent_name}...")
        logging.info(f"Fetching daily metrics for {agent_name} ({start_date_str_day} to {end_date_str_day})")
        try:
            metrics_params = { "startDate": start_date_str_day, "endDate": end_date_str_day }
            metrics_response = requests.get(f"{base_url}/api/public/metrics/daily", headers=headers, params=metrics_params)
            logging.info(f"Daily Metrics API Status Code: {metrics_response.status_code}")
            print(f"   Daily Metrics API Status: {metrics_response.status_code}")

            if metrics_response.status_code == 200:
                metrics_data = metrics_response.json().get('data', [])
                logging.info(f"Fetched {len(metrics_data)} daily metric records for {agent_name}")
                print(f"   Fetched {len(metrics_data)} daily metric records.")
                # Process metrics data
                for i, day_data in enumerate(metrics_data):
                     log_prefix = f"DailyMetric {agent_name} Date {day_data.get('date')}:"
                     logging.debug(f"{log_prefix} Processing record {i+1}/{len(metrics_data)}")
                     day_metrics = {
                        'date': day_data.get('date'),
                        'agent': agent_name,
                        'countTraces': day_data.get('countTraces', 0),
                        'countObservations': day_data.get('countObservations', 0),
                        'totalCost': day_data.get('totalCost', 0.0) or 0.0, # Ensure float, default 0.0
                        'totalTokens': 0, # Initialize, sum from usage
                        'modelUsage': [] # Store detailed model usage
                     }
                     total_day_tokens = 0
                     for usage in day_data.get('usage', []):
                        # Get usage, default to 0 if null or missing
                        input_usage = usage.get('inputUsage', 0) or 0
                        output_usage = usage.get('outputUsage', 0) or 0
                        # Get totalUsage, default to sum of input/output if missing/null
                        tokens = usage.get('totalUsage')
                        if tokens is None:
                            tokens = input_usage + output_usage
                        else:
                            tokens = tokens or 0 # Ensure 0 if totalUsage is null

                        cost = usage.get('totalCost', 0.0) or 0.0 # Ensure float, default 0.0
                        model_name = usage.get('model', 'Unknown')

                        total_day_tokens += tokens
                        day_metrics['modelUsage'].append({
                             'model': model_name,
                             'inputUsage': input_usage,
                             'outputUsage': output_usage,
                             'totalUsage': tokens,
                             'countTraces': usage.get('countTraces', 0),
                             'countObservations': usage.get('countObservations', 0),
                             'totalCost': cost
                        })
                        # Add convenience fields for chart/table data structures
                        day_metrics[f'model_{model_name}_tokens'] = tokens
                        day_metrics[f'model_{model_name}_cost'] = cost
                        logging.debug(f"{log_prefix} Model: {model_name}, Tokens: {tokens}, Cost: {cost}")

                     day_metrics['totalTokens'] = total_day_tokens
                     all_daily_metrics_processed.append(day_metrics)
                     logging.info(f"{log_prefix} Processed Day - Total Tokens: {total_day_tokens}, Total Cost: {day_metrics['totalCost']}")

            else:
                 logging.error(f"Failed to fetch daily metrics for {agent_name}: {metrics_response.status_code} - {metrics_response.text[:500]}...")
                 print(f"   ERROR: Failed to fetch daily metrics (Status: {metrics_response.status_code}). Check log.")

        except Exception as e:
            logging.error(f"Exception processing daily metrics for {agent_name}: {e}", exc_info=True)
            print(f"   ERROR: Exception during daily metrics processing for {agent_name}. Check log.")


        # 2. Traces
        print(f"   Fetching Traces for {agent_name}...")
        agent_traces = fetch_paginated_data(base_url, "/api/public/traces", headers, common_params_trace_obs.copy())
        print(f"   Processing {len(agent_traces)} fetched traces for {agent_name}...")
        for i, trace in enumerate(agent_traces):
            timestamp = parse_datetime(trace.get('timestamp'))
            logging.debug(f"Processing trace {i+1}/{len(agent_traces)} (ID: {trace.get('id')})")
            all_traces_processed.append({
                'id': trace.get('id'),
                'timestamp': timestamp, # Use datetime object
                'name': trace.get('name'),
                'userId': trace.get('userId'),
                'sessionId': trace.get('sessionId'), # Get top-level sessionId
                'metadata': trace.get('metadata'), # Keep metadata if needed elsewhere
                'tags': trace.get('tags', []),
                'agent': agent_name,
                # Placeholders - will be updated after processing observations
                'totalTokens': 0,
                'totalCost': 0.0,
                'messageCount': 0,
                'startTime': timestamp, # Needed for session duration calculation, use trace ts as default
                'endTime': timestamp # Placeholder, update with last observation time
            })
        print(f"   Finished processing traces for {agent_name}.")


        # 3. Sessions (use 'fromCreatedAt'/'toCreatedAt')
        print(f"   Fetching Sessions for {agent_name}...")
        session_params = {
             "limit": 100,
             "fromCreatedAt": start_date_str_iso, # Use ISO format with Z
             "toCreatedAt": end_date_str_iso,     # Use ISO format with Z
        }
        agent_sessions = fetch_paginated_data(base_url, "/api/public/sessions", headers, session_params)
        print(f"   Processing {len(agent_sessions)} fetched sessions for {agent_name}...")
        for i, session in enumerate(agent_sessions):
            created_at = parse_datetime(session.get('createdAt'))
            logging.debug(f"Processing session {i+1}/{len(agent_sessions)} (ID: {session.get('id')})")
            all_sessions_processed.append({
                'id': session.get('id'),
                'createdAt': created_at, # Use datetime object
                'projectId': session.get('projectId'), # Included from API example
                'agent': agent_name,
                 # Placeholders - will be updated after processing traces/observations
                'userId': None, # Get from associated traces later
                'durationSeconds': 0,
                'traceCount': 0,
                'totalTokens': 0,
                'totalCost': 0.0,
                'startTime': created_at, # Use createdAt as default start
                'endTime': created_at # Placeholder, update with last trace/obs time
            })
        print(f"   Finished processing sessions for {agent_name}.")


        # 4. Observations (Explain potential slowness)
        print(f"   Fetching Observations for {agent_name}... (This may take time depending on data volume)")
        logging.info(f"Fetching observations for {agent_name}. Note: This can be slow for large date ranges/data volumes.")
        agent_observations = fetch_paginated_data(base_url, "/api/public/observations", headers, common_params_trace_obs.copy())
        print(f"   Processing {len(agent_observations)} fetched observations for {agent_name}...")
        for i, obs in enumerate(agent_observations):
            # Use startTime and endTime from observation response example
            start_time = parse_datetime(obs.get('startTime'))
            end_time = parse_datetime(obs.get('endTime'))
            usage_data = obs.get('usage', {}) or {} # Ensure usage is a dict, default empty

            log_prefix = f"Obs {agent_name} ID {obs.get('id')}:"
            logging.debug(f"{log_prefix} Processing observation {i+1}/{len(agent_observations)} (TraceID: {obs.get('traceId')})")

            # Extract tokens/cost from usage object, handle potential nulls/missing keys
            input_tokens = usage_data.get('input', 0) or 0
            output_tokens = usage_data.get('output', 0) or 0
            total_tokens = usage_data.get('total') # Get total first
            # If total is explicitly null or missing, fall back to summing input/output
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens
            else:
                total_tokens = total_tokens or 0 # Ensure 0 if total is null

            # Cost: Use totalCost if available, otherwise sum input/output costs
            input_cost = usage_data.get('inputCost', 0.0) or 0.0
            output_cost = usage_data.get('outputCost', 0.0) or 0.0
            total_cost = usage_data.get('totalCost') # Get totalCost
            if total_cost is None:
                 total_cost = input_cost + output_cost
            else:
                 total_cost = total_cost or 0.0 # Ensure 0.0 if totalCost is null


            logging.debug(f"{log_prefix} Usage - InTok: {input_tokens}, OutTok: {output_tokens}, TotalTok: {total_tokens}, TotalCost: {total_cost:.6f}")

            all_observations_raw.append({
                'id': obs.get('id'),
                'traceId': obs.get('traceId'),
                'type': obs.get('type'),
                'name': obs.get('name'),
                'startTime': start_time, # Use datetime object
                'endTime': end_time,     # Use datetime object
                'model': obs.get('model'), # Get model directly from observation
                'inputTokens': input_tokens,
                'outputTokens': output_tokens,
                'totalTokens': total_tokens,
                'totalCost': total_cost, # Use calculated total_cost
                'level': obs.get('level'),
                'statusMessage': obs.get('statusMessage'),
                'parentObservationId': obs.get('parentObservationId'),
                'metadata': obs.get('metadata'),
                'agent': agent_name # Add agent name for easier processing
            })
        print(f"   Finished processing observations for {agent_name}.")

        logging.info(f"--- Finished fetching data for agent: {agent_name} ---")
        print(f"=== Finished fetching data for {agent_name} ===")

    # --- Data Aggregation (Across All Agents) ---
    total_obs_count = len(all_observations_raw)
    total_trace_count = len(all_traces_processed)
    total_session_count = len(all_sessions_processed)
    print(f"\n=== Starting Data Aggregation Across All Agents ===")
    print(f"   Total Observations Fetched: {total_obs_count}")
    print(f"   Total Traces Fetched: {total_trace_count}")
    print(f"   Total Sessions Fetched: {total_session_count}")
    logging.info(f"Starting Aggregation. Observations: {total_obs_count}, Traces: {total_trace_count}, Sessions: {total_session_count}")

    # 1. Aggregate Observation Data per Trace
    print("   Aggregating observation data per trace...")
    logging.info("Aggregating observation data per trace...")
    # Initialize trace_metrics: Use defaultdict for cleaner aggregation
    # Store startTime and endTime as potential datetime objects
    trace_metrics = defaultdict(lambda: {'totalTokens': 0, 'totalCost': 0.0, 'messageCount': 0, 'startTime': None, 'endTime': None})
    # Aggregate model usage across ALL observations (not just per trace)
    model_usage_agg = defaultdict(lambda: {'totalTokens': 0, 'totalCost': 0.0, 'observationCount': 0})

    for i, obs in enumerate(all_observations_raw):
        trace_id = obs.get('traceId')
        if (i + 1) % 5000 == 0: # Log progress every 5000 observations
             logging.info(f"Processing observation aggregation: {i+1}/{total_obs_count}")
             print(f"      Aggregating observation {i+1}/{total_obs_count}...")
        if not trace_id:
            logging.warning(f"Observation {obs.get('id')} has no traceId. Skipping trace aggregation for this obs.")
            continue

        # Aggregate core metrics
        trace_metrics[trace_id]['totalTokens'] += obs.get('totalTokens', 0)
        trace_metrics[trace_id]['totalCost'] += obs.get('totalCost', 0.0)
        trace_metrics[trace_id]['messageCount'] += 1

        # Track min start and max end times for the trace using observation times
        obs_start_time = obs.get('startTime') # Already a datetime object or None
        obs_end_time = obs.get('endTime') or obs_start_time # Fallback to start time if end time is null

        current_trace_start = trace_metrics[trace_id]['startTime']
        current_trace_end = trace_metrics[trace_id]['endTime']

        if obs_start_time:
            if current_trace_start is None or obs_start_time < current_trace_start:
                trace_metrics[trace_id]['startTime'] = obs_start_time
        if obs_end_time: # Should be a datetime object or None
             if current_trace_end is None or obs_end_time > current_trace_end:
                 trace_metrics[trace_id]['endTime'] = obs_end_time

        # Aggregate model usage globally
        model = obs.get('model')
        if model: # Only aggregate if model is present
             model_key = model or "Unknown" # Handle None or empty string models
             model_usage_agg[model_key]['totalTokens'] += obs.get('totalTokens', 0)
             model_usage_agg[model_key]['totalCost'] += obs.get('totalCost', 0.0)
             model_usage_agg[model_key]['observationCount'] += 1


    # Update traces list with aggregated metrics
    updated_trace_count = 0
    for trace in all_traces_processed:
        trace_id = trace.get('id')
        if trace_id in trace_metrics:
            metrics = trace_metrics[trace_id]
            trace.update(metrics) # Add aggregated tokens, cost, messageCount, startTime, endTime
             # Use trace timestamp as fallback ONLY if observation times were completely missing for this trace
            if trace['startTime'] is None:
                logging.warning(f"Trace {trace_id} had no observations with startTime, using trace timestamp as fallback.")
                trace['startTime'] = trace['timestamp']
            if trace['endTime'] is None:
                logging.warning(f"Trace {trace_id} had no observations with endTime, using startTime as fallback.")
                trace['endTime'] = trace['startTime'] # Fallback to calculated start time
            updated_trace_count += 1
        else:
             # If a trace has no observations, its metrics remain 0, startTime/endTime remain trace timestamp
             logging.warning(f"Trace {trace_id} had no associated observations in fetched data.")
             if trace['startTime'] is None: trace['startTime'] = trace['timestamp']
             if trace['endTime'] is None: trace['endTime'] = trace['timestamp']

    logging.info(f"Finished aggregating trace metrics. Updated {updated_trace_count}/{total_trace_count} traces with observation data.")
    print(f"   Finished aggregating trace metrics. Updated {updated_trace_count} traces.")

    # 2. Aggregate Trace Data per Session
    print("   Aggregating trace data per session...")
    logging.info("Aggregating trace data per session...")
    session_metrics = defaultdict(lambda: {'traceCount': 0, 'totalTokens': 0, 'totalCost': 0.0, 'userIds': set(), 'startTime': None, 'endTime': None})

    # Group traces by session ID first
    traces_grouped_by_session = defaultdict(list)
    for i, trace in enumerate(all_traces_processed):
         session_id = trace.get('sessionId')
         if (i + 1) % 1000 == 0: # Log progress
             logging.info(f"Processing trace for session aggregation: {i+1}/{total_trace_count}")
             # print(f"      Aggregating trace {i+1}/{total_trace_count} for session...") # Can be too verbose
         if session_id:
             traces_grouped_by_session[session_id].append(trace)

    sessions_found_in_traces = len(traces_grouped_by_session)
    logging.info(f"Found {sessions_found_in_traces} unique session IDs referenced in traces.")
    print(f"   Found {sessions_found_in_traces} unique session IDs in traces.")

    # Calculate session metrics from grouped traces
    for session_id, traces_in_session in traces_grouped_by_session.items():
        log_prefix = f"Session {session_id}:"
        logging.debug(f"{log_prefix} Aggregating metrics with {len(traces_in_session)} traces.")
        session_start = None
        session_end = None
        for trace in traces_in_session:
            session_metrics[session_id]['traceCount'] += 1
            session_metrics[session_id]['totalTokens'] += trace.get('totalTokens', 0) # Use aggregated trace tokens
            session_metrics[session_id]['totalCost'] += trace.get('totalCost', 0.0)   # Use aggregated trace cost
            if trace.get('userId'):
                session_metrics[session_id]['userIds'].add(trace.get('userId'))

            # Track session start/end based on its traces' start/end times (which came from observations)
            trace_start_time = trace.get('startTime') # Already a datetime object or None
            trace_end_time = trace.get('endTime')     # Already a datetime object or None

            if trace_start_time:
                if session_start is None or trace_start_time < session_start:
                    session_start = trace_start_time
            if trace_end_time:
                 if session_end is None or trace_end_time > session_end:
                     session_end = trace_end_time
        # Store the calculated start/end for the session
        session_metrics[session_id]['startTime'] = session_start
        session_metrics[session_id]['endTime'] = session_end
        logging.debug(f"{log_prefix} Aggregated Tokens: {session_metrics[session_id]['totalTokens']}, Cost: {session_metrics[session_id]['totalCost']:.6f}, Start: {session_start}, End: {session_end}")


    # Update the main session list with aggregated metrics
    updated_session_count = 0
    missing_session_ids = 0
    for session in all_sessions_processed:
        session_id = session.get('id')
        if session_id in session_metrics:
            metrics = session_metrics[session_id]
            session.update(metrics) # Add traceCount, totalTokens, totalCost, userIds, startTime, endTime

            # Use session createdAt as fallback ONLY if trace times were missing
            if session['startTime'] is None:
                 logging.warning(f"Session {session_id} had no traces with startTime, using session createdAt as fallback.")
                 session['startTime'] = session['createdAt']
            if session['endTime'] is None:
                 logging.warning(f"Session {session_id} had no traces with endTime, using calculated startTime as fallback.")
                 session['endTime'] = session['startTime'] # Fallback to calculated start time

            # Calculate duration ONLY if we have valid start and end times
            if session['startTime'] and session['endTime']:
                 # Ensure both are datetime objects before subtraction
                 if isinstance(session['startTime'], datetime) and isinstance(session['endTime'], datetime):
                     duration = session['endTime'] - session['startTime']
                     session['durationSeconds'] = max(0, duration.total_seconds())
                     logging.debug(f"Session {session_id} duration: {session['durationSeconds']:.2f}s")
                 else:
                      logging.warning(f"Session {session_id} has invalid start/end time types, cannot calculate duration.")
                      session['durationSeconds'] = 0
            else:
                session['durationSeconds'] = 0


            # Assign a userId (e.g., the first one found alphabetically)
            if session['userIds']:
                session['userId'] = sorted(list(session['userIds']))[0]
            else:
                 session['userId'] = None # Ensure userId is None if no traces had one
            del session['userIds'] # Clean up temporary set

            updated_session_count += 1
        else:
            # This session ID was fetched but no traces referenced it. Keep its metrics as 0/None.
            logging.warning(f"Session {session_id} fetched but no associated traces found in the fetched trace data. Metrics will be zero/defaults.")
            missing_session_ids += 1
            # Ensure defaults are set for aggregation fields if not updated
            session['traceCount'] = 0
            session['totalTokens'] = 0
            session['totalCost'] = 0.0
            session['userId'] = None
            session['durationSeconds'] = 0
            # startTime/endTime will remain session.createdAt from initial processing


    logging.info(f"Finished aggregating session metrics. Updated {updated_session_count}/{total_session_count} sessions with trace data.")
    if missing_session_ids > 0:
        logging.warning(f"{missing_session_ids} sessions had no corresponding traces found in the fetched data.")
    print(f"   Finished aggregating session metrics. Updated {updated_session_count} sessions.")
    if missing_session_ids > 0:
        print(f"   Warning: {missing_session_ids} sessions had no corresponding traces found.")


    # --- Write Processed Data to JSON Files ---
    print("\n=== Writing Processed Data to JSON Files ===")
    logging.info("Starting to write processed data to JSON files.")

    # Ensure NpEncoder handles datetime correctly before writing
    output_files = {
        "all_daily_metrics.json": all_daily_metrics_processed,
        "all_traces.json": all_traces_processed,
        "all_sessions.json": all_sessions_processed,
        # Only write observations if truly needed, as it's large and slows down fetching
        # "all_observations.json": all_observations_raw,
    }
    # Decide whether to write the large observations file
    write_observations_file = False # Set to True if needed for debugging/detail views
    if write_observations_file:
        output_files["all_observations.json"] = all_observations_raw
        print("   Including all_observations.json in output.")
        logging.info("Including all_observations.json in output.")
    else:
        print("   Skipping write of all_observations.json (can be large).")
        logging.info("Skipping write of all_observations.json.")


    for filename, data_list in output_files.items():
        filepath = os.path.join(data_dir, filename)
        print(f"   Writing {len(data_list)} items to {filepath}...")
        logging.info(f"Writing {len(data_list)} items to {filepath}")
        try:
            with open(filepath, "w") as f:
                json.dump(data_list, f, indent=2, cls=NpEncoder) # Use NpEncoder
            logging.info(f"Successfully wrote {filepath}")
            print(f"      Done.")
        except TypeError as e:
            logging.error(f"TypeError writing {filepath}: {e}. Check NpEncoder for unhandled types.", exc_info=True)
            print(f"      ERROR: TypeError writing {filepath}. Check logs.")
        except Exception as e:
            logging.error(f"Failed to write {filepath}: {e}", exc_info=True)
            print(f"      ERROR: Failed to write {filepath}. Check logs.")


    # --- Generate and Write Summary Files ---
    print("\n=== Generating and Writing Summary Files ===")
    logging.info("Starting generation of summary JSON files.")

    summary_files_to_generate = {}

    # 1. agent_comparison.json (Use Agent names from configs as the source of truth)
    agent_comparison_data = defaultdict(lambda: {'countTraces': 0})
    for trace in all_traces_processed:
        agent_comparison_data[trace['agent']]['countTraces'] += 1
    # Ensure all configured agents are present, even if they have 0 traces
    summary_files_to_generate['agent_comparison.json'] = [
        {'agent': cfg['name'], 'countTraces': agent_comparison_data[cfg['name']]['countTraces']}
        for cfg in agent_configs
    ]

    # 2. daily_tokens_by_agent.json & 3. daily_cost_by_agent.json
    daily_tokens = defaultdict(lambda: defaultdict(int))
    daily_cost = defaultdict(lambda: defaultdict(float))
    # Use the already processed daily metrics which contain agent info
    for row in all_daily_metrics_processed:
        date_str = row['date'] # Already YYYY-MM-DD string
        agent = row['agent']
        # Use .get() with default 0 to handle potentially missing keys robustly
        daily_tokens[date_str][agent] += row.get('totalTokens', 0)
        daily_cost[date_str][agent] += row.get('totalCost', 0.0)
    # Convert to list format for JSON, ensuring agents with 0 usage on a day are included if needed
    # Get all unique agents across all days
    all_agents_in_period = set(row['agent'] for row in all_daily_metrics_processed)
    daily_tokens_list = []
    for date in sorted(daily_tokens.keys()):
        entry = {'date': date}
        # Ensure all agents have an entry for this date, defaulting to 0
        for agent in all_agents_in_period:
            entry[agent] = daily_tokens[date].get(agent, 0)
        daily_tokens_list.append(entry)
    daily_cost_list = []
    for date in sorted(daily_cost.keys()):
         entry = {'date': date}
         for agent in all_agents_in_period:
             entry[agent] = daily_cost[date].get(agent, 0.0)
         daily_cost_list.append(entry)

    summary_files_to_generate['daily_tokens_by_agent.json'] = daily_tokens_list
    summary_files_to_generate['daily_cost_by_agent.json'] = daily_cost_list


    # 4. user_metrics.json (Aggregated from processed traces)
    user_dict = defaultdict(lambda: {
        'totalTraces': 0, 'agents': set(), 'firstSeen': None, 'lastSeen': None,
        'totalTokens': 0, 'totalCost': 0.0
    })
    for trace in all_traces_processed:
        user = trace.get('userId')
        if not user: continue # Skip traces without userId
        agent = trace.get('agent')
        ts = trace.get('timestamp') # Trace timestamp (already datetime obj)

        user_dict[user]['totalTraces'] += 1
        if agent: user_dict[user]['agents'].add(agent)
        user_dict[user]['totalTokens'] += trace.get('totalTokens', 0) # Use aggregated from trace
        user_dict[user]['totalCost'] += trace.get('totalCost', 0.0)   # Use aggregated from trace

        # Update first/last seen times using datetime comparison
        first_seen = user_dict[user]['firstSeen']
        last_seen = user_dict[user]['lastSeen']
        if ts: # Ensure ts is a valid datetime object
            if first_seen is None or ts < first_seen: user_dict[user]['firstSeen'] = ts
            if last_seen is None or ts > last_seen: user_dict[user]['lastSeen'] = ts

    user_metrics = []
    for user, info in user_dict.items():
        user_metrics.append({
            'userId': user,
            'totalTraces': info['totalTraces'],
            'agents': sorted(list(info['agents'])), # Sort agents for consistency
            'firstSeen': info['firstSeen'], # Keep as datetime for NpEncoder
            'lastSeen': info['lastSeen'],   # Keep as datetime for NpEncoder
            'totalTokens': info['totalTokens'],
            'totalCost': info['totalCost']
        })
    # Sort users by last seen date (most recent first), handling potential None values
    user_metrics.sort(key=lambda x: x['lastSeen'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    summary_files_to_generate['user_metrics.json'] = user_metrics


    # 5. agent_distribution.json (Based on trace counts per agent)
    # Reusing agent_comparison_data calculated earlier
    summary_files_to_generate['agent_distribution.json'] = [
        {'agent': cfg['name'], 'count': agent_comparison_data[cfg['name']]['countTraces']}
         for cfg in agent_configs
    ]

    # 6. conversation_lengths.json (Use messageCount from aggregated trace data)
    summary_files_to_generate['conversation_lengths.json'] = [
        {'conversationId': trace['id'], 'length': trace.get('messageCount', 0)}
        for trace in all_traces_processed if trace.get('messageCount', 0) > 0
    ]

    # 7. model_token_usage.json (Use the globally aggregated model data)
    model_token_usage = []
    for model, stats in model_usage_agg.items():
        model_token_usage.append({
            'model': model, # Already includes 'Unknown' if applicable
            'totalTokens': stats['totalTokens'],
            'totalCost': stats['totalCost'],
            'observationCount': stats['observationCount']
        })
    # Sort by cost (highest first)
    model_token_usage.sort(key=lambda x: x['totalCost'], reverse=True)
    summary_files_to_generate['model_token_usage.json'] = model_token_usage


    # Write all summary files
    for filename, data_content in summary_files_to_generate.items():
        filepath = os.path.join(data_dir, filename)
        print(f"   Writing {len(data_content)} summary items to {filepath}...")
        logging.info(f"Writing summary file {filepath} with {len(data_content)} items")
        try:
            with open(filepath, "w") as f:
                json.dump(data_content, f, indent=2, cls=NpEncoder) # Use NpEncoder
            logging.info(f"Successfully wrote {filepath}")
            print(f"      Done.")
        except TypeError as e:
            logging.error(f"TypeError writing summary file {filepath}: {e}. Check NpEncoder.", exc_info=True)
            print(f"      ERROR: TypeError writing {filepath}. Check logs.")
        except Exception as e:
            logging.error(f"Failed to write summary file {filepath}: {e}", exc_info=True)
            print(f"      ERROR: Failed to write {filepath}. Check logs.")

    # --- Completion ---
    end_time_script = datetime.now()
    duration = end_time_script - start_time_script
    logging.info(f"--- Langfuse Data Fetch Script Finished: {end_time_script} ---")
    logging.info(f"Total execution time: {duration}")
    print(f"\n=== Data Processing Complete ===")
    print(f"Total execution time: {duration}")
    print(f"All data processed and saved to the '{data_dir}' directory.")
    logging.info("Script finished successfully.")
    return True

if __name__ == "__main__":
    # Fetch data for the last 7 days by default to reduce load, adjust as needed
    fetch_langfuse_data(days_back=7) # Reduced default days_back