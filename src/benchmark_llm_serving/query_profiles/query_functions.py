import sys
import json
import aiohttp
import argparse
import traceback
from typing import List
from datetime import datetime

from benchmark_llm_serving.backends import BackEnd
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput


async def query_function(query_input: QueryInput, session: aiohttp.ClientSession, completions_url: str, results: List[QueryOutput],
                        args: argparse.Namespace, backend: BackEnd) -> QueryOutput:
    """Queries the completions API to get the output

    Args:
        query_input (QueryInput) : The query input to use
        session (aiohttp.ClientSession) : The aiohttp session
        completions_url (str) : The url of the completions API
        results (list) : The list of results to which we will add the output
        args (argparse.Namespace) : The cli args
        backend (Backend) : The backend to consider
    
    Returns:
        QueryOutput : The output of the query
    """
    body = backend.get_payload(query_input, args)
    headers = backend.get_completions_headers()
    output = QueryOutput()
    output.starting_timestamp = datetime.now().timestamp()
    output.prompt = query_input.prompt
    most_recent_timestamp = output.starting_timestamp
    try:
        async with session.post(url=completions_url, json=body, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    
                    # Some backends add a prefix to the response. We remove it
                    chunk = chunk_bytes.decode("utf-8")
                    chunk = backend.remove_response_prefix(chunk)
                    # Some backends send useless messages. We don't consider them
                    if backend.test_chunk_validity(chunk):
                        # If the stream is ending, we save the timestamp as ending time
                        if backend.check_end_of_stream(chunk):
                            output.ending_timestamp = datetime.now().timestamp()
                            output.success = True
                        # Otherwise, we add the response to the already generated text
                        else:
                            timestamp = datetime.now().timestamp()
                            json_chunk = json.loads(chunk)
                            newly_generated_text = backend.get_newly_generated_text(json_chunk)
                            if len(newly_generated_text):
                                output.timestamp_of_tokens_arrival.append(timestamp)
                                output.generated_text += newly_generated_text
                            backend.add_prompt_length(json_chunk, output)
            else:
                output.success = False
                output.error = response.reason or ""
    except:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
    # We add the result to the rest of the results
    results.append(output)
    return output