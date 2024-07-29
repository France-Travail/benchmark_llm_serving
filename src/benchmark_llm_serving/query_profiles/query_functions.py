import sys
import json
import aiohttp
import argparse
import traceback
from typing import List
from datetime import datetime

from benchmark_llm_serving import backends
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput


async def query_function(query_input: QueryInput, session: aiohttp.ClientSession, completions_url: str, results: List[QueryOutput],
                        args: argparse.Namespace) -> QueryOutput:
    """Queries the completions API to get the output

    Args:
        query_input (QueryInput) : The query input to use
        session (aiohttp.ClientSession) : The aiohttp session
        completions_url (str) : The url of the completions API
        results (list) : The list of results to which we will add the output
        args (argparse.Namespace) : The cli args
    
    Returns:
        QueryOutput : The output of the query
    """
    body = backends.get_payload(query_input, args)
    headers = backends.get_completions_headers(args)
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
                    chunk = backends.decode_remove_response_prefix(chunk_bytes, args)
                    # Some backends send useless messages. We don't consider them
                    if backends.test_chunk_validity(chunk, args):
                        # If the stream is ending, we save the timestamp as ending time
                        if backends.check_end_of_stream(chunk, args):
                            output.ending_timestamp = datetime.now().timestamp()
                            output.success = True
                        # Otherwise, we add the response to the already generated text
                        else:
                            timestamp = datetime.now().timestamp()
                            json_chunk = json.loads(chunk)
                            newly_generated_text = backends.get_newly_generated_text(json_chunk, args)
                            if len(newly_generated_text):
                                output.timestamp_of_tokens_arrival.append(timestamp)
                                output.generated_text += newly_generated_text
                            backends.add_prompt_length(json_chunk, output, args)
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