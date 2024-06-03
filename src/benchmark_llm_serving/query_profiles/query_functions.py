import sys
import json
import aiohttp
import argparse
import traceback
from typing import List
from datetime import datetime

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
    body = {
        "prompt": query_input.prompt,
        "model": args.model,
        "max_tokens": args.output_length,
        "min_tokens": args.output_length,
        "temperature": 0,
        "repetition_penalty": 1.2,
        "stream": True
                }
    output = QueryOutput()
    output.starting_timestamp = datetime.now().timestamp()
    output.prompt = query_input.prompt
    most_recent_timestamp = output.starting_timestamp
    try:
        async with session.post(url=completions_url, json=body) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    # The OpenAI API prefixes its response by "data: " so we remove it
                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    # If the stream is ending, we save the timestamp as ending time
                    if chunk == "[DONE]":
                        output.ending_timestamp = datetime.now().timestamp()
                        output.success = True
                    # Otherwise, we add the response to the already generated text
                    else:
                        timestamp = datetime.now().timestamp()
                        output.timestamp_of_tokens_arrival.append(timestamp)
                        json_chunk = json.loads(chunk)
                        data = json_chunk['choices'][0]['text']
                        output.generated_text += data
                        if "usage" in json_chunk:
                            if json_chunk['usage'] is not None:
                                output.prompt_length = json_chunk['usage']['prompt_tokens']
                        
                        
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