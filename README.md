# TheElicitors
Eliciting Reasoning in LLMs Using Logprob-Based Rewards done during EE-556 Reinforcement Learning Course

| Model Name   | AIME24 (30 questions) | AIME25 (30 questions) | MATH_500 (500 questions) | GPQA:diamond (198 questions) | U_math (900 questions) |
|--------------|--------|--------|----------|--------------|--------|
|google/gemma-3-1b-it| 0.0333 |0.0|0.434|0.288|0.128|
|Jeremmmyyyyy/gemma-3-1b-Math|0.0 |0.0|0.446|0.032|0.13|
|||||||


### MATH : accuracy score on the test set
|Reward|no normalization|length normalization|z-score|min-max|
|--------------|--------------|--------|--------|--------|
|Baseline|0.0085|-|-|-|
|Rule based |0.468|*|-|*|
|Log Probabilities||||

* here the two stars indicate that 

### Poetry : average rewards over all the samples in the test set
|Reward|no normalization|length normalization|z-score|min-max|
|--------------|--------------|--------|--------|--------|
|Baseline|0.0*|-|-|-|
|Rule based ||-|-|-|
|Log Probabilities||||


In order to run the logprob-based reward model, you need to change a file in the trl library : 
- first make sure it is installed in your environment (pip install trl[vllm])
- Then go to the trl library folder. Check where all the packages are installed with the following command:
```bash
pip list | tail -n +3 | xargs -exec pip show
```
Once you are in the trl folder, go to the trl/scripts/vllm_serve.py and go down until you find the function 
```python
@app.post("/generate/", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
```

Update the function to the following code:
```python
@app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
            logprobs=1,
        )
        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        
        logprobs = [output.cumulative_logprob for outputs in all_outputs for output in outputs.outputs]
        print(logprobs)
        return {
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            }
```
This will allow you to get the CUMULATIVE logprobs of the generated tokens along with the generated tokens ids.

See [here](https://github.com/vllm-project/vllm/issues/5234) for implementation details.

