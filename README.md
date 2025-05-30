# TheElicitors
Eliciting Reasoning in LLMs Using Logprob-Based Rewards done during EPFL EE-556 Reinforcement Learning Course

## Project Overview
We explore inference-time reinforcement learning to elicit reasoning in language models using Group Relative Policy Optimization (GRPO) with rule-based and logprob-based rewards. Our approach includes a custom loss masking strategy to align training with reasoning-only rewards. Evaluated on math and poetry tasks, logprob rewards improve reasoning plausibility and generalization. These results highlight the potential of logprob signals as domain-agnostic rewards for structured and creative reasoning.

## Environment Setup
To run this project, please follow these steps:
1. **Python Packages:**  
   Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```
2. **TRL and vLLM Setup for Custom LogProb GRPO:**  
   To run our Custom LogProb GRPO, we must patch the `trl/scripts/vllm_serve.py` file by updating the `generate` function as described below. This fix is needed because, by default, trl generation doesn't provide log probabilities.  
     
   Please see the **Log-Probability Reward Model Setup** Section below for detailed patch instructions.

## Folder Structure
Below is a detailed summary of the important folders in the **src** repository:
- **scripts:**  
  Contains the training scripts that demonstrate how to run training jobs for the GRPO Trainer in rule-based and logprob-based settings. One can also modify the training arguments and task (math or poem-completion).
- **evaluation:**  
  Contains the evaluation code including helper scripts and the `run_eval.sh` script (which can also be executed from the root) used for running experiments on the generated completions.
- **train:**  
  Contains the Python files for training using either rule-based or logprob-based GRPO. This folder also includes helper modules that define rewards, data processing, and additional support functions.
- **recipes:**  
  Contains configuration and training argument files specifically for the GRPO Trainer.
- **notebooks:**  
  Hosts the Jupyter Notebook used to create and process the poetry dataset.
- **configs:**  
  Contains configuration scripts for deepspeed training. This is especially useful for multi-node training setups.


## Training Scripts
Training scripts can be found in the `src/scripts` folder. They demonstrate how to run training jobs for the GRPO Trainers using the arguments provided in the `src/recipes` folder. 

To start a training session, simply run the following script from the root directory:
```bash
sbatch scripts/run_train_V100.sh
```
_Note: You will need to adapt the script depending on your hardware requirements._

## Log-Probability Reward Model Setup
In order to run the logprob-based reward model, you need to change a file in the trl library : 
- first make sure it is installed in your environment (`pip install trl[vllm]`)
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


## Experiments and Results

For a detailed account of our experimental procedures, methodologies, and findings, please refer to the project report located at the root of this directory, `TheElicitorsFinalReport.pdf`.