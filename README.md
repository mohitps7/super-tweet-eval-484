# tweet_eval_484

Evaluates tweet_eval tasks on the latest large language models as of April 2025.

Install dependencies:
`pip install -r requirements.txt`

Example to run tasks (currently supports `topic`, `sentiment`, `hate` tasks):
```
python3 classify_tweets.py \
--api-key API_KEY \
[--model MODEL_NAME] \
[--max-samples N] \
[--throttle SECONDS] \
[--output-path PATH] \
[--tasks TASK1 TASK2 ...]
```
e.g.

```
python3 classify_tweets.py \
--api-key API_KEY \
--max-samples 20
--throttle 20
```