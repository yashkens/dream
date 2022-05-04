import logging
import time
import os

import sentry_sdk
import torch
from flask import Flask, request, jsonify
from sentry_sdk.integrations.flask import FlaskIntegration
from transformers import AutoModelForCausalLM, AutoTokenizer

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
import RAKE

stop_words = stopwords.words('english')
rake = RAKE.Rake(stop_words)

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME_OR_PATH = os.environ.get("PRETRAINED_MODEL_NAME_OR_PATH")
logging.info(f"PRETRAINED_MODEL_NAME_OR_PATH = {PRETRAINED_MODEL_NAME_OR_PATH}")
DEFAULT_CONFIDENCE = 0.9
ZERO_CONFIDENCE = 0.0
# MAX_HISTORY_DEPTH = 3

try:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("dialogpt is set to run on cuda")

    logger.info("dialogpt is ready")
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel("WARNING")


def get_keywords(sent):
    words = rake.run(sent.lower(), minFrequency=1, maxWords=1)
    keywords = [word[0] for word in words]
    return keywords


def generate_response(context, model, tokenizer):
    last_utt = context[-1]
    logger.info(f"Last utterance: {last_utt}")
    # keywords = get_keywords(last_utt)
    # logger.info(f"Keywords:{' # '.join(keywords)}")
    # title = 'Cats'
    # if keywords:
    #     title = keywords[0]
    # else:
    #     logger.info(f"No keywords")
    # input_ids = tokenizer.encode(title + ' <EOT> ', return_tensors="pt")
    # tmp_prompt = 'rick <EOT> rick grew troubled # family turned gangs # long robbery # turn # happy <EOL>'
    tmp_prompt = 'My cat'
    input_ids = tokenizer.encode(tmp_prompt, return_tensors="pt")

    with torch.no_grad():
        if torch.cuda.is_available():
            bot_input_ids = input_ids.to("cuda")
        chat_history_ids = model.generate(
            bot_input_ids, do_sample=True, max_length=150, temperature=0.7, top_k=20, top_p=0.9, pad_token_id=tokenizer.eos_token_id
        )
        if torch.cuda.is_available():
            chat_history_ids = chat_history_ids.cpu()
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True)


@app.route("/respond", methods=["POST"])
def respond():
    st_time = time.time()
    contexts = request.json.get("utterances_histories", [])

    try:
        responses = []
        confidences = []
        for context in contexts:
            response = generate_response(context, model, tokenizer)
            if len(response) > 3:
                # drop too short responses
                responses += [response]
                confidences += [DEFAULT_CONFIDENCE]
            else:
                responses += [""]
                confidences += [ZERO_CONFIDENCE]
    except Exception as exc:
        logger.exception(exc)
        sentry_sdk.capture_exception(exc)
        responses = [""] * len(contexts)
        confidences = [ZERO_CONFIDENCE] * len(contexts)

    total_time = time.time() - st_time
    logger.info(f"masked_lm exec time: {total_time:.3f}s")
    return jsonify(list(zip(responses, confidences)))