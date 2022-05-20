import logging
import time
import os

import sentry_sdk
import torch
from flask import Flask, request, jsonify
from sentry_sdk.integrations.flask import FlaskIntegration
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import RAKE
import re
from nltk.tokenize import sent_tokenize

stop_words = stopwords.words('english')
rake = RAKE.Rake(stop_words)

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# PRETRAINED_MODEL_NAME_OR_PATH = os.environ.get("PRETRAINED_MODEL_NAME_OR_PATH")
# logging.info(f"PRETRAINED_MODEL_NAME_OR_PATH = {PRETRAINED_MODEL_NAME_OR_PATH}")
DEFAULT_CONFIDENCE = 0.9
ZERO_CONFIDENCE = 0.0
# MAX_HISTORY_DEPTH = 3
device = 'cpu'

try:
    tokenizer = GPT2Tokenizer.from_pretrained('finetuned2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('finetuned2')
    if torch.cuda.is_available():
        model.to("cuda")
        device = "cuda"
        logger.info("prompt_storygpt is set to run on cuda")

    # model.load_state_dict(torch.load('filtered_ROCStories_gpt_medium.pt', map_location=torch.device('cpu')))

    logger.info("gpt is ready")
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel("WARNING")


def generate_part(texts, max_len, temp, num_sents, first):
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model.generate(**encoding, max_length=max_len, length_penalty=-100.0, temperature=temp,
                                       do_sample=True)
    generated_texts = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)

    texts = []
    for text in generated_texts:
        text = re.sub('\(.*?\)', '', text)  # delete everything in ()
        # text = re.sub("[#%\\(\)\*\+<=>\\\^|~]+", '.', text)
        text = text.replace(' .', '.').replace('..', '.').replace('..', '.')
        sents = sent_tokenize(text)
        text = text[:len(' '.join(sents[:num_sents]))]
        if text[-1] not in '.!?;':
            text += '.'
        if first:
            text += " In the end,"
        texts.append(text)
    return texts


def generate_response(context):
    # last_utt = context[-1]
    full_context = context
    if 'story' in full_context[-1]:
        full_context = full_context[:-1]
    logger.info(f"Full contexts in StoryGPT service: {full_context}")
    texts = ["Let me share a happy story about travel. I went to Mexico"]
    first_texts = generate_part(texts, 30, 1, 4, first=True) # 100
    logger.info(f"First part ready: {first_texts[0]}")
    final_texts = generate_part(first_texts * 2, 50, 0.8, 5, first=False) # 150

    logger.info(f"Generated: {final_texts[0]}")
    reply = final_texts[0]
    return reply


@app.route("/respond", methods=["POST"])
def respond():
    st_time = time.time()
    contexts = request.json.get("utterances_histories", [])

    try:
        responses = []
        confidences = []
        for context in contexts:
            response = generate_response(context)
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