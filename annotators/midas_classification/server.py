import logging
import time
import os

import numpy as np
import sentry_sdk
import spacy
from deeppavlov import build_model
from flask import Flask, request, jsonify

sentry_sdk.init(os.getenv("SENTRY_DSN"))

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

EMPTY_SIGN = ": EMPTY >"

try:
    spacy_nlp = spacy.load("en_core_web_sm")
    model = build_model("midas_conv_bert.json", download=True)
    m = model(["hi"])
except Exception as e:
    logger.exception("Midas not loaded")
    sentry_sdk.capture_exception(e)
    raise e


label_to_act = {
    0: "opinion",
    1: "statement",
    2: "pos_answer",
    3: "neg_answer",
    4: "command",
    5: "comment",
    6: "other_answers",
    7: "open_question_factual",
    8: "yes_no_question",
    9: "complaint",
    10: "open_question_opinion",
    11: "appreciation",
    12: "dev_command",
}

logger.info(f"Considered classes dictionary: {label_to_act}")


def predict(inputs):
    logger.info(f"Inputs {inputs}")
    if len(inputs) == 0:
        return []
    elif len(inputs) == 1 and inputs[0].strip() == EMPTY_SIGN:
        logger.warning("Calling MIDAS with empty inputs. Check out why")
        return [{}]
    try:
        predictions = model(inputs)
        responses = [
            {class_name: pred_value.astype(np.float64) for class_name, pred_value in zip(label_to_act.values(), preds)}
            for preds in predictions
        ]
    except Exception as e:
        responses = [{class_name: 0 for class_name in label_to_act.values()} for _ in inputs]
        sentry_sdk.capture_exception(e)
        logger.exception(e)
    return responses


def recombine_responses(responses, dialog_ids, n_dialogs):
    dialog_ids = np.array(dialog_ids)
    responses = np.array(responses)

    final_responses = []
    for i in range(n_dialogs):
        curr_responses = responses[dialog_ids == i]
        final_responses.append(list(curr_responses))
    return final_responses


def spacy_sent_tokenize(sentence):
    doc = spacy_nlp(sentence)
    segments = []
    for sent in doc.sents:
        segments.append(sent.text)
    return segments


@app.route("/model", methods=["POST"])
def respond():
    st_time = time.time()
    dialogs = request.json["dialogs"]
    dialog_ids = []
    inputs = []
    for i, dialog in enumerate(dialogs):
        if len(dialog["bot_utterances"]):
            tokenized_sentences = (
                dialog["bot_utterances"][-1].get("annotations", {}).get("sentseg", {}).get("segments", [])
            )
            if len(tokenized_sentences) == 0:
                prev_bot_uttr_text = dialog["bot_utterances"][-1].get("text", "").lower()
                tokenized_sentences = spacy_sent_tokenize(prev_bot_uttr_text)
            context = tokenized_sentences[-1].lower() if len(tokenized_sentences) > 0 else ""
        else:
            context = ""
        if len(dialog["human_utterances"]):
            sentences = dialog["human_utterances"][-1].get("annotations", {}).get("sentseg", {}).get("segments", [])
            if len(sentences) == 0:
                sentences = spacy_sent_tokenize(dialog["human_utterances"][-1].get("text", "").lower())
        else:
            sentences = []

        for sent in sentences:
            input_ = f"{context} {EMPTY_SIGN} {sent}"
            inputs.append(input_)
            dialog_ids.append(i)

    responses = predict(inputs)

    final_responses = recombine_responses(responses, dialog_ids, len(dialogs))

    logging.info(f"midas_classification exec time {time.time() - st_time}")
    return jsonify(final_responses)


@app.route("/batch_model", methods=["POST"])
def batch_respond():
    st_time = time.time()
    bot_utterances = request.json["sentences"]
    human_utterances = request.json["last_human_utterances"]
    bot_utterances_sentences = [
        spacy_sent_tokenize(utterance) if isinstance(utterance, str) else utterance for utterance in bot_utterances
    ]
    dialog_ids = []
    inputs = []
    for i, bot_utterance_sents in enumerate(bot_utterances_sentences):
        if human_utterances[i]:
            if isinstance(human_utterances[i], str):
                context = spacy_sent_tokenize(human_utterances[i])[-1].lower()
            elif isinstance(human_utterances[i], list):
                context = human_utterances[i][-1].lower()
            else:
                context = ""
        else:
            context = ""
        for utterance in bot_utterance_sents:
            inputs += [f"{context} {EMPTY_SIGN} {utterance.lower()}"]
            dialog_ids += [i]

    responses = predict(inputs)  # list of dicts, dict keys - classes, dict values - probas

    final_responses = recombine_responses(responses, dialog_ids, len(bot_utterances))

    logging.info(f"midas_classification exec time {time.time() - st_time}")
    return jsonify([{"batch": final_responses}])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
