import logging
import common.dff.integration.context as int_ctx
import scenario.response_funcs as response_funcs

from df_engine.core import Actor, Context


logger = logging.getLogger(__name__)


def intent_catcher_response(ctx: Context, actor: Actor, *args, **kwargs) -> str:
    annotated_utterance = int_ctx.get_last_human_utterance(ctx, actor)
    intention, confidence = get_detected_intents(annotated_utterance)

    response = ""
    if intention is not None and confidence > 0:
        logger.debug(f"Intent is defined as {intention}")
        dialog = int_ctx.get_dialog(ctx, actor)
        dialog["seen"] = dialog["called_intents"][intention]
        funcs = response_funcs.get_respond_funcs()[intention]
        response = funcs(ctx, actor, intention)
        # Special formatter which used in AWS Lambda to identify what was the intent
        while "#+#" in response:
            response = response[: response.rfind(" #+#")]
        logger.info(f"Response: {response}; intent_name: {intention}")
        try:
            response += " #+#{}".format(intention)
        except TypeError:
            logger.error(f"TypeError intent_name: {intention} response: {response};")
            response = "Hmmm... #+#{}".format(intention)
        # todo: we need to know what intent was called
        # current workaround is to use only one intent if several were detected
        # and to append special token with intent_name
    else:
        logger.debug("Intent is not defined")

    if response == "":
        logger.error(f"response is empty for intents: {get_intents(annotated_utterance).items()}")

    return response


def default_response(ctx: Context, actor: Actor, *args, **kwargs) -> str:
    annotated_utterance = int_ctx.get_last_human_utterance(ctx, actor)
    logger.error(f"response is empty for intents: {get_intents(annotated_utterance).items()}")
    return ""


def set_confidence_from_input(ctx: Context, actor: Actor, *args, **kwargs) -> Context:
    _, confidence = get_detected_intents(int_ctx.get_last_human_utterance(ctx, actor))
    int_ctx.set_confidence(ctx, actor, confidence)
    return ctx


def get_intents(annotated_utterance):
    annotations = annotated_utterance.get("annotations", {})
    return annotations.get("intent_catcher", {})


def get_detected_intents(annotated_utterance):
    intents = get_intents(annotated_utterance)
    intent, confidence = None, 0.0
    for key, value in intents.items():
        if value.get("detected", 0) == 1:
            confidence_current = value.get("confidence", 0.0)
            if confidence_current > confidence:
                intent, confidence = key, confidence_current

    return intent, confidence
