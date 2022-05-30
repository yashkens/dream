import logging
import re
import os
from . import response as loc_rsp

from df_engine.core import Context, Actor

import common.dff.integration.context as int_ctx

logging.basicConfig(format="%(asctime)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel("WARNING")

STORY_TYPE = os.getenv("STORY_TYPE")
# logger.info(f"Story type in conditions: {STORY_TYPE}")


def has_story_type(ctx: Context, actor: Actor) -> bool:
    return bool(loc_rsp.get_story_type(ctx, actor))


def has_story_left(ctx: Context, actor: Actor) -> bool:
    return bool(loc_rsp.get_story_left(ctx, actor))


def is_tell_me_a_story(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    return bool(
        re.search("tell", ctx.last_request, re.IGNORECASE) and re.search("story", ctx.last_request, re.IGNORECASE)
    )


def is_asked_for_a_story(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    prev_node = loc_rsp.get_previous_node(ctx)
    return prev_node != "which_story_node"


def needs_scripted_story(ctx: Context, actor: Actor) -> bool:
    if STORY_TYPE == 'scripted':
        return True
    # logger.info(f"Story TYPE: {STORY_TYPE}")
    return False


def has_story_intent(ctx: Context, actor: Actor) -> bool:
    utt = int_ctx.get_last_human_utterance(ctx, actor)
    if utt["text"]:
        story_intent = utt['annotations']['intent_catcher']['tell_me_a_story']['detected']
        logger.info(f"Story intent value: {story_intent}")
        if story_intent == 1:
            return True
    return False

