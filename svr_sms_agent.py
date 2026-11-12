# svr_sms_agent.py
# SMS Agent — sends SMS text messages via third-party API.
# Uses google-adk to_a2a() for A2A protocol handling.

import os
import json
import uuid
import requests
from env_config import load_charm_env
from contextlib import asynccontextmanager

from google.adk.agents import LlmAgent
from model_factory import build_llm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

import warnings
warnings.filterwarnings("ignore")

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_charm_env(env_path)

# Logging
from shared_logger import SharedLogger
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")

# Agent logger (conversation tracing)
from agent_logger import agent_logger, make_callbacks

# CHARM 3.0 guard: this process acts as SMSAgent. Only callers named in charm_policy.json may
# reach it, every inbound message is screened and wrapped as untrusted, and tool calls are gated.
import charm_guard as guard
guard.init_identity("SMSAgent")

# Configuration
APP_NAME = "sms_agent"
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")
ACTIVE_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-opus-5-5")

# SMS service config
sms_url = os.getenv("SMS_URL")
sms_key = os.getenv("SMS_KEY")

# Agent card — dict for registry registration, AgentCard object for to_a2a()
AGENT_CARD_DICT = {
    "name": "SMSAgent",
    "description": "A SMS agent that takes a phone number and message content and uses its send_sms skill to send SMS text messages to the phone number.",
    "url": "https://localhost:7003",
    "version": "0.1",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "send_sms",
            "name": "Send SMS",
            "description": "Receives a message and phone number and Sends an SMS text to provided phone number with the provided content.",
            "tags": ["sms", "text", "message"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json"],
        }
    ],
}

AGENT_CARD = AgentCard(
    name="SMSAgent",
    description=AGENT_CARD_DICT["description"],
    url="https://localhost:7003",
    version="0.1",
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["application/json", "text/plain"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[AgentSkill(
        id="send_sms",
        name="Send SMS",
        description="Receives a message and phone number and Sends an SMS text to provided phone number with the provided content.",
        tags=["sms", "text", "message"],
    )],
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def send_sms(msg: str, phone: str) -> dict:
    """
    The send_sms tool will take the input from the Agent and call a third party
    SMS service to send an SMS text message. The service URL and API key are loaded from the
    environment as sms_url and sms_key.

    Parameters:
        msg: the message to send via SMS service.
        phone: the phone number to send the SMS to. The phone number should be 10 digits only.

    Returns:
        A dictionary representing the JSON returned by the SMS service.
    """
    agent_logger.standard("SMSAgent", "tool_call", f"Tool called: send_sms(phone={phone})")
    agent_logger.conversation("SMSAgent", "tool_call",
                              f"📱 Calling SMS API: phone={phone}, message=\"{msg}\"")

    response = requests.post(sms_url, data={
        'phone': phone,
        'message': msg,
        'key': sms_key,
        'sender': 'Fairwind Technologies',
    })

    if response.status_code == 200:
        try:
            result = response.json()
            success = result.get("success", False)
            quota = result.get("quotaRemaining", "unknown")
            # "success" only means the provider accepted it. The ID is how to check delivery later.
            agent_logger.standard("SMSAgent", "tool_response",
                                  f"SMS API responded: {'success' if success else 'failed'} "
                                  f"(textId: {result.get('textId')}, quota remaining: {quota}, "
                                  f"error: {result.get('error')})")
            agent_logger.conversation("SMSAgent", "tool_response",
                                      f"{'📤' if success else '❌'} SMS provider: {'accepted, delivery not confirmed' if success else 'failed'} — {quota} credits remaining")
            # Include quota info in the result for the LLM to report
            result["quotaRemaining"] = quota
            return result
        except ValueError:
            logger.error(f"Failed to parse JSON response: {response.text}")
            return {"error": True, "message": "Failed to parse SMS response"}
    else:
        agent_logger.conversation("SMSAgent", "tool_error",
                                  f"❌ SMS API failed: {response.status_code}")
        return {"error": True, "message": f"Send SMS failed with status {response.status_code}: {response.text}"}


# ---------------------------------------------------------------------------
# Registration with Registry Agent
# ---------------------------------------------------------------------------

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        logger.error(f"Could not find config file {CONFIG_FILE}! Exiting...")
        exit(1)

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

async def register(appname: str):
    logger.debug(f"Registering App: {appname}")
    config = load_config()
    agent_id = config.get("agent_id", "")
    reg_url = config.get("reg_url")
    if not reg_url:
        logger.error("Could not load the registration URL from config file.")
        exit(1)

    payload = {
        "id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "tool_name": "register_agent",
        "message": {"parts": [{"text": json.dumps(AGENT_CARD_DICT)}]},
    }

    response = guard.signed_post_json_sync(reg_url, "RegistryAgent", payload)

    if response.status_code == 200:
        try:
            response_data = response.json()
            agent_id = response_data.get("id")
            config["agent_id"] = agent_id
            save_config(config)
            agent_logger.standard("SMSAgent", "registration", f"Registered with registry (id: {agent_id})")
            agent_logger.conversation("SMSAgent", "registration",
                                      f"🚀 Starting up, registered with registry (id: {agent_id})")
        except ValueError:
            logger.error(f"Failed to parse registration response: {response.text}")
    else:
        logger.error(f"Registration failed with status {response.status_code}: {response.text}")


# ---------------------------------------------------------------------------
# Agent creation and A2A app
# ---------------------------------------------------------------------------

AGENT_PROMPT = f"""
You are an agent powered by {ACTIVE_MODEL}. Your primary responsibility is to send SMS messages by using your send_sms tool.
You operate within a multi-agent environment where other trusted agents are available and they will be connecting to you to
send SMS messages.

The user request will be a string of text that should specify the phone number to send the SMS message to, and the body of the
SMS message to send. However, this format is unstructured, so you will need to use your reasoning skills and NLP to read the
user request, understand the context, and extract the phone number and SMS message content from the user request.

The phone number will likely be in E.164 format, with a + prefix, country code, and subscriber number. However, you could also
receive it as a 10 digit number, in which case you can pass that in directly.

IT IS IMPORTANT TO NOTE: If the number starts with a plus sign (+), it must be transformed to its url encoded value of %2B
and replaced with that value in the number. The SMS system is capable of handling other text characters and spaces. Only the
+ sign must be url_encoded before being sent to the send_sms tool.

If the size of the message text is more than 160 characters, the SMS service will break it into multiple messages.

An example user request could be 'Text Hey Mike how are you to 555-555-5555'. In this case, the msg would be 'Hey Mike how are you'
and the sms_phone would be '555-555-5555'

Another example user request could be 'Send an SMS to +1 (555) 555-5555 saying Time for dinner'. In this case, the msg would be
'Time for dinner' and the sms_phone would be '%2B1 (555) 555-5555'

If the user provides incomplete information (e.g., missing message content), do not attempt to send the message.
Instead, ask the user to provide the missing information first.

When the SMS provider accepts the message, say it was accepted (delivery is not confirmed) and always include the quotaRemaining value from the send_sms response in your reply,
for example: "SMS accepted by the provider; delivery not confirmed. (697 SMS credits remaining)"

MOST IMPORTANTLY:
- THINK DEEPLY - SPEED TO REPLY IS NOT AS IMPORTANT AS CORRECT RESPONSE.
"""

# Build callbacks from agent_logger
callbacks = guard.make_guard_callbacks(
    "SMSAgent", inbound="peer",
    base_callbacks=make_callbacks("SMSAgent", ACTIVE_MODEL),
    # The model may ask to text anyone; this gate sends only to numbers an approval covers.
    tool_rules={"send_sms": guard.rule_capability("send_sms", recipient_arg="phone", kind="phone")},
)

# Create the agent
sms_agent = LlmAgent(
    name="sms_agent",
    model=build_llm(ACTIVE_MODEL),
    description="An agent that sends SMS text messages using a third-party SMS service.",
    instruction=AGENT_PROMPT + guard.GUARD_PREAMBLE,
    tools=[send_sms],
    **callbacks,
)

# Lifespan: register with registry on startup
@asynccontextmanager
async def lifespan(app):
    await register(APP_NAME)
    yield
    logger.info("Shutting down SMS agent.")

# Create the A2A Starlette app
app = to_a2a(
    sms_agent,
    host="127.0.0.1",
    port=7003,
    agent_card=AGENT_CARD,
    lifespan=lifespan,
)
app.add_middleware(guard.GuardMiddleware, agent_name="SMSAgent")

if __name__ == "__main__":
    print("Run with: uvicorn svr_sms_agent:app --host 127.0.0.1 --port 7003")
