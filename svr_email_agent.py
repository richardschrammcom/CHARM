# svr_email_agent.py
# Email Agent — sends emails via SMTP (MXRoute / any SMTP provider).
# Uses google-adk to_a2a() for A2A protocol handling.

import os
import json
import uuid
import smtplib
import ssl
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from env_config import load_charm_env
from contextlib import asynccontextmanager

from google.adk.agents import LlmAgent
from model_factory import build_llm
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

# CHARM 3.0 guard: this process acts as EmailAgent. Only callers named in charm_policy.json may
# reach it, every inbound message is screened and wrapped as untrusted, and tool calls are gated.
import charm_guard as guard
guard.init_identity("EmailAgent")

# Configuration
APP_NAME = "email_agent"
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")
ACTIVE_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-opus-5-5")

# SMTP configuration
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)  # Defaults to SMTP_USER if not set


# Agent card — dict for registry, AgentCard object for to_a2a()
AGENT_CARD_DICT = {
    "name": "EmailAgent",
    "description": "An email agent that sends emails via SMTP. Requires a recipient address, subject line, and message body.",
    "url": "https://localhost:7001",
    "version": "0.2.0",
    "provider": {
        "organization": "Fairwind Technologies, LLC.",
        "url": "https://www.fairwindtechnologies.com",
    },
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["application/json", "text/plain"],
    "defaultOutputModes": ["application/json", "text/plain"],
    "skills": [
        {
            "id": "send_email",
            "name": "Send Email",
            "description": "Send an email to one or more recipients. Requires the To address, a subject line, and the message body.",
            "tags": ["send", "email"],
            "examples": [
                "Send an email to test@test.com with a subject line of 'Touch Base' and a body of 'Hi! Just reaching out to say Hi!'"
            ],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json", "text/plain"],
        }
    ],
}

AGENT_CARD = AgentCard(
    name="EmailAgent",
    description=AGENT_CARD_DICT["description"],
    url="https://localhost:7001",
    version="0.2.0",
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["application/json", "text/plain"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[AgentSkill(
        id="send_email",
        name="Send Email",
        description="Send an email to one or more recipients. Requires the To address, a subject line, and the message body.",
        tags=["send", "email"],
    )],
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def send_email(to: str, subject: str, body: str) -> dict:
    """
    Send an email via SMTP.

    Parameters:
        to: recipient email address (or comma-separated list for multiple recipients).
        subject: the email subject line.
        body: the email body content (plain text).

    Returns:
        A dictionary with 'success' (boolean) and 'message' (status description).
    """
    agent_logger.standard("EmailAgent", "tool_call",
                          f"Tool called: send_email(to={to}, subject=\"{subject}\")")
    agent_logger.conversation("EmailAgent", "tool_call",
                              f"📧 Sending email to {to}, subject: \"{subject}\"")

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        agent_logger.conversation("EmailAgent", "tool_error",
                                  "❌ SMTP not configured — check SMTP_HOST, SMTP_USER, SMTP_PASS in .env")
        return {"success": False, "message": "SMTP credentials not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASS in .env"}

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_FROM
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=ssl.create_default_context())  # verifies the server cert
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        agent_logger.standard("EmailAgent", "tool_response", f"Email accepted by the mail server for {to}")
        agent_logger.conversation("EmailAgent", "tool_response",
                                  f"📤 Mail server accepted the email for {to} (delivery not confirmed)")
        return {"success": True, "message": f"Accepted by the mail server for {to}. Delivery is not confirmed."}

    except smtplib.SMTPAuthenticationError as e:
        agent_logger.conversation("EmailAgent", "tool_error",
                                  f"❌ SMTP authentication failed: {e}")
        return {"success": False, "message": f"SMTP authentication failed: {e}"}
    except smtplib.SMTPException as e:
        agent_logger.conversation("EmailAgent", "tool_error",
                                  f"❌ SMTP error: {e}")
        return {"success": False, "message": f"SMTP error: {e}"}
    except Exception as e:
        agent_logger.conversation("EmailAgent", "tool_error",
                                  f"❌ Failed to send email: {e}")
        return {"success": False, "message": f"Failed to send email: {e}"}


# ---------------------------------------------------------------------------
# Config & Registration
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
            agent_logger.standard("EmailAgent", "registration", f"Registered with registry (id: {agent_id})")
            agent_logger.conversation("EmailAgent", "registration",
                                      f"🚀 Starting up, registered with registry (id: {agent_id})")
        except ValueError:
            logger.error(f"Failed to parse registration response: {response.text}")
    else:
        logger.error(f"Registration failed with status {response.status_code}: {response.text}")


# ---------------------------------------------------------------------------
# Agent creation and A2A app
# ---------------------------------------------------------------------------

AGENT_PROMPT = """
You are an email agent. Your job is to send emails using the send_email tool.

You will receive requests to send emails. Extract the following from each request:
- to: the recipient email address
- subject: the email subject line
- body: the email body content

If the request is structured as JSON with "to", "subject", and "body" fields, use those directly.
If the request is natural language, parse it to extract the recipient, subject, and body.

If any required field (to, subject, or body) is missing, respond asking for the missing information.
Do not send an email with empty fields.

After sending, report the result to the caller. A successful result means the mail server accepted the
message, not that it was delivered, so say "accepted by the mail server" and never "delivered".
"""

callbacks = guard.make_guard_callbacks(
    "EmailAgent", inbound="peer",
    base_callbacks=make_callbacks("EmailAgent", ACTIVE_MODEL),
    # The model may ask to send anything; this gate sends only to recipients an approval covers.
    tool_rules={"send_email": guard.rule_capability("send_email", recipient_arg="to", kind="email")},
)

email_agent = LlmAgent(
    name="email_agent",
    model=build_llm(ACTIVE_MODEL),
    description="Email agent that sends emails via SMTP.",
    instruction=AGENT_PROMPT + guard.GUARD_PREAMBLE,
    tools=[send_email],
    **callbacks,
)

@asynccontextmanager
async def lifespan(app):
    # Validate SMTP config at startup
    if not SMTP_HOST:
        logger.warning("SMTP_HOST not set — email sending will fail. Set SMTP_HOST, SMTP_USER, SMTP_PASS in .env")
    else:
        logger.info(f"SMTP configured: {SMTP_HOST}:{SMTP_PORT} as {SMTP_FROM}")

    await register(APP_NAME)
    yield
    logger.info("Shutting down Email Agent.")

app = to_a2a(
    email_agent,
    host="127.0.0.1",
    port=7001,
    agent_card=AGENT_CARD,
    lifespan=lifespan,
)
app.add_middleware(guard.GuardMiddleware, agent_name="EmailAgent")

if __name__ == "__main__":
    print("Run with: uvicorn svr_email_agent:app --host 127.0.0.1 --port 7001")
