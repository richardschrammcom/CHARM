# svr_chat_agent.py
# Chat Agent — orchestrator that queries the registry to discover and delegate to specialist agents.
# Uses google-adk to_a2a() for A2A protocol handling.

import os
import json
import uuid
import requests   # used only for startup registration (blocking is fine there)
import httpx      # async HTTP for in-request delegation (keeps the event loop free)
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

# CHARM 3.0 guard: this process acts as ChatAgent. It signs every outbound call, and its
# approval broker holds side effects until the human approves them in the dashboard.
import charm_guard as guard
from google.adk.tools.tool_context import ToolContext
guard.init_identity("ChatAgent")
broker = guard.ApprovalBroker("ChatAgent")

# Configuration
APP_NAME = "chat_agent"
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")
ACTIVE_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-opus-5-5")
# Timeout for one A2A call to a specialist agent. Must exceed the Web Agent's worst case
# (it can issue several browse_web calls, each up to WEB_AGENT_TIMEOUT), but stay below
# the dashboard's DASHBOARD_CHAT_TIMEOUT.
REMOTE_AGENT_TIMEOUT = int(os.getenv("CHAT_REMOTE_TIMEOUT", "720"))
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")
ASSISTANT_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "assistant_profile.json")

def load_assistant_profile() -> dict:
    """Load assistant profile (name and personality)."""
    try:
        with open(ASSISTANT_PROFILE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"name": "Sam", "personality": "You are a helpful assistant with a friendly disposition."}

def load_user_profile() -> str:
    """Load user profile and format as prompt context."""
    try:
        with open(USER_PROFILE_PATH, "r") as f:
            profile = json.load(f)
        lines = []
        if profile.get("name"): lines.append(f"Name: {profile['name']}")
        if profile.get("email"): lines.append(f"Email: {profile['email']}")
        if profile.get("phone"): lines.append(f"Phone: {profile['phone']}")
        if profile.get("role"): lines.append(f"Role: {profile['role']}")
        if profile.get("notes"): lines.append(f"Notes: {profile['notes']}")
        if lines:
            return "\n\nUSER PROFILE (the person you are chatting with):\n" + "\n".join(lines) + "\nUse this information when relevant (e.g., if they say 'email me' use their email, if they say 'text me' use their phone).\n"
    except Exception:
        pass
    return ""

# Agent card — dict for registry, AgentCard object for to_a2a()
AGENT_CARD_DICT = {
    "name": "ChatAgent",
    "description": "A chat agent that takes input request from the user and dynamically enlists other agents to assist.",
    "url": "https://localhost:7002",
    "version": "0.1",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "query_registry",
            "name": "Query Register Agent",
            "description": "Queries the registry to find agents it needs.",
            "tags": ["register", "registry", "agent"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json"],
        },
        {
            "id": "call_remote_agent",
            "name": "Call Agent",
            "description": "Calls another agent.",
            "tags": ["agent"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json"],
        },
    ],
}

AGENT_CARD = AgentCard(
    name="ChatAgent",
    description=AGENT_CARD_DICT["description"],
    url="https://localhost:7002",
    version="0.1",
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["application/json", "text/plain"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[
        AgentSkill(id="query_registry", name="Query Register Agent",
                   description="Queries the registry to find agents it needs.",
                   tags=["register", "registry", "agent"]),
        AgentSkill(id="call_remote_agent", name="Call Agent",
                   description="Calls another agent.",
                   tags=["agent"]),
    ],
)


# ---------------------------------------------------------------------------
# Config helpers
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def query_registry(user_req: str) -> dict:
    """
    The query_registry tool searches the trusted agents registry on the local network by
    calling the registry agent and passing in a string of text describing the functionality
    being sought from an agent on the network.

    Parameters:
        user_req: a string describing the type of functionality being sought.

    Returns:
        A dictionary listing the agents the registry believes can solve the request, ordered
        by the most likely to least likely. Each element contains a confidence score and the
        agent card of the agent.
    """
    agent_logger.standard("ChatAgent", "tool_call", f"Tool called: query_registry(\"{user_req}\")")
    agent_logger.conversation("ChatAgent", "tool_call",
                              f"🔍 Asking registry: \"{user_req}\"")

    config = load_config()
    agent_id = config.get("agent_id", "")
    reg_url = config.get("reg_url")
    if not reg_url:
        logger.error("Could not load the registration URL from config file.")
        return {"error": True, "message": "Registry URL not configured"}

    payload = {
        "id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "tool_name": "search_agents",
        "message": {"parts": [{"text": user_req}]},
    }

    try:
        async with httpx.AsyncClient(timeout=30, verify=guard.tls_context()) as client:
            response = await guard.signed_post_json(client, reg_url, "RegistryAgent", payload)
    except httpx.HTTPError as e:
        agent_logger.conversation("ChatAgent", "tool_error", f"❌ Registry unreachable: {e}")
        return {"error": True, "message": f"Registry unreachable: {e}"}

    if response.status_code == 200:
        try:
            result = response.json()
            # Parse registry response to log agent matches
            messages = result.get("messages", [])
            if messages:
                parts = messages[-1].get("parts", [])
                if parts:
                    response_data = parts[0].get("text", "")
                    # Try to extract agent names and scores for conversation logging
                    if isinstance(response_data, list):
                        agent_summaries = []
                        for entry in response_data:
                            name = entry.get("name", "?")
                            score = entry.get("confidence_score", "?")
                            agent_summaries.append(f"{name}={score}")
                        summary = ", ".join(agent_summaries)
                        agent_logger.standard("ChatAgent", "tool_response",
                                              f"Registry returned {len(response_data)} agents")
                        agent_logger.conversation("ChatAgent", "tool_response",
                                                  f"📋 Registry returned {len(response_data)} agents. Scores: {summary}")
                    else:
                        agent_logger.standard("ChatAgent", "tool_response", "Registry responded")
                        agent_logger.conversation("ChatAgent", "tool_response", "📋 Registry responded")
            return result
        except ValueError:
            logger.error(f"Failed to parse JSON response: {response.text}")
            return {"error": True, "message": "Failed to parse registry response"}
    else:
        agent_logger.conversation("ChatAgent", "tool_error",
                                  f"❌ Registry query failed: {response.status_code}")
        return {"error": True, "message": f"Registry query failed: {response.status_code}: {response.text}"}


async def _resolve_agent_card(agent_id: str) -> dict:
    """Look an agent up in the registry by ID. The model never supplies an address."""
    config = load_config()
    payload = {"id": str(uuid.uuid4()), "agent_id": agent_id, "tool_name": "get_agent_card",
               "message": {"parts": [{"text": ""}]}}
    async with httpx.AsyncClient(timeout=30, verify=guard.tls_context()) as client:
        resp = await guard.signed_post_json(client, config.get("reg_url"), "RegistryAgent", payload)
    if resp.status_code != 200:
        raise guard.GuardError(f"no registered agent with id '{agent_id}'")
    return resp.json()


async def call_remote_agent(agent_id: str, query: str, tool_context: ToolContext) -> dict:
    """
    The call_remote_agent tool delegates a task to another agent on the network using the Google
    A2A protocol. Identify the agent by the "agent_id" that query_registry returned for it; the
    harness looks up its address in the registry and signs the request.

    Parameters:
        agent_id: the "agent_id" of an agent returned by query_registry.
        query: the specific request for that agent. For EmailAgent, send JSON with "to", "subject"
               and "body". For SMSAgent, include the phone number and the message text.

    Returns:
        A dictionary with 'error' (boolean), 'agent' (the agent's name), and either 'message'
        (the agent's reply) or 'notice' (an explanation from your harness).
    """
    # 2.1 took the whole agent card, including its URL, as a model-supplied argument. In 3.0 the
    # model names an agent ID; code resolves the address from the registry and checks it.
    try:
        card = await _resolve_agent_card(agent_id)
    except (guard.GuardError, httpx.HTTPError) as e:
        agent_logger.conversation("ChatAgent", "tool_error", f"❌ Could not resolve agent '{agent_id}': {e}")
        return {"error": True, "agent": "unknown",
                "notice": f"No registered agent has id '{agent_id}'. Use an agent_id returned by query_registry."}

    agent_name = card.get("name", "Unknown")
    agent_url = card.get("url", "")
    if not guard.allowed_agent_address(agent_name, agent_url):
        guard.emit_guard_event("ChatAgent", "guard_block",
                               f"🛡️ Refused to call {agent_name}: {agent_url} is not an allowed agent host")
        return {"error": True, "blocked_by": "charm_guard", "agent": agent_name,
                "notice": f"Blocked by the CHARM guard: {agent_name} is registered at a host that is not allowed."}

    agent_logger.standard("ChatAgent", "tool_call",
                          f"Tool called: call_remote_agent → {agent_name}")
    agent_logger.conversation("ChatAgent", "tool_call",
                              f"📞 Delegating to {agent_name} at {agent_url}: \"{query}\"")

    # Side effects (email, SMS) need a capability token naming the exact recipients.
    capability = None
    side = guard.side_effect_for(agent_name)
    if side:
        recipients = guard.extract_recipients(side["kind"], query)
        if not recipients:
            return {"error": True, "agent": agent_name,
                    "notice": f"Could not tell who {agent_name} would contact. Put the recipient in the "
                              f"request (for email, JSON with a \"to\" field)."}
        tainted = guard.session_tainted(tool_context)
        if guard.approval_required(side["kind"], recipients, tainted):
            reasons = []
            if tainted:
                reasons.append("this conversation has read external or suspicious content")
            off_list = [r for r in recipients if not guard.on_allowlist(side["kind"], r)]
            if off_list:
                reasons.append("not on your allowlist: " + ", ".join(off_list))
            if not reasons:
                reasons.append("approval mode is 'always'")
            decision = await broker.request(agent_name, side["action"], side["kind"], recipients,
                                            preview=query, reason="; ".join(reasons))
            if not decision.approved:
                return {"error": True, "agent": agent_name,
                        "notice": f"Not approved ({decision.reason}). Nothing was sent. Tell the user."}
            capability = decision.capability
        else:
            capability = guard.mint_capability(agent_name, side["action"], recipients, side["kind"])
            guard.emit_guard_event("ChatAgent", "guard_allow",
                                   f"🛡️ Auto-approved {side['action']} to {', '.join(recipients)} "
                                   f"(allowlisted, clean session)")

    # Use A2A JSON-RPC protocol (message/send)
    jsonrpc_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }

    try:
        async with httpx.AsyncClient(timeout=REMOTE_AGENT_TIMEOUT, verify=guard.tls_context()) as client:
            result = await guard.signed_post_json(client, agent_url, agent_name, jsonrpc_payload,
                                                  capability=capability)
    except (httpx.ConnectError, httpx.TimeoutException):
        # 2.1 silently re-queried the registry and called any substitute scoring 70+. In 3.0,
        # switching to a different agent is a new decision the user should see.
        agent_logger.standard("ChatAgent", "tool_error", f"{agent_name} is unreachable at {agent_url}")
        agent_logger.conversation("ChatAgent", "tool_error",
                                  f"❌ {agent_name} is unreachable at {agent_url} — agent may be offline")
        return {"error": True, "agent": agent_name,
                "notice": f"The agent '{agent_name}' is not responding (it may be offline). Tell the user. "
                          f"Do not switch to a substitute agent without asking them."}

    if result.status_code == 401:
        guard.emit_guard_event("ChatAgent", "guard_block", f"🛡️ {agent_name} rejected our signed request")
        return {"error": True, "agent": agent_name,
                "notice": f"{agent_name} refused the request: {result.text[:200]}"}
    if result.status_code != 200:
        agent_logger.conversation("ChatAgent", "tool_error",
                                  f"❌ {agent_name} returned error: {result.status_code}")
        return {"error": True, "agent": agent_name,
                "notice": f"Agent call failed with status {result.status_code}."}

    rpc_response = result.json()
    external = guard.is_external_content_agent(agent_name)

    if "error" in rpc_response:
        error_msg = rpc_response["error"].get("message", "Unknown error")
        agent_logger.conversation("ChatAgent", "tool_error",
                                  f"❌ {agent_name} RPC error: {error_msg}")
        return {"error": True, "agent": agent_name, "external": external, "message": error_msg}

    # Extract response text from A2A result
    task_result = rpc_response.get("result", {})
    artifacts = task_result.get("artifacts", [])
    agent_reply_text = ""
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                agent_reply_text += part.get("text", "")

    if agent_reply_text:
        agent_logger.standard("ChatAgent", "tool_response",
                              f"{agent_name} responded successfully")
        agent_logger.conversation("ChatAgent", "tool_response",
                                  f"📨 {agent_name} responded: \"{agent_reply_text[:150]}{'...' if len(agent_reply_text) > 150 else ''}\"")
        # The guard's after-tool callback wraps this reply as untrusted, and marks the session
        # tainted if the agent brings in external content (e.g. the WebAgent).
        return {"error": False, "agent": agent_name, "external": external, "message": agent_reply_text}
    else:
        agent_logger.conversation("ChatAgent", "tool_error",
                                  f"❌ {agent_name}: no text in response artifacts")
        return {"error": True, "agent": agent_name, "notice": "No response text from agent"}


# ---------------------------------------------------------------------------
# Registration with Registry Agent
# ---------------------------------------------------------------------------

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
            agent_logger.standard("ChatAgent", "registration", f"Registered with registry (id: {agent_id})")
            agent_logger.conversation("ChatAgent", "registration",
                                      f"🚀 Starting up, registered with registry (id: {agent_id})")
        except ValueError:
            logger.error(f"Failed to parse registration response: {response.text}")
    else:
        logger.error(f"Registration failed with status {response.status_code}: {response.text}")


# ---------------------------------------------------------------------------
# Agent creation and A2A app
# ---------------------------------------------------------------------------

_assistant = load_assistant_profile()
_assistant_name = _assistant.get("name", "Sam")
_assistant_personality = _assistant.get("personality", "You are a helpful assistant.")

AGENT_PROMPT = f"""
Your name is {_assistant_name}. {_assistant_personality}

You are powered by {ACTIVE_MODEL}. Be conversational and natural in your responses.
For action requests (sending emails, texts, etc.), start your response with a brief natural acknowledgment like "Sure thing!",
"Absolutely!", "Done!", etc. followed by what you did. For greetings and questions, just respond naturally.

You operate within a multi-agent environment where other trusted agents are available to assist with specific capabilities that you may lack.

Each agent in the network registers itself with the Registry Agent using an AGENT_CARD that describes the agent and its skills.
If a user request requires functionality you do not have (e.g., sending a text message), you must:
1. Use your `query_registry` tool to contact the Registry Agent and ask if any agents can fulfill the needed capability.
2. Review the returned list of agents (ranked by confidence) and determine if any are capable of performing the task.
3. If one is suitable, use your `call_remote_agent` tool with that agent's `agent_id` (from the registry results) and the subtask.
4. Wait for the response, then incorporate the results into your final response to the user.
5. Inform the user that assistance was provided by another agent, including the name(s) from the AGENT_CARD(s).

Example: If the user asks you to generate a poem and text it to 555-555-5555:
- You generate the poem yourself.
- Then query the registry to find an agent that can send text messages.
- If one is found, you send the request to that agent with the poem.
- After receiving the response, report the outcome to the user.

You may need to split a user request into multiple subtasks for multiple agents.
Always attempt to fulfill as much of the request as you can yourself, and transparently communicate when and how
other agents were used.

Be collaborative, capable, and clear in your responses.

REGISTRY SEARCH RULES:
- Search the registry ONCE per capability you need. Do NOT rephrase and search again if the first search returns no good matches.
- Only use `call_remote_agent` with an `agent_id` that was ACTUALLY RETURNED by `query_registry` with a high confidence score (70+).
- NEVER invent or guess an agent_id. If query_registry did not return it, it does not exist.
- If the registry returns no agents with a confidence score above 70 for the needed capability, do NOT call any agent.
  Instead, tell the user clearly and specifically, for example:
  "I'm not able to do that myself ({ACTIVE_MODEL} does not have that capability), and there is no agent currently
  registered on the network that can help with this either. You may want to check back later if new agents are added."

CONVERSATION CONTEXT:
You have access to the full conversation history with the user. When the user refers to a previous request
(e.g., "try again", "send it again", "retry that"), look back through the conversation to find the relevant
details (email addresses, phone numbers, message content, etc.) and use them. Do NOT ask the user to repeat
information they have already provided in this conversation.

SPECIAL INSTRUCTIONS FOR EMAIL REQUESTS:
When the user asks you to send an email, extract the email information from their request or from conversation history:
- recipient email address (to)
- subject line
- body content

Only ask for missing information if it was never provided in any message in this conversation.

When calling the EmailAgent, structure your query as JSON like this:
{{"to":"recipient@email.com","subject":"Subject Line","body":"Complete message body content"}}

CRITICAL: Never send an email request with an empty body field.

SPECIAL INSTRUCTIONS FOR SMS MESSAGES:
Keep every SMS message text to 160 characters or fewer, so it goes out as a single text and arrives quickly.
When the user asks you to text a summary, write a one- or two-sentence version that fits; if they also asked
for an email, the full summary goes in the email.

SPECIAL INSTRUCTIONS FOR SMS RESPONSES:
When the SMSAgent responds with quota information (e.g., quotaRemaining), include it in your response to the user.
For example: "You have 697 remaining messages in your SMS quota."

SPECIAL INSTRUCTIONS FOR CITATIONS / SOURCES:
When a web research agent (e.g. WebAgent) answers, its response ends with a "Sources:" section
listing the actual URLs it visited. Treat those as the only valid citations.
- If the user asks for a report/answer "with citations", "with sources", "with references", or
  similar, include a "Sources" section in your report (or email body) listing those exact URLs.
- If the user did NOT ask for citations, you may omit the Sources section to keep the answer clean,
  but retain the URLs from the agent's response in this conversation so you can add citations if the
  user asks for them afterward (e.g. "resend that with sources").
- NEVER fabricate, guess, or add URLs that were not in the agent's "Sources:" list. If no sources
  were provided, say so rather than inventing any.

MOST IMPORTANTLY:
- THINK DEEPLY - SPEED TO REPLY IS NOT AS IMPORTANT AS CORRECT RESPONSE.
- ALWAYS SEARCH THE REGISTRY ONCE USING 'query_registry' BEFORE TELLING THE USER YOU CANNOT HELP.
- WHEN EMAIL OR SMS SUCCEEDS, say the provider or mail server accepted it. It is not confirmed delivered, so never say "delivered" or "on its way".
- IF THE REGISTRY RETURNS NO SUITABLE AGENT, TELL THE USER WHAT YOU CANNOT DO AND WHY (no capability and no registered agent).

APPROVALS:
Sending email or SMS may pause while the user approves it in the dashboard. That is expected. If an
action comes back as not approved, tell the user plainly that nothing was sent. Never try to get around
an approval or a guard block by rephrasing, splitting the request, or choosing another agent.
""" + load_user_profile() + guard.GUARD_PREAMBLE

# Build callbacks from agent_logger
callbacks = guard.make_guard_callbacks(
    "ChatAgent", inbound="user",
    base_callbacks=make_callbacks("ChatAgent", ACTIVE_MODEL),
    untrusted_tools=["call_remote_agent"],
)

# Create the agent
chat_agent = LlmAgent(
    name="chat_agent",
    model=build_llm(ACTIVE_MODEL),
    description="An agent that satisfies user chat requests, and dynamically enlists other agents on the network to assist as needed.",
    instruction=AGENT_PROMPT,
    tools=[query_registry, call_remote_agent],
    **callbacks,
)

# Lifespan: register with registry on startup
@asynccontextmanager
async def lifespan(app):
    await register(APP_NAME)
    yield
    logger.info("Shutting down chat agent.")

# Create the A2A Starlette app
app = to_a2a(
    chat_agent,
    host="127.0.0.1",
    port=7002,
    agent_card=AGENT_CARD,
    lifespan=lifespan,
)
# CHARM 3.0: the dashboard posts approvals here; every request must be signed.
app.router.routes.append(broker.starlette_route())
app.add_middleware(guard.GuardMiddleware, agent_name="ChatAgent")

if __name__ == "__main__":
    print("Run with: uvicorn svr_chat_agent:app --host 127.0.0.1 --port 7002")
