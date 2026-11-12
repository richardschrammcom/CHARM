#!/bin/bash

# Test script for SMS Agent (svr_sms_agent.py)
# Make sure the agent is running on localhost:7003 before running these tests
# CHARM 3.0: without a CHARM client cert these calls fail at the TLS handshake, so curl prints
# nothing. Note /tasks/send is not the SMS Agent's route (it serves A2A at /), so don't use this
# script as proof of what 2.1 does. Add `-v` to see the refused handshake.

BASE_URL="https://localhost:7003"

echo "Testing SMS Agent..."
echo "===================="

# Test 1: Agent Card endpoint
echo "1. Testing Agent Card endpoint..."
curl -sk -X GET "${BASE_URL}/.well-known/agent.json" | jq '.'
echo -e "\n"

# Test 2: Simple SMS request
echo "2. Testing simple SMS request..."
curl -sk -X POST "${BASE_URL}/tasks/send" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-task-001",
    "message": {
      "parts": [
        {
          "text": "Send an SMS to 513.400.7390 saying Hello from the SMS agent test!"
        }
      ]
    }
  }' | jq '.'
echo -e "\n"

