#!/usr/bin/env bash
# Test yaj server with a sample multi-question request.
# Usage: ./scripts/test_decisions.sh
#
# Requires: yaj server running on localhost:8000
#   yaj serve --strategy logprob

set -euo pipefail

BASE_URL="${YAJ_BASE_URL:-http://localhost:8000}"

time curl "${BASE_URL}/api/alpha/decisions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "typesafe/jev-1.13",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this message convey urgency?",
        "criteria": {
          "true": "Explicitly time-sensitive",
          "false": "No urgency expressed"
        }
      },
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }' | python3 -m json.tool
