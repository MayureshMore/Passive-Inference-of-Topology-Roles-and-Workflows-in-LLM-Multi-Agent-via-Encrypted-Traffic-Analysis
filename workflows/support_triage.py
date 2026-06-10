"""
Support triage workflow — ticket classified, routed, and escalated.
Characteristic traffic: initial ticket payload, rapid small responses,
and a single escalation hop.

Tickets are intentionally mixed: short tickets (~300-800B) and long tickets
with attached stack traces or log excerpts (~1500-3000B).  Long tickets overlap
with CR's small-snippet range so the classifier must use structural signals.
"""

from __future__ import annotations

import random

from .base import BaseWorkflow, WorkflowClass

_TICKETS = [
    # Billing
    "My account shows a charge of $149 but I'm on the free plan. Please investigate immediately.",
    "We were double-charged for November — two identical invoices totalling $2,398. Need urgent reversal.",
    "Our annual contract renewal was auto-renewed at the old rate instead of the negotiated $8,400. Please correct.",
    "Invoice #INV-2024-09812 shows the wrong company name. We need a corrected invoice for tax purposes.",
    "We cancelled our subscription on Oct 3rd but were still charged on Nov 1st. Please refund.",

    # Technical / API
    "The API is returning 503 errors intermittently since about 2 hours ago. Our production pipeline is blocked.",
    "Webhook deliveries stopped at 14:22 UTC. No events received since. Our order processing is halted.",
    "The REST API /v2/reports endpoint is returning 400 Bad Request for valid parameters that worked last week.",
    "SDK v3.2.1 throws NullPointerException on init in Android 14. Stack trace attached. Regression from v3.2.0.",
    "Rate limit headers show X-RateLimit-Remaining: 0 even on a fresh API key with no prior calls.",

    # Authentication
    "I can't log in — it says 'invalid credentials' even after resetting my password three times.",
    "SSO with Okta broke after our IT team updated the SAML metadata URL. Error: invalid_assertion.",
    "Our service account token expired and auto-rotation failed. All automated jobs are failing since midnight.",
    "Two-factor authentication is sending codes to an old phone number we no longer have access to. Account locked.",
    "After password reset, system shows 'account suspended'. We haven't violated any terms. Need account restored.",

    # Feature / Product
    "Feature request: please add bulk CSV import for user roles — manually adding 500+ users is not viable.",
    "The dashboard export to PDF is cutting off the rightmost columns on A4 paper. Affects all chart types.",
    "Dark mode preference resets to light mode every time we clear browser cookies. Please persist server-side.",
    "The search API doesn't support fuzzy matching — all queries require exact strings. Competitors have this.",
    "Mobile app on iOS 17.4 crashes when opening the notifications panel. Reproducible on iPhone 15 Pro.",

    # Data / Compliance
    "Our data export job has been running for 6 hours and hasn't completed. Job ID: EXP-9182. Data needed urgently.",
    "GDPR data deletion request for user ID 48821. Legal deadline is 72 hours under Article 17 GDPR.",
    "We need an audit log of all admin actions taken on our account between Jan 1 – Mar 31 for SOC 2 review.",
    "The compliance report shows our data is stored in EU-WEST-1 but our DPA specifies EU-CENTRAL-1 only.",
    "User data is appearing in another tenant's exports — potential data isolation breach. Treat as P1.",

    # Integration
    "Salesforce sync stopped after the March platform update. New contacts not appearing. Error: OAuth token expired.",
    "Zapier integration failing with 'authentication error' — regenerated API key but issue persists.",
    "Slack notifications stopped after we migrated to a new Slack workspace. Webhooks return 404.",
    "Jira sync duplicating issues — each ticket appears twice in the integration dashboard since last Thursday.",
    "Google Calendar integration shows events in UTC despite timezone being set to IST in account settings.",
]

# Long tickets with attached traces/logs (~1500-3000B): overlap with CR small-snippet range
_LONG_TICKETS = [
    """\
[P1 CRITICAL] Production payment service crash at 03:14 UTC — revenue blocked

payments-api v2.4.1 has been down since 03:14 UTC. Queue depth at crash: 1,247 orders.

Traceback (most recent call last):
  File "/opt/app/payments/server.py", line 341, in process_payment
    result = await self.gateway.charge(amount=payload["amount"], token=payload["card_token"])
  File "/opt/app/payments/gateway/stripe_adapter.py", line 89, in charge
    response = await self._client.post("/v1/charges", json=charge_data, timeout=30.0)
  File "/opt/app/vendor/httpx/_client.py", line 1374, in request
    return await self.send(request, auth=auth, follow_redirects=follow_redirects)
  File "/opt/app/vendor/httpx/_transports/asyncio.py", line 77, in handle_async_request
    with request_context(request=request):
  File "/opt/app/payments/gateway/stripe_adapter.py", line 112, in charge
    raise GatewayTimeoutError(f"Stripe API timeout after 30s: {payload['order_id']}")
payments.exceptions.GatewayTimeoutError: Stripe API timeout after 30s: ORD-20241127-98421

The above exception was the direct cause of the following exception:

  File "/opt/app/payments/worker.py", line 134, in _process
    charged = await self.processor.process_payment(task.payload)
  File "/opt/app/payments/server.py", line 345, in process_payment
    raise PaymentProcessingError(f"Gateway failure for order {payload['order_id']}") from exc
payments.exceptions.PaymentProcessingError: Gateway failure for order ORD-20241127-98421

Process: 14823 | Host: payments-prod-03 | Last 50 successful payments completed before failure.""",

    """\
[P2 HIGH] DB connection pool exhausted — 34% of API requests returning 503 since 11:29 UTC

Service: api-gateway v5.1.2 on prod-cluster-west
DB: PostgreSQL 15 (RDS db.r6g.2xlarge, max_connections=200)
First error: 2024-11-14T11:29:47Z

Error log excerpt (api-gateway-prod-07, last 15 min):

2024-11-14T11:29:47Z [ERROR] asyncpg.TooManyConnectionsError: sorry, too many clients already
2024-11-14T11:29:51Z [ERROR] Connection acquire timeout 5.0s — pool=main checked_out=70/70
2024-11-14T11:30:01Z [ERROR] asyncpg.TooManyConnectionsError: sorry, too many clients already
2024-11-14T11:30:04Z [ERROR] Health check FAILED — DB latency=timeout (threshold=100ms)
2024-11-14T11:30:08Z [ERROR] Connection acquire timeout 5.0s — pool=main checked_out=70/70
2024-11-14T11:30:12Z [WARN]  Circuit breaker OPEN after 5 consecutive DB failures
2024-11-14T11:30:15Z [ERROR] 503 Service Unavailable: GET /api/v2/reports/monthly
2024-11-14T11:30:22Z [INFO]  Pool stats: size=50 checked_out=70 overflow=20 idle=0 invalidated=3
2024-11-14T11:30:29Z [ERROR] Long-running query detected: query_id=7f3a1b duration=127s state=active
2024-11-14T11:30:40Z [ERROR] Connection acquire timeout 5.0s — pool=main checked_out=70/70
2024-11-14T11:30:47Z [WARN]  Memory usage: 87.4% (14.2GB/16.0GB) — approaching OOM threshold

pg_stat_activity: 198 active connections (limit=200), 47 in state=idle_in_transaction.
Blocking query: SELECT * FROM events WHERE... — no index, full table scan on 340M-row table.""",

    """\
[P2 HIGH] Memory leak in event-processor — OOM kills every ~4 hours since v3.7.0 deploy

Service: event-processor v3.7.0 (Node.js 20.10 on k8s prod-east)
Pattern: heap grows ~200MB/hour; OOM kill after ~4h; auto-restarts
Regression: started with v3.7.0; v3.6.8 was stable

Latest OOM: 2024-11-14T07:23:11Z
k8s event: Container event-processor-worker OOMKilled. Limit: 2Gi. Last usage: 1.98Gi.

Heap growth profile between restarts:
  T+0h: heap_used=312MB  heap_total=450MB  external=28MB
  T+1h: heap_used=498MB  heap_total=640MB  external=31MB
  T+2h: heap_used=729MB  heap_total=890MB  external=34MB
  T+3h: heap_used=1012MB heap_total=1180MB external=38MB
  T+3.8h: OOMKilled

--heapsnapshot-near-heap-limit retained-size top 5:
  1. (closure)       823,441 objects  | 498.2 MB retained
  2. EventEmitter     14,292 objects  | 127.4 MB retained
  3. Array         2,341,892 items    |  89.1 MB retained
  4. Map             189,441 objects  |  67.3 MB retained
  5. Promise         441,291 objects  |  34.8 MB retained

Suspected cause: PR #1847 adds per-message tracing middleware that calls addListener()
on each message but never calls removeListener() on completion.
Diff in event-processor/middleware/tracer.js confirms missing cleanup.
Request: confirm root cause, advise on hotfix or rollback to v3.6.8.""",

    """\
[P2 HIGH] Kubernetes pod crash-loop after config update — ML inference service down

Service: inference-api v4.2.0 (Python 3.11, PyTorch 2.1.0) on prod-gpu-cluster
Deployed: 2024-11-13T22:15Z | First crash: 2024-11-13T22:17Z | Currently: CrashLoopBackOff

kubectl logs inference-api-7c9d4f-m2xpk (last 30 lines before crash):

2024-11-13T22:17:03Z INFO  Loading model: llama-3.1-8b-instruct-q4 from /models/cache
2024-11-13T22:17:08Z INFO  CUDA devices found: 2 (A100-80GB × 2)
2024-11-13T22:17:09Z INFO  Initialising tokenizer from /models/tokenizer
2024-11-13T22:17:11Z INFO  Model load started — estimated 45s
2024-11-13T22:17:34Z WARN  CUDA memory fragmentation detected — running defrag
2024-11-13T22:17:56Z INFO  Model loaded: 7.2GB VRAM used (of 80GB available)
2024-11-13T22:18:01Z INFO  Starting HTTP server on :8080
2024-11-13T22:18:03Z INFO  Health check endpoint /health registered
2024-11-13T22:18:04Z ERROR Failed to bind metrics endpoint :9090 — address already in use
2024-11-13T22:18:04Z ERROR SIGTERM received — initiating graceful shutdown
2024-11-13T22:18:09Z INFO  Shutdown complete

kubectl describe pod inference-api-7c9d4f-m2xpk:
  Liveness probe failed: HTTP probe failed with statuscode: 000
  Readiness probe failed: connection refused (port 8080)
  Last State: Terminated | Reason: Error | Exit Code: 1

The metrics port conflict (:9090) appears to be caused by a prometheus-node-exporter
DaemonSet deployed at 22:14Z that now occupies :9090 on every node.
Inference service crashes before completing startup, triggering liveness failure loop.""",

    """\
[P3 MEDIUM] OAuth callback redirect_uri mismatch breaking SSO for enterprise customers

Affected: All customers using SAML/OAuth SSO through Okta, Azure AD, or Google Workspace
Started: 2024-11-12T09:00Z (correlated with DNS migration to new load balancer IPs)
Reports received: 23 enterprise accounts, ~1,200 affected users

OAuth error log (auth-service v6.3.1, 2024-11-12T09:05-09:45Z):

09:05:12 ERROR OAuthCallbackError: redirect_uri_mismatch
  client_id=ent_okta_prod_4821
  received_redirect_uri=https://app-new.example.com/auth/callback
  registered_redirect_uri=https://app.example.com/auth/callback
09:07:33 ERROR OAuthCallbackError: redirect_uri_mismatch
  client_id=ent_azure_prod_1923
  received_redirect_uri=https://app-new.example.com/auth/callback
  registered_redirect_uri=https://app.example.com/auth/callback
09:12:44 ERROR SAMLAssertionError: invalid_destination
  issuer=https://sso.enterprise-customer.com/saml/metadata
  destination=https://app-new.example.com/saml/acs
  expected=https://app.example.com/saml/acs

Root cause hypothesis: DNS migration on 2024-11-12 changed CNAME from
app.example.com → app-new.example.com, but OAuth app registrations still
reference old URI. SP metadata in customer IdPs also references old ACS URLs.

Affected customers have been notified. Need: updated redirect_uri in all
OAuth app registrations + guidance to enterprise customers to update SP metadata.""",
]

_TRIAGE_ASKS = [
    "Classify this support ticket, assign priority (P1–P4), and route to the correct team with justification.",
    "Triage this ticket: determine severity, category, applicable SLA, and draft an initial customer response.",
    "Determine if this needs immediate escalation. If yes, specify escalation path and required approvals.",
    "Categorise and prioritise this request, then outline the step-by-step resolution process with owner assignments.",
    "Assess customer impact and business risk. Classify urgency, assign to a team, and set resolution timeline.",
]


class SupportTriageWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.SUPPORT_TRIAGE

    def generate_prompt(self) -> str:
        # 40% long tickets with trace/log attachments to overlap with CR payload sizes
        pool = _LONG_TICKETS if random.random() < 0.40 else _TICKETS
        ticket = random.choice(pool)
        ask = random.choice(_TRIAGE_ASKS)
        return f"{ask}\n\nTICKET:\n{ticket}"
