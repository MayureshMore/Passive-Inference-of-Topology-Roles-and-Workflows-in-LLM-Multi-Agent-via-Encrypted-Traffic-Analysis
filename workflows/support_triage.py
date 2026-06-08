"""
Support triage workflow — ticket classified, routed, and escalated.
Characteristic traffic: short initial payload, rapid small responses,
and a single escalation hop (shorter burst structure than research workflow).
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
        ticket = random.choice(_TICKETS)
        ask = random.choice(_TRIAGE_ASKS)
        return f"{ask}\n\nTICKET:\n{ticket}"
