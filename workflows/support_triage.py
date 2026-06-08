"""
Support triage workflow — ticket classified, routed, and escalated.
Characteristic traffic: short initial payload, rapid small responses,
and a single escalation hop (shorter burst structure than research workflow).
"""

from __future__ import annotations

import random

from .base import BaseWorkflow, WorkflowClass

_TICKETS = [
    "My account shows a charge of $149 but I'm on the free plan. Please investigate.",
    "The API is returning 503 errors intermittently since about 2 hours ago. High priority.",
    "I can't log in — it says 'invalid credentials' even after resetting my password.",
    "Feature request: please add dark mode to the dashboard.",
    "Our data export has been running for 6 hours and hasn't completed. Job ID: X-9182.",
    "The mobile app crashes on iOS 17.4 when opening the notifications panel.",
    "We need to upgrade from 10 to 50 seats ASAP — procurement is waiting on a quote.",
    "GDPR data deletion request for user ID 48821. Legal deadline: 3 days.",
    "Integration with Salesforce stopped syncing after the March update.",
    "Billing invoice PDF is corrupted — can't open it in Adobe Reader or Chrome.",
]

_TRIAGE_ASKS = [
    "Classify this support ticket, assign priority (P1-P4), and route to the right team.",
    "Triage this ticket: severity, category, SLA, and recommended first response.",
    "Determine if this needs immediate escalation and draft an initial customer reply.",
    "Categorise and prioritise this request, then outline the resolution steps.",
]


class SupportTriageWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.SUPPORT_TRIAGE

    def generate_prompt(self) -> str:
        ticket = random.choice(_TICKETS)
        ask = random.choice(_TRIAGE_ASKS)
        return f"{ask}\n\nTICKET:\n{ticket}"
