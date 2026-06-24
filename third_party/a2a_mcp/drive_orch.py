"""Drive the a2a_mcp orchestrator (:10101) over A2A, MULTI-TURN: auto-answer the
planner's clarifying questions (same contextId) until the workflow completes and
fans out to the air/hotel/car specialists. Prints each turn's final state."""
import asyncio
import sys
from uuid import uuid4

import httpx
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendStreamingMessageRequest

ORCH = "http://localhost:10101"

# One rich answer that satisfies the planner's decision tree for most questions.
ANSWER = (
    "Use Narita (NRT). Economy class. Any reputable 4-star hotel near Shinjuku is "
    "fine. A standard mid-size rental car is fine. Please proceed with your best "
    "recommended options and complete the flight, hotel, and car bookings now."
)


def status_of(res):
    st = getattr(res, "status", None)
    if st is None:
        return None, None
    txt = ""
    msg = getattr(st, "message", None)
    if msg is not None:
        for p in msg.parts:
            r = getattr(p, "root", p)
            txt += getattr(r, "text", "") or ""
    return getattr(st, "state", None), txt


async def drive(query: str) -> None:
    context_id = uuid4().hex
    msg = query
    async with httpx.AsyncClient(timeout=240.0) as hc:
        card = await A2ACardResolver(hc, ORCH).get_agent_card()
        client = A2AClient(hc, card)
        for turn in range(1, 7):
            payload = {"message": {"role": "user", "parts": [{"kind": "text", "text": msg}],
                                   "messageId": uuid4().hex, "contextId": context_id}}
            req = SendStreamingMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
            final_state, question, chunks = None, None, 0
            try:
                async for chunk in client.send_message_streaming(req):
                    chunks += 1
                    res = getattr(chunk.root, "result", chunk.root)
                    state, txt = status_of(res)
                    if state is not None:
                        s = str(state)
                        if "input_required" in s:
                            final_state, question = "input_required", txt
                        elif "completed" in s:
                            final_state = "completed"
                        elif "working" in s and final_state is None:
                            final_state = "working"
            except Exception as e:  # noqa: BLE001
                print(f"  turn {turn}: STREAM ERROR {type(e).__name__}: {str(e)[:160]}")
                return
            print(f"  turn {turn}: {chunks} chunks, final={final_state}"
                  + (f", Q={question[:80]!r}" if question else ""))
            if final_state == "input_required":
                msg = ANSWER
                continue
            break
        print("DONE")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Plan a trip."
    asyncio.run(drive(q))
