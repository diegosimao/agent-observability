"""Agent tools. The "external systems" are simulated, with injected latency and
failures so the dashboards have something to show."""

import random
import time

from langchain_core.tools import ToolException, tool

from . import chaos

_ORDERS = {
    "1001": {"status": "delivered", "delivered_on": "2026-10-02", "items": 2, "total": 349.90},
    "1002": {"status": "in transit", "estimated_delivery": "2026-10-21", "items": 1, "total": 89.90},
    "1003": {"status": "awaiting payment", "items": 3, "total": 1299.00},
    "1004": {"status": "cancelled", "reason": "out of stock", "total": 59.90},
}

_POLICIES = {
    "returns": "Free returns within 30 days of delivery, unused and in the original packaging.",
    "exchange": "Size or color exchanges within 30 days; the first exchange shipment is free.",
    "warranty": "90-day statutory warranty plus the manufacturer's warranty when stated on the product page.",
    "refund": "Refunds within 7 business days after the return is received, to the original payment method.",
}


def _latency(mean_s: float) -> None:
    time.sleep(random.lognormvariate(0, 0.5) * mean_s * chaos.state.latency_multiplier)


@tool
def get_order(order_id: str) -> dict:
    """Look up an order's status, total and delivery estimate by order number."""
    _latency(0.15)
    order = _ORDERS.get(order_id.strip().lstrip("#"))
    if order is None:
        raise ToolException(f"Order {order_id} not found in the ERP.")
    return {"order_id": order_id, **order}


@tool
def calculate_shipping(zip_code: str) -> dict:
    """Calculate shipping cost and delivery time for a Brazilian ZIP code (CEP)."""
    _latency(0.4)
    # The "carrier API" is the fragile part of the system.
    if random.random() < chaos.state.tool_error_rate:
        raise ToolException("Carrier API timeout (504).")
    digits = "".join(c for c in zip_code if c.isdigit())
    region = int(digits[0]) if digits else 0
    return {"zip_code": zip_code, "delivery_days": 2 + region, "cost": round(12.9 + region * 3.5, 2)}


@tool
def get_policy(topic: str) -> str:
    """Return the store policy on returns, exchange, warranty or refund."""
    _latency(0.05)
    key = topic.lower()
    for name, text in _POLICIES.items():
        if name[:5] in key:
            return text
    return "No policy found for this topic."


@tool
def create_ticket(summary: str, priority: str = "normal") -> dict:
    """Open a ticket for a human agent. Priority: low, normal or high."""
    _latency(0.2)
    return {"ticket": f"TK-{random.randint(10000, 99999)}", "priority": priority, "sla_hours": 4 if priority == "high" else 24}


TOOLS = [get_order, calculate_shipping, get_policy, create_ticket]


def tool_error_to_model(exc: Exception, request) -> str | None:
    """Used by ToolErrorMiddleware: expected failures (ToolException) go back to
    the model as an error ToolMessage, so it can recover. The exception leaves
    the tool first, so the tool run is marked as an error in LangSmith, which is
    NOT what happens with `handle_tool_error=True` (that records the failure as a
    success). Any other exception propagates and fails the whole trace."""
    if isinstance(exc, ToolException):
        return f"Error: {exc}"
    return None
