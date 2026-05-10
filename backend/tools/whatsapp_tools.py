"""
Simulated WhatsApp tool.

This does NOT integrate with a real provider; it persists campaign records
to the database so the rest of the system (UI, audit, analytics) can work
end-to-end.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from backend.database.db import session_scope
from backend.database.models import WhatsAppCampaign
from backend.services.logging_service import get_logger
from backend.utils.constants import CampaignStatus

logger = get_logger(__name__)


def send_whatsapp_message(
    *,
    customer_id: int,
    phone: str,
    message: str,
    campaign_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulate sending a WhatsApp message and persist the campaign record.

    Returns:
        {
            "status": "SENT",
            "phone": "+91...",
            "message": "...",
            "timestamp": "2024-01-01T00:00:00Z",
            "campaign_id": int,
            "campaign_run_id": str | None
        }
    """
    if not message or not phone:
        return {
            "status": CampaignStatus.FAILED.value,
            "phone": phone,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": "phone or message missing",
        }

    sent_at = datetime.utcnow()
    record_id: Optional[int] = None
    try:
        with session_scope() as s:
            row = WhatsAppCampaign(
                customer_id=customer_id,
                phone=phone,
                message=message,
                status=CampaignStatus.SENT.value,
                sent_at=sent_at,
                campaign_run_id=campaign_run_id,
            )
            s.add(row)
            s.flush()
            record_id = row.id

        logger.info(
            "Simulated WhatsApp send | customer_id=%s | phone=%s | run=%s",
            customer_id,
            phone,
            campaign_run_id,
        )
        return {
            "status": CampaignStatus.SENT.value,
            "phone": phone,
            "message": message,
            "timestamp": sent_at.isoformat() + "Z",
            "campaign_id": record_id,
            "campaign_run_id": campaign_run_id,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("WhatsApp simulation failed: %s", exc)
        return {
            "status": CampaignStatus.FAILED.value,
            "phone": phone,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(exc),
            "campaign_run_id": campaign_run_id,
        }
