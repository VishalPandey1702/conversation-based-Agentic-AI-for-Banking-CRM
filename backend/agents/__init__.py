"""Agent package - one module per single-responsibility agent."""
from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.customer_discovery_agent import CustomerDiscoveryAgent
from backend.agents.scoring_agent import ScoringAgent
from backend.agents.recommendation_agent import RecommendationAgent
from backend.agents.outreach_agent import OutreachAgent
from backend.agents.campaign_agent import CampaignAgent
from backend.agents.conversation_agent import ConversationAgent, conversation_agent

__all__ = [
    "SupervisorAgent",
    "CustomerDiscoveryAgent",
    "ScoringAgent",
    "RecommendationAgent",
    "OutreachAgent",
    "CampaignAgent",
    "ConversationAgent",
    "conversation_agent",
]
