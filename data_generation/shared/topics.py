# ── Topics & Domains ──────────────────────────────────────────────────────────

LONG_CONTEXT_TOPICS = [
    "scientific expedition", "historical mystery", "company acquisition",
    "medical case study", "archaeological discovery", "legal investigation",
    "engineering project", "financial audit", "environmental study",
    "biographical profile", "travel journal", "product development timeline",
]

DIALOGUE_TOPICS = [
    "planning a home renovation", "job interview preparation", "trip planning",
    "learning a new skill", "medical consultation", "software debugging session",
    "investment planning", "recipe development", "book club discussion",
    "startup pitch preparation", "research collaboration", "apartment hunting",
]

AGENTIC_DOMAINS = [
    "data pipeline debugging", "cloud infrastructure setup",
    "multi-step file processing", "API integration workflow",
    "database migration", "security audit", "deployment automation",
    "document processing pipeline", "web scraping and analysis",
    "ML experiment tracking", "codebase refactoring", "report generation",
]

# ── Modifiers & Constraints (For generating 10,000+ unique seeds) ─────────────

LONG_CONTEXT_CONSTRAINTS = [
    "the story contains conflicting accounts from two different characters",
    "the critical fact is buried in a footnote or a postscript",
    "a crucial piece of evidence is initially dismissed as irrelevant",
    "there is a translation error that masks the true meaning",
    "the narrative jumps back and forth in time confusingly",
    "the document has missing pages or redacted sections",
    "the text is written in an extremely formal and dense academic style",
    "the author of the document is revealed to be unreliable at the end"
]

DIALOGUE_CONSTRAINTS = [
    "one speaker is extremely impatient and keeps interrupting",
    "there is a misunderstanding about a key term that isn't resolved until later",
    "the user changes their mind twice about what they want",
    "the assistant makes a minor mistake early on that the user corrects",
    "the conversation spans multiple days with casual greetings in between",
    "the user keeps getting distracted by a secondary, irrelevant issue",
    "the tone of the conversation starts formal but becomes very casual",
    "the user asks hypothetical 'what if' questions before settling on reality"
]

AGENTIC_CONSTRAINTS = [
    "a legacy system API abruptly crashes halfway through and requires a retry",
    "authentication tokens expire during the run causing a permission error",
    "the agent receives a massive JSON payload and must extract one nested ID",
    "a requested file is locked, forcing the agent to find a backup path",
    "the initial database query returns zero results due to a typo in the user prompt",
    "a critical dependency is missing and the agent must install it first",
    "the agent is restricted to read-only access for the first 3 steps",
    "an intermediate script goes into an infinite loop and must be killed"
]
