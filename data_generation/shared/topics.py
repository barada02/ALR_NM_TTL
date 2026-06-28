# ── Long Context Collections ───────────────────────────────────────────────────

LONG_CONTEXT_TOPICS = [
    "scientific expedition", "historical mystery", "company acquisition",
    "medical case study", "archaeological discovery", "legal investigation",
    "engineering project", "financial audit", "environmental study",
    "biographical profile", "travel journal", "product development timeline",
    "geopolitical crisis simulation", "space telescope data analysis",
    "supply chain failure investigation", "patent litigation lawsuit",
    "rare disease outbreak tracking", "urban infrastructure redesign",
    "maritime salvage operation", "high-frequency trading incident review",
    "disaster response post-mortem", "heritage building restoration",
    "agricultural yield drop study", "telecommunications blackout report",
    "cybersecurity breach investigation", "deep-well geothermal exploration",
    "wildlife migration anomaly", "corporate espionage allegation",
    "academic fraud investigation", "subatomic particle research analysis"
]

LONG_CONTEXT_SETTINGS = [
    "subglacial Lake Tartarus in Antarctica", "dense unexplored sector of the Amazon basin",
    "virtual cleanroom of a silicon manufacturing plant", "high-security records vault of the Swiss central bank",
    "remote observatory in the Atacama desert", "submerged wreckage of a WW2 cargo submarine",
    "unincorporated industrial zone of Shenzhen", "closed-loop biosphere habitat in Arizona",
    "abandoned salt mine repurposed as a dark matter detector", "corporate headquarters of a multinational agricultural firm",
    "municipal water filtration facility of Chicago", "offshore gas platform in the North Sea",
    "automated logistics warehouse in Rotterdam", "containment building of a decommissioned nuclear reactor",
    "isolated research station in Svalbard", "server farm under a mountain in Norway",
    "highly classified military testing range in Nevada", "historical archive library in Kyoto",
    "quarantined ward of a tropical disease clinic", "construction site of a suspension bridge in Istanbul",
    "geostationary satellite telemetry database", "coastal wetlands reserve in Louisiana",
    "family office estate archives in Vienna", "cargo vessel cargo logs in transit across the Pacific",
    "high-voltage power transmission grid control center", "wind tunnel facility in Munich",
    "underground transit network in London", "astrophysics laboratory in Geneva",
    "hydrothermal vent field in the Mariana Trench", "desalination plant in Dubai"
]

LONG_CONTEXT_ROLES = [
    "a cynical lead structural engineer", "a meticulous forensic accountant",
    "a sleep-deprived clinical investigator", "an eager junior archaeologist",
    "a skeptical insurance claims adjuster", "a senior cybersecurity incident commander",
    "a defensive lead product manager", "a suspicious patent attorney",
    "a retired maritime salvage master", "a strict environmental compliance inspector",
    "a detail-oriented biostatistician", "a paranoid software architect",
    "a seasoned geopolitical analyst", "a stubborn chief geologist",
    "a meticulous historical biographer", "an anxious air traffic controller",
    "a strict agricultural inspector", "a chief telecommunications officer",
    "a forensic handwriting analyst", "a cynical hedge fund risk analyst",
    "a senior disaster recovery coordinator", "a stubborn manufacturing plant manager",
    "a meticulous aerospace technician", "a skeptical tax auditor",
    "a chief medical examiner"
]

LONG_CONTEXT_CONSTRAINTS = [
    "the story contains conflicting accounts from two different characters",
    "the critical fact is buried in a footnote or a postscript",
    "a crucial piece of evidence is initially dismissed as irrelevant",
    "there is a translation error that masks the true meaning",
    "the narrative jumps back and forth in time confusingly",
    "the document has missing pages or redacted sections",
    "the text is written in an extremely formal and dense academic style",
    "the author of the document is revealed to be unreliable at the end",
    "the critical fact is disguised as a typo in a series of measurements",
    "important details are presented as an offhand comment in a personal letter",
    "a key fact is hidden inside an appendiced table of irrelevant data",
    "the critical information is spread across three separate dates",
    "the document uses archaic jargon or obsolete technical terminology",
    "the narrator is writing under duress and uses subtle sarcasm to convey the truth",
    "the key fact is mentioned only once in a list of minor expenses",
    "the narrative is structured as a series of diary entries with missing dates",
    "the document contains a warning that was ignored by the management",
    "the truth is hidden in a transcript of a noisy radio transmission",
    "a critical measurement is given in an unusual unit system to confuse readers",
    "the narrator is trying to cover up their own mistake while reporting facts",
    "the critical fact is only revealed by comparing two different versions of a table",
    "the document is written in a dry, bureaucratic, passive-voice style",
    "the narrative is interrupted by system alerts and log entries",
    "the key fact is revealed in an interview transcript at the very end",
    "the document contains an encrypted note or coded sequence"
]

# ── Dialogue Collections ──────────────────────────────────────────────────────

DIALOGUE_TOPICS = [
    "planning a home renovation", "job interview preparation", "trip planning",
    "learning a new skill", "medical consultation", "software debugging session",
    "investment planning", "recipe development", "book club discussion",
    "startup pitch preparation", "research collaboration", "apartment hunting",
    "fitness goal planning", "purchasing a new car", "organizing a charity event",
    "arranging a family reunion", "discussing a performance review", "choosing a college major",
    "drafting a marketing campaign", "planning a conference layout", "negotiating a lease contract",
    "coordinating a movie production", "writing a podcast script", "setting up a home lab",
    "purchasing insurance coverage", "choosing a retirement community", "planning a wedding menu",
    "discussing garden design", "organizing a local sports tournament", "setting up a community pantry"
]

DIALOGUE_SETTINGS = [
    "over a video call with lag and interruptions", "during a noisy subway commute",
    "over casual text messages spanning several days", "in a formal board room meeting",
    "while walking through a crowded street market", "during a quiet coffee shop conversation",
    "in an elevator ride interrupted by multiple stops", "over a series of quick voice notes",
    "while cooking dinner in a busy kitchen", "during a late-night coding hackathon session",
    "while reviewing blueprints at a dusty construction site", "in a doctor's waiting room",
    "during a walk in a windy park", "over a high-stress Slack/Teams chat thread",
    "while waiting for a flight at an airport terminal", "during a museum tour",
    "while driving together on a long road trip", "over a series of emails with delayed responses",
    "during a shared lunch break at a noisy cafeteria", "while hiking up a steep trail",
    "in a quiet library archive section", "during a virtual reality workspace meeting",
    "over a walkie-talkie link with static", "while sitting in a traffic jam",
    "in a university laboratory during an experiment", "during a theater rehearsal intermission",
    "while assembling flat-pack furniture together", "over a discord voice channel while gaming",
    "in a hotel lobby during a check-in rush", "during a pet training session"
]

DIALOGUE_PERSONAS = [
    "a homeowner planning renovations with a contractor",
    "a job candidate interviewing with an HR director",
    "a traveler booking plans with an agent",
    "a student learning concepts from a tutor",
    "a patient discussing health with a specialist",
    "a junior developer getting guidance from a tech lead",
    "an investor discussing portfolio with a financial advisor",
    "a chef planning a menu with a restaurant owner",
    "a reader discussing books with a club member",
    "a founder pitching to a venture capitalist",
    "two researchers sharing notes from different fields",
    "a tenant discussing terms with a leasing agent",
    "a fitness client consulting a personal trainer",
    "a customer buying a car from a salesperson",
    "a community coordinator organizing volunteers",
    "a family member arranging plans with a planner",
    "an employee discussing career with a manager",
    "a student seeking advice from an advisor",
    "a client requesting updates from a designer",
    "a safety inspector reviewing compliance with a manager",
    "a director aligning project scope with a producer",
    "a podcast host interviewing a guest expert",
    "a tech buyer asking questions to a hardware vendor",
    "a policyholder claiming coverage with an adjuster",
    "a family representative planning care with a director"
]

DIALOGUE_CONSTRAINTS = [
    "one speaker is extremely impatient and keeps interrupting",
    "there is a misunderstanding about a key term that isn't resolved until later",
    "the user changes their mind twice about what they want",
    "the assistant makes a minor mistake early on that the user corrects",
    "the conversation spans multiple days with casual greetings in between",
    "the user keeps getting distracted by a secondary, irrelevant issue",
    "the tone of the conversation starts formal but becomes very casual",
    "the user asks hypothetical 'what if' questions before settling on reality",
    "one speaker is trying to hide a piece of bad news diplomatically",
    "the speakers keep getting interrupted by incoming phone calls or notifications",
    "both speakers are using highly specialized jargon and have to explain it to each other",
    "the dialogue is structured as a question-and-answer interview session",
    "one speaker has a poor memory and keeps asking for things to be repeated",
    "the conversation is a negotiation where both sides make compromises",
    "the speakers are planning a surprise and must speak in coded references",
    "the assistant is overly formal and uses robotic phrasing, which annoys the user",
    "the user is in a rush and demands bullet-point summaries rather than paragraphs",
    "the conversation keeps looping back to a disagreement about budget",
    "one speaker is a complete beginner and asks very basic, clarifying questions",
    "the dialogue has multiple side conversations that are eventually dismissed",
    "the speakers have opposite opinions on a design choice and argue politely",
    "the assistant has to explain a complex policy or set of rules twice",
    "the user is describing a dream or highly abstract concept",
    "the conversation starts with a complaint and ends with a mutual resolution",
    "one speaker keeps using idioms that the other speaker takes literally"
]

# ── Agentic Collections ────────────────────────────────────────────────────────

AGENTIC_DOMAINS = [
    "data pipeline debugging", "cloud infrastructure setup",
    "multi-step file processing", "API integration workflow",
    "database migration", "security audit", "deployment automation",
    "document processing pipeline", "web scraping and analysis",
    "ML experiment tracking", "codebase refactoring", "report generation",
    "container registry cleanup", "DNS routing troubleshooting", "IAM policy verification",
    "backup restoration testing", "log aggregation setup", "API gateway configuration",
    "microservice health monitoring", "load balancer failover test", "SSL certificate renewal automation",
    "database indexing optimization", "dependency vulnerability scanning", "secrets rotation workflow",
    "ci/cd pipeline performance profiling", "distributed cache warming", "serverless function testing",
    "network firewall rule auditing", "billing anomaly investigation", "rate-limiting configuration test"
]

AGENTIC_ENVIRONMENTS = [
    "a legacy Kubernetes cluster with memory leaks", "a hybrid cloud deployment with AWS and GCP",
    "a local development environment on Windows WSL2", "a secure banking sandbox with strict access control",
    "a serverless AWS Lambda environment using API Gateway", "a distributed database cluster across three regions",
    "a private bare-metal virtualization rack running Proxmox", "a continuous integration runner with limited disk space",
    "a microservices grid managed by HashiCorp Nomad", "a multi-tenant SaaS production environment",
    "a developer workstation running macOS with Homebrew", "a staging database with anonymized user records",
    "a legacy mainframe system exposed via a modern REST wrapper", "an edge computing node on a remote cell tower",
    "a compliance-audited healthcare data lake in GCP", "a high-performance compute cluster with Slurm scheduler",
    "a sandbox environment with mock billing services", "a headless Linux server with no external internet access",
    "a Docker Compose stack running on a single developer machine", "a federated GraphQL gateway aggregation layer",
    "a disaster recovery standby site in an isolated region", "a telemetry system collecting millions of IoT logs",
    "a secure CI/CD pipeline running on self-hosted runners", "an elastic search cluster with fragmented indices",
    "a multi-cloud data warehouse using BigQuery and Snowflake", "a legacy SVN repository migrated to GitLab",
    "a localized staging cluster running Minikube", "an API gateway with strict token verification middleware",
    "a high-frequency trading simulation engine", "a distributed cron-job runner with worker locks"
]

AGENTIC_TOOLS = [
    "Docker, AWS CLI, PostgreSQL, git, custom bash utility scripts",
    "kubectl, Helm, Prometheus, grep, curl",
    "Python, pandas, requests, beautifulsoup4, sqlite3",
    "gcloud CLI, BigQuery SDK, gsutil, jq, git",
    "terraform, ansible, ssh, systemctl, journalctl",
    "npm, webpack, node, git, custom shell scripts",
    "postgres CLI, pg_dump, pg_restore, aws s3, openssl",
    "docker-compose, curl, openssl, nginx, hosts file editor",
    "git, gh CLI, python, pylint, black, pytest",
    "aws cli, secretsmanager, vault, openssl, bash",
    "ping, traceroute, nslookup, iptables, tcpdump",
    "python, scikit-learn, mlflow, git, curl",
    "java, maven, sonar-scanner, git, docker",
    "redis-cli, curl, custom bash watcher script, systemctl",
    "splunk cli, grep, awk, curl, jq"
]

AGENTIC_CONSTRAINTS = [
    "a legacy system API abruptly crashes halfway through and requires a retry",
    "authentication tokens expire during the run causing a permission error",
    "the agent receives a massive JSON payload and must extract one nested ID",
    "a requested file is locked, forcing the agent to find a backup path",
    "the initial database query returns zero results due to a typo in the user prompt",
    "a critical dependency is missing and the agent must install it first",
    "the agent is restricted to read-only access for the first 3 steps",
    "an intermediate script goes into an infinite loop and must be killed",
    "disk space runs out on the runner, requiring a temporary directory cleanup",
    "the primary database is in read-only mode, forcing a failover check",
    "the network link to a secondary cloud provider drops and reconnects automatically",
    "a deprecated API parameter returns an unexpected 400 Bad Request error",
    "the agent must fetch information from two separate configs and merge them",
    "a webhook response is delayed by 15 seconds, forcing a polling loop",
    "an SSH key is missing passphrase parameters, requiring agent configuration",
    "the SSL certificate of the target domain is expired, forcing an insecure bypass flag",
    "a rate limit of 5 requests per minute is hit, requiring sleep intervals",
    "a database lock is held by a dead process, requiring a kill query first",
    "the output file path contains a space that breaks simple shell commands",
    "a configuration file has invalid YAML syntax that must be corrected programmatically",
    "the service account has delete permissions revoked, requiring a migration workaround",
    "a tool output returns raw XML that the agent must parse using regex or utility",
    "the agent has to compare hashes of two large files to verify integrity",
    "a critical environment variable is empty and must be loaded from a backup file",
    "the system timezone mismatch causes date calculations to fail, requiring UTC translation"
]
