"""Structured requirements extracted from the Senior AI Engineer JD."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobRequirements:
    """Key requirements for the Senior AI Engineer — Founding Team role."""

    title: str = "Senior AI Engineer — Founding Team"
    company: str = "Redrob AI"
    min_experience: float = 4.0
    ideal_experience: float = 6.5
    max_experience: float = 12.0
    preferred_locations: tuple = (
        "noida", "pune", "delhi", "gurgaon", "gurugram", "bangalore", "bengaluru",
        "hyderabad", "mumbai", "pune", "noida",
    )
    preferred_country: str = "India"
    preferred_work_modes: tuple = ("hybrid", "flexible", "remote", "onsite")

    # Titles that strongly indicate fit (ordered by strength)
    strong_titles: tuple = (
        "senior ai engineer", "lead ai engineer", "staff machine learning engineer",
        "senior machine learning engineer", "senior ml engineer", "senior nlp engineer",
        "machine learning engineer", "ml engineer", "ai engineer", "nlp engineer",
        "applied ml engineer", "applied scientist", "senior data scientist",
        "recommendation systems engineer", "search engineer", "ranking engineer",
        "ai specialist", "ai research engineer", "applied ai engineer",
        "data scientist", "junior ml engineer",
    )

    # Titles that are keyword-stuffer traps — heavy penalty
    trap_titles: tuple = (
        "marketing manager", "hr manager", "accountant", "content writer",
        "graphic designer", "sales executive", "customer support", "civil engineer",
        "mechanical engineer", "operations manager", "business analyst",
        "project manager",
    )

    # Core skills the JD actually needs (with weights)
    core_skills: dict = field(default_factory=lambda: {
        "embeddings": 1.0, "sentence transformers": 1.0, "semantic search": 1.0,
        "information retrieval": 1.0, "vector": 0.9, "faiss": 0.9, "pinecone": 0.9,
        "milvus": 0.9, "elasticsearch": 0.8, "opensearch": 0.8, "pgvector": 0.8,
        "qdrant": 0.8, "weaviate": 0.8, "rag": 0.9, "retrieval": 0.9,
        "ranking": 0.9, "learning to rank": 1.0, "recommendation systems": 0.8,
        "llm": 0.8, "fine-tuning": 0.8, "fine-tuning llms": 0.9, "lora": 0.8,
        "qlora": 0.8, "peft": 0.7, "nlp": 0.7, "pytorch": 0.7, "tensorflow": 0.6,
        "hugging face": 0.7, "transformers": 0.7, "python": 0.5,
        "ndcg": 0.6, "evaluation": 0.5, "a/b test": 0.5,
    })

    # Skills that indicate wrong specialization per JD
    wrong_specialization: tuple = (
        "computer vision", "speech recognition", "robotics", "image classification",
        "object detection", "gan", "gans",
    )

    # Consulting firms — career penalty if entire career
    consulting_firms: tuple = (
        "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "lti", "mindtree", "mphasis",
    )

    # Product/tech companies — career bonus
    product_signals: tuple = (
        "product", "platform", "startup", "series", "saas", "scale",
        "recommendation", "search", "ranking", "retrieval", "embedding",
        "pipeline", "production", "deployed", "a/b", "ndcg", "mrr",
    )

    jd_text: str = """
Senior AI Engineer Founding Team at Redrob AI talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems for recruiters.
Production experience with embeddings-based retrieval, vector databases, hybrid search.
Strong Python. Evaluation frameworks for ranking: NDCG, MRR, MAP, A/B testing.
LLM fine-tuning LoRA QLoRA PEFT. Learning-to-rank models.
Ship end-to-end ranking search recommendation systems at product companies.
Not pure research. Not framework-only LangChain tutorials. Writes production code.
Location Pune Noida India hybrid. 5-9 years experience applied ML AI.
Mentoring team growth. Scrappy product engineering with deep ML systems knowledge.
"""


JD = JobRequirements()
