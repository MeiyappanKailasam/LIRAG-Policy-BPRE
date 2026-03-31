"""
Configuration Module for LIRAG Policy Agent
=============================================
Centralized configuration for queries and settings.
Change the QUERY variable here to evaluate all scripts with a different query.
"""

# ============================================================================
# MAIN QUERY VARIABLE - Change this to test all scripts with a different query
# ============================================================================
QUERY = "What are the objectives and policy goals of government initiatives in education and scholarships, and how do they support inclusion and national development?"

# ============================================================================
# ALTERNATIVE QUERIES (examples for quick testing)
# ============================================================================
ALTERNATIVE_QUERIES = {
    "disability_scholarships": "What are the eligibility conditions for post-matric scholarships for students with disabilities?",
    "income_limit": "What is the income limit to apply for post-matric scholarship?",
    "fees_covered": "What fees are covered under scholarship schemes?",
    "aadhaar_requirement": "Is Aadhaar mandatory for scholarship disbursal?",
    "hostel_accommodation": "Is hostel accommodation covered under this scheme?",
}

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
# Retrieval settings
RETRIEVAL_TOP_K = 10  # Number of clauses to retrieve
POLICY_FILTER_TOP_K = 5  # Final clauses to use for generation/verification

# Model settings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
USE_LLM_VERIFICATION = True  # Use LLM-based verification (LIRAG mode)
USE_BASELINE = False  # Use dense-only baseline

# API settings
GROQ_MODEL = "mixtral-8x7b-32768"  # Groq verification model
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI fallback model

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
In any Python file, import and use:

    from config import QUERY, ALTERNATIVE_QUERIES
    
    # Use main query
    answer, clauses = policy_agent(QUERY)
    
    # Or use alternative
    answer, clauses = policy_agent(ALTERNATIVE_QUERIES['income_limit'])
    
    # Or easily switch queries
    test_queries = [QUERY, ALTERNATIVE_QUERIES['disability_scholarships']]
    for q in test_queries:
        result = policy_agent(q)
"""
