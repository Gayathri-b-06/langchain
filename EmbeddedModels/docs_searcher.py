from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Loads API key from .env file
load_dotenv()

# =========================================================
# EMBEDDING MODELS
# =========================================================

# Model used for converting documents into vectors
docs_embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document"
)

# Model used for converting user query into vector
query_embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_query"
)

# =========================================================
# DOCUMENTS
# =========================================================

avengers_docs = [
    "The Avengers is a superhero team formed by Iron Man, Captain America, Thor, Hulk, Black Widow, and Hawkeye.",

    "Iron Man, also known as Tony Stark, is a genius billionaire who created advanced armored suits.",

    "Captain America, also known as Steve Rogers, is a super soldier who carries a vibranium shield.",

    "Thor is the God of Thunder from Asgard and wields the powerful hammer Mjolnir.",

    "Hulk is Bruce Banner who transforms into a giant green superhero with immense strength.",

    "Black Widow, also known as Natasha Romanoff, is a highly skilled spy and combat expert.",

    "Hawkeye, also known as Clint Barton, is a master archer and an important member of the Avengers.",

    "Loki is Thor's brother and one of the main villains in the Avengers universe.",

    "Thanos is a powerful villain who seeks all six Infinity Stones to balance the universe.",

    "The Infinity Gauntlet allows Thanos to use the power of all Infinity Stones.",

    "Avengers: Endgame is one of the most popular Marvel movies where the Avengers fight Thanos.",

    "Doctor Strange is the master of mystic arts and helps the Avengers in saving reality.",

    "Spider-Man, also known as Peter Parker, fights alongside the Avengers using his spider powers.",

    "Black Panther, also known as TChalla, is the king of Wakanda and a strong Avenger ally.",

    "Scarlet Witch, also known as Wanda Maximoff, has powerful reality-altering abilities.",

    "Vision is an android created using JARVIS, vibranium, and the Mind Stone.",

    "Ant-Man can shrink or grow using Pym Particles and helps in the time travel mission.",

    "Captain Marvel is one of the strongest heroes in the Marvel universe with cosmic powers.",

    "Nick Fury is the director of SHIELD and the person who initiates the Avengers project.",

    "Avengers headquarters refers to the main central base where the Avengers plan missions and operations."
]

# =========================================================
# USER QUERY
# =========================================================

query = "Where do the Avengers plan their missions and operations?"

# =========================================================
# GENERATE DOCUMENT EMBEDDINGS
# =========================================================

print("Generating document embeddings...\n")

docs_embeddings = docs_embedding_model.embed_documents(
    avengers_docs
)

# =========================================================
# GENERATE QUERY EMBEDDING
# =========================================================

print("Generating query embedding...\n")

query_embedding = query_embedding_model.embed_query(
    query
)

# =========================================================
# CALCULATE COSINE SIMILARITY
# =========================================================

scores = cosine_similarity(
    [query_embedding],
    docs_embeddings
)[0]

# =========================================================
# PRINT ALL SIMILARITY SCORES
# =========================================================

print("📊 All Similarity Scores")
print("-" * 70)

for i, (doc, score) in enumerate(zip(avengers_docs, scores)):
    print(f"[{i:2d}] Score: {score:.4f}")
    print(f"Document: {doc}")
    print()

# =========================================================
# GET TOP 3 RESULTS
# =========================================================

top_k = 3

top_results = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1],
    reverse=True
)[:top_k]

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n🔍 USER QUERY")
print(query)

print("\n✅ TOP MATCHING DOCUMENTS")
print("=" * 70)

for rank, (idx, score) in enumerate(top_results, start=1):

    print(f"\n🏆 Rank #{rank}")
    print(f"Document Index : {idx}")
    print(f"Similarity     : {score:.4f}")
    print(f"Matched Text   :")
    print(avengers_docs[idx])

    print("-" * 70)