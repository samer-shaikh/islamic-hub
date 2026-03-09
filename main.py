# main.py — Islamic Quiz API with pg_trgm Duplicate Detection
# Run: uvicorn main:app --reload
# pip install fastapi uvicorn asyncpg pydantic
#
# ┌─────────────────────────────────────────────────────┐
# │  HOW pg_trgm WORKS                                  │
# │                                                     │
# │  Old way:  Python fetches ALL rows → loops each     │
# │            → slow as DB grows (O(n))                │
# │                                                     │
# │  New way:  PostgreSQL breaks text into trigrams     │
# │            (3-char chunks) + GIN index              │
# │            → DB finds matches in ONE indexed query  │
# │            → fast even with 100,000+ questions      │
# └─────────────────────────────────────────────────────┘

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, Literal
import asyncpg
import os

app = FastAPI(title="Islamic Hub - Quiz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:samer786@localhost:5432/islamic_hub"
)

# ── SIMILARITY THRESHOLD ─────────────────────────────
# 0.45 = 45% trigram similarity triggers duplicate warning
# Lower  → catches more (more false positives)
# Higher → only catches very close matches
SIMILARITY_THRESHOLD = 0.45

# ── SCHEMAS ──────────────────────────────────────────

class QuizQuestion(BaseModel):
    category: Literal["Quran", "Hadith", "Fiqh", "Seerah", "Aqeedah", "Other"]
    difficulty: Literal["Easy", "Medium", "Hard"]
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: Optional[str] = None
    entered_by: Optional[str] = "Anonymous"

    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class ConfirmSave(BaseModel):
    category: Literal["Quran", "Hadith", "Fiqh", "Seerah", "Aqeedah", "Other"]
    difficulty: Literal["Easy", "Medium", "Hard"]
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: Optional[str] = None
    entered_by: Optional[str] = "Anonymous"

# ── DB ────────────────────────────────────────────────

async def get_db():
    return await asyncpg.connect(DATABASE_URL)

# ── STARTUP: enable pg_trgm + create table + GIN index ──

@app.on_event("startup")
async def startup():
    conn = await get_db()
    try:
        # 1. Enable the pg_trgm extension (only needs to run once ever)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

        # 2. Create the questions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id             SERIAL PRIMARY KEY,
                category       VARCHAR(50)  NOT NULL,
                difficulty     VARCHAR(10)  NOT NULL,
                question       TEXT         NOT NULL,
                option_a       TEXT         NOT NULL,
                option_b       TEXT         NOT NULL,
                option_c       TEXT         NOT NULL,
                option_d       TEXT         NOT NULL,
                correct_answer CHAR(1)      NOT NULL,
                explanation    TEXT,
                entered_by     VARCHAR(100) DEFAULT 'Anonymous',
                created_at     TIMESTAMP    DEFAULT NOW()
            );
        """)

        # 3. Create GIN index on the question column using trigrams
        #    This is what makes similarity search FAST
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_questions_trgm
            ON quiz_questions
            USING GIN (question gin_trgm_ops);
        """)

        print("✅ pg_trgm enabled, table ready, GIN index created.")
    finally:
        await conn.close()

# ── HELPER: insert a question ─────────────────────────

async def insert_question(conn, q):
    return await conn.fetchrow("""
        INSERT INTO quiz_questions
            (category, difficulty, question, option_a, option_b,
             option_c, option_d, correct_answer, explanation, entered_by)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        RETURNING id, created_at
    """,
        q.category, q.difficulty, q.question,
        q.option_a, q.option_b, q.option_c, q.option_d,
        q.correct_answer, q.explanation, q.entered_by
    )

# ── ROUTES ────────────────────────────────────────────

@app.post("/quiz/check", status_code=200)
async def check_duplicate(q: QuizQuestion):
    """
    Step 1: Check for similar questions using pg_trgm GIN index.

    PostgreSQL does the similarity check internally using trigrams —
    no Python loop, no fetching all rows. Just one fast indexed query.

    Returns:
      status: "clear"      → saved immediately, no duplicates
      status: "duplicates" → similar questions found, show comparison UI
    """
    conn = await get_db()
    try:
        await conn.execute(f"SELECT set_limit({SIMILARITY_THRESHOLD});")

        similar_rows = await conn.fetch("""
            SELECT
                id,
                question,
                option_a,
                option_b,
                option_c,
                option_d,
                correct_answer,
                category,
                ROUND( similarity(question, $1)::numeric * 100 ) AS score
            FROM quiz_questions
            WHERE question % $1
            ORDER BY similarity(question, $1) DESC
            LIMIT 3;
        """, q.question)

        if similar_rows:
            matches = [
                {
                    "id":             row["id"],
                    "question":       row["question"],
                    "option_a":       row["option_a"],
                    "option_b":       row["option_b"],
                    "option_c":       row["option_c"],
                    "option_d":       row["option_d"],
                    "correct_answer": row["correct_answer"],
                    "category":       row["category"],
                    "similarity":     int(row["score"]),
                }
                for row in similar_rows
            ]
            return {"status": "duplicates", "matches": matches}

        # No duplicates — save right away
        row = await insert_question(conn, q)
        return {
            "status":  "clear",
            "id":      row["id"],
            "message": "JazakAllah Khair! Question saved."
        }

    finally:
        await conn.close()


@app.post("/quiz/force-save", status_code=201)
async def force_save(q: ConfirmSave):
    """Step 2: User confirmed it's NOT a duplicate — save it anyway."""
    conn = await get_db()
    try:
        row = await insert_question(conn, q)
        return {"success": True, "id": row["id"], "message": "Question saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()


@app.get("/quiz/all")
async def get_all(category: Optional[str] = None, difficulty: Optional[str] = None):
    conn = await get_db()
    try:
        query = "SELECT * FROM quiz_questions WHERE 1=1"
        params = []
        if category:
            params.append(category)
            query += f" AND category = ${len(params)}"
        if difficulty:
            params.append(difficulty)
            query += f" AND difficulty = ${len(params)}"
        query += " ORDER BY created_at DESC"
        rows = await conn.fetch(query, *params)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@app.get("/quiz/stats")
async def get_stats():
    conn = await get_db()
    try:
        total   = await conn.fetchval("SELECT COUNT(*) FROM quiz_questions")
        by_cat  = await conn.fetch("SELECT category, COUNT(*) as count FROM quiz_questions GROUP BY category")
        by_diff = await conn.fetch("SELECT difficulty, COUNT(*) as count FROM quiz_questions GROUP BY difficulty")
        return {
            "total":         total,
            "by_category":   {r["category"]:  r["count"] for r in by_cat},
            "by_difficulty": {r["difficulty"]: r["count"] for r in by_diff},
        }
    finally:
        await conn.close()


@app.delete("/quiz/{question_id}")
async def delete_question(question_id: int):
    conn = await get_db()
    try:
        result = await conn.execute("DELETE FROM quiz_questions WHERE id = $1", question_id)
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Question not found")
        return {"success": True, "message": f"Question {question_id} deleted."}
    finally:
        await conn.close()
