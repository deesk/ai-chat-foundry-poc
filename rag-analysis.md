# RAG Behaviour Analysis Report
 
**Project:** Linfox Melbourne Logistics RAG Chatbot  
**Author:** Sandesh (GitHub: deesk)  
 
---
 
## TL;DR
 
**What happens when RAG gets it wrong and why**
 
1. Cutting content into fixed chunks causes incomplete answers. When related information gets split across two chunks and the chunks do not share enough keywords to link them, the system retrieves only one chunk. The AI answers confidently with half the information, unaware the other half exists. This is the most common failure mode. [[details]](#part-1-initial-testing-original-dataset-no-loading-zones)
 
2. How you name and write content affects what gets found. When two different sections use similar naming patterns, the system can confuse them and mix answers together. Distinct naming keeps sections separate. Keywords that appear inside the content itself, not just in headings, create stronger matches. The more a keyword repeats within a chunk, the more strongly that chunk gets associated with that topic. [[details]](#part-2-naming-convention-testing-alpha-vs-numeric-loading-zones)
 
3. The most dangerous failure is a confident wrong answer. The AI does not know what it does not know. When data is split across chunks and only half is retrieved, the AI answers as if the half it has is everything. A sorry response at least signals uncertainty. A confident wrong answer does not. [[details]](#part-1-initial-testing-original-dataset-no-loading-zones)
 
4. The chatbot (GPT model) uses both retrieved data and conversation history by default. Within a session, it draws on previous answers when relevant. Unrelated topics do not bleed across turns. However, follow-up questions like "is that all?" fail for a specific reason. These questions contain no domain keywords so the knowledge base finds nothing relevant to retrieve. With no retrieved data, the system fires a fixed sorry response that overrides everything including conversation history. [[details]](#part-3-conversation-history-effect)
 
5. Testing must be done in isolated sessions. The same question asked at different points in a conversation can give different results due to conversation history accumulating across turns. Each test must start fresh. [[details]](#part-3-conversation-history-effect)
 
6. "Only use this context" system prompt instructions do not work. By default GPT uses both RAG output and conversation history to formulate answers. Telling it what it can use changes nothing. Only telling it what it cannot use changes behaviour. Explicit boundary markers like `[CONTEXT START]` and `[CONTEXT END]` combined with "Ignore conversation history" ensured GPT answered strictly from RAG output only, not from previous conversation history. [[details]](#part-5-system-prompt-evolution)
 
7. The word "context" means something different to GPT than to developers. The system prompt instructed GPT to "only use the context data provided" intending RAG output only. GPT interpreted "context" as everything visible to it, both RAG output and conversation history. To confirm this, GPT was temporarily prompted to print what it used to answer each question. The output proved GPT was treating conversation history as context, leading to the fix in point 6. [[details]](#part-4-investigation-resolving-the-context-ambiguity-mystery)
 
---
 
## Detailed Analysis
 
---
 
## How This RAG System Works
 
Technical flow:
 
1. Source `.txt` file split into chunks of 4 sentences each (fixed-size chunking).
2. Each chunk converted to a 100-dimensional vector using `text-embedding-3-small`.
3. All vectors stored in Azure AI Search as an index.
4. When a question is asked, it is also converted to a vector.
5. Azure Search compares the question vector against all chunk vectors.
6. Top 5 most similar chunks returned (`k_nearest_neighbors=5`).
7. Top 5 chunks injected into the GPT-4o-mini system prompt as `context`.
8. GPT-4o-mini reads context and full conversation history, then formulates an answer.
 
Key constraint: GPT only sees the top 5 chunks. Answers in chunk 6 or beyond are never retrieved.
 
[DIAGRAM 1: place rag_query_flow.png here]
 
[DIAGRAM 2: place data_pipeline.png here]
 
---
 
## How Context and Conversation History Work Together
 
Every turn, GPT receives two things combined:
 
```python
messages = prompt_messages + messages
```
 
`prompt_messages` is the system prompt built from RAG context:
 
```
role: system
content: "You are Melbourne logistics assistant...
          [CONTEXT START]
          top 5 chunks from Azure Search
          [CONTEXT END]"
```
 
`messages` is the full conversation history from the chat interface:
 
```
role: user       content: "list all zones"
role: assistant  content: "Loading zones are Zone A..."
role: user       content: "list delivery zones"
role: assistant  content: "Delivery zones are Zone 1..."
role: user       content: "list all zones"   (current question)
```
 
The chat interface sends the entire conversation history with every request. GPT has no server-side memory. The app creates the illusion of memory by re-sending all previous turns each time. Turn 50 sends the previous 49 turns plus the current question. Turn 100 sends the previous 99 turns plus the current question.
 
RAG searches fresh every turn using only the current question. Conversation history does not affect which chunks are retrieved. These are two separate and independent systems.
 
Two separate sources of context for GPT:
 
RAG retrieval provides the top 5 chunks from Azure Search, searched fresh every turn using the current question only.
 
Conversation history provides all previous questions and answers in the session, re-sent by the chat interface every turn.
 
---
 
## Dataset Structure
 
Source: `src/static/data/melbourne-logistics.txt`
 
Depot locations: 3
Delivery zones: 4 (Zone 1-4)
Freight types: 4 (Ambient, Cold chain, Frozen, Dangerous goods)
Common delay reasons: 4 (Traffic, Weather, Capacity, Customs)
Driver shift times: 4 shifts
Escalation process: 4 levels
Loading zones added for testing: 7
 
---
 
## Part 1: Initial Testing (Original Dataset, No Loading Zones)
 
### What worked correctly
 
"What are the delivery zones?" returned all 4 zones. "What shift times do drivers work?" returned all 4 shifts. "How does the escalation process work?" returned all 4 levels. "How many depot locations?" returned all 3 depots. "What are common delay reasons?" returned only 2 of 4. "What freight types does Linfox handle?" returned only 3 of 4.
 
### Missing data: Freight types
 
Question: "What freight types does Linfox handle?"
Expected: Ambient, Cold chain, Frozen, Dangerous goods.
Returned: Ambient, Cold chain, Frozen. Dangerous goods missing.
 
GPT presented 3 freight types with full confidence, unaware a 4th exists. A confident wrong answer — no signal to the user that information is missing.
 
[SCREENSHOT 1: place ss1_freight_types.png here]
 
Root cause: fixed-size chunking. The 4-sentence cut placed "Dangerous goods" into a different chunk alongside delay reason content. Chunk data from `embeddings.csv`:
 
Chunk 4 retrieved correctly:
```
Token: "FREIGHT TYPES: Ambient: Standard palletised freight. 
        Cold chain: 2-8 degrees celsius. Frozen: Below -18 degrees celsius."
Embedding: [-0.063, 0.037, 0.062, -0.109, 0.053, 0.042, -0.061, -0.126 ...]
```
 
Chunk 5 NOT retrieved (mixed content dilutes vector):
```
Token: "Dangerous goods: ADG code compliant vehicles only. 
        COMMON DELAY REASONS: Traffic: Monash Freeway and Westgate Bridge peak hours. 
        Weather: Flooding in Laverton and Derrimut corridors."
Embedding: [-0.001, 0.067, 0.124, -0.000, -0.012, 0.125, -0.096, -0.045 ...]
```
 
Chunk 5 contains three unrelated topics mixed together. When converted to a vector, the meaning gets pulled in multiple directions at once — like mixing too many paint colours and ending up with a muddy result that does not clearly represent any single colour. The vector is not strongly associated with freight types or delay reasons, so it scores lower than expected for either query and may not make the top 5.
 
[DIAGRAM 3: place chunking_problem.png here]
 
### Missing data: Common delay reasons
 
Question: "What are common delay reasons?"
Expected: Traffic, Weather, Capacity, Customs.
Returned: Traffic and Weather only.
 
[SCREENSHOT 2: place delay_reasons_partial.png here]
 
Chunk 6 NOT retrieved (mixed with shift times):
 
```
Token: "Capacity: Peak periods October-December. 
        Customs: International freight clearance at Tullamarine. 
        DRIVER SHIFT TIMES: Early shift: 4am to 12pm."
Embedding: [-0.113, 0.083, 0.142, -0.109, 0.060, 0.028, 0.058, 0.092 ...]
```
 
Chunk 6 vector dominated by shift time semantics, poor match for delay reason queries.
 
### Why escalation worked but delay reasons did not
 
All 4 escalation levels landed in a single chunk:
 
```
Token: "Level 1: Driver contacts depot supervisor. 
        Level 2: Depot supervisor contacts operations manager. 
        Level 3: Operations manager contacts state manager. 
        Level 4: State manager contacts national operations."
Embedding: [0.004, 0.039, 0.175, -0.063, -0.024, -0.010, 0.052, -0.030 ...]
```
 
"Level" appears in every sentence, making this chunk highly relevant to escalation queries. All 4 levels in one embedding means one retrieval gets everything.
 
Whether a section lands in one chunk or multiple is determined by sentence count and where the 4-sentence boundary falls, not by logical section boundaries. Escalation got lucky. Freight types and delay reasons did not.
 
---

## Part 2: Naming Convention Testing (Alpha vs Numeric Loading Zones)
 
7 loading zones added in two variants. 
- Alpha naming: Zone A through Zone G. 
- Numeric naming: Zone 1 through Zone 7. 
Delivery zones already used numeric naming (Zone 1-4).
 
### Alpha naming results
 

### Numeric naming results

