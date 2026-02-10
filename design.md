# Design Document

## Project: InfinityCircle
**Team:** InfinityCircle  
**Version:** 1.0  
**Date:** February 2026

---

## 1. System Overview

InfinityCircle is an AI-powered multilingual platform designed to bridge the gap between complex government schemes and rural citizens. The system leverages Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and cloud-based AI services to transform dense PDF documents into personalized, voice-enabled, WhatsApp-ready explanations.

### Core Capabilities

- **Intelligent Document Processing:** Extracts and understands government scheme PDFs using AWS Textract and vector embeddings
- **Conversational AI:** Natural language understanding for text and voice queries in multiple Indian languages
- **Personalized Recommendations:** Geo-based filtering and AI-powered eligibility prediction
- **Fraud Prevention:** Pattern detection and scheme verification to protect users
- **Multi-channel Delivery:** WhatsApp integration with voice output for maximum accessibility
- **Analytics Intelligence:** Real-time insights for government stakeholders

### Design Principles

1. **Voice-First:** Prioritize voice interaction for low-literacy users
2. **Simplicity:** Complex information distilled into actionable insights
3. **Accessibility:** Work on basic smartphones with low bandwidth
4. **Privacy:** User data protection and compliance with Indian regulations
5. **Scalability:** Cloud-native architecture for handling millions of users
6. **Accuracy:** High-quality AI responses with source attribution

---

## 2. High-Level Architecture

### Architecture Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   WhatsApp   │  │  Web Portal  │  │  Voice Interface     │  │
│  │   Business   │  │  (Dashboard) │  │  (Speech I/O)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│              (FastAPI / Flask REST API)                          │
│         Authentication | Rate Limiting | Routing                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │   Query    │  │    Geo     │  │   Eligibility Engine     │  │
│  │ Processor  │  │  Filter    │  │   (ML Prediction)        │  │
│  └────────────┘  └────────────┘  └──────────────────────────┘  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │   Fraud    │  │ Translation│  │   Response Generator     │  │
│  │ Detection  │  │  Service   │  │   (LLM Integration)      │  │
│  └────────────┘  └────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │  Document  │  │  Embedding │  │   Vector Search          │  │
│  │  Ingestion │  │  Generator │  │   (FAISS/Pinecone)       │  │
│  └────────────┘  └────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DATA & AI SERVICES                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │  MongoDB   │  │    LLM     │  │   AWS AI Services        │  │
│  │ (Profiles, │  │  (OpenAI/  │  │   (Textract, Translate,  │  │
│  │  Schemes)  │  │  Claude)   │  │   Polly)                 │  │
│  └────────────┘  └────────────┘  └──────────────────────────┘  │
│  ┌────────────┐  ┌────────────┐                                 │
│  │  AWS S3    │  │  Redis     │                                 │
│  │  (PDFs)    │  │  (Cache)   │                                 │
│  └────────────┘  └────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

#### Frontend Layer

**WhatsApp Interface**
- Primary user interface for rural citizens
- Handles text and voice message input/output
- Implements quick reply buttons and interactive menus
- Manages conversation state and context
- Delivers formatted scheme summaries with emojis and bullet points

**Government Dashboard (Web)**
- React-based responsive web application
- Real-time analytics visualization using Chart.js/D3.js
- Role-based access control for different government departments
- Export functionality for reports (PDF, Excel)
- Scheme management interface for administrators

**Voice Interface Module**
- Integrates with AWS Polly for text-to-speech
- Handles speech-to-text conversion for voice queries
- Optimizes audio compression for low bandwidth
- Supports 5 Indian languages with natural pronunciation

#### Backend API Layer

**FastAPI REST API**
- High-performance async API framework
- RESTful endpoints for all system operations
- WebSocket support for real-time dashboard updates
- API versioning for backward compatibility
- Comprehensive error handling and logging

**Key Endpoints:**
- `POST /api/v1/query` - Process user queries (text/voice)
- `GET /api/v1/schemes` - Retrieve schemes with filters
- `POST /api/v1/eligibility` - Check eligibility for schemes
- `POST /api/v1/profile` - Create/update user profile
- `GET /api/v1/analytics` - Fetch dashboard metrics
- `POST /api/v1/fraud/report` - Report fraud attempts

**Authentication & Security**
- JWT-based authentication for dashboard users
- API key authentication for WhatsApp webhook
- Rate limiting (100 requests/minute per user)
- Input validation and sanitization
- CORS configuration for web dashboard

#### RAG Pipeline

**Document Ingestion Service**
- Monitors S3 bucket for new PDF uploads
- Triggers AWS Textract for text extraction
- Handles multi-page and multi-column PDFs
- Extracts tables, forms, and structured data
- Stores raw text in MongoDB with metadata

**Text Processing & Chunking**
- Cleans extracted text (removes headers, footers, noise)
- Identifies scheme sections (eligibility, benefits, process)
- Chunks documents into 500-token segments with 50-token overlap
- Preserves context across chunks
- Tags chunks with metadata (scheme name, department, state)

**Embedding Generator**
- Uses OpenAI text-embedding-ada-002 or similar model
- Generates 1536-dimensional vectors for each chunk
- Batch processing for efficiency (100 chunks/batch)
- Caches embeddings to avoid recomputation
- Stores embeddings in vector database

**Vector Database (FAISS/Pinecone)**
- Stores scheme document embeddings
- Performs fast semantic similarity search (< 100ms)
- Supports filtering by metadata (state, district, category)
- Returns top-k relevant chunks (k=5 default)
- Handles 1000+ scheme documents efficiently

#### LLM Layer

**Query Understanding**
- Processes natural language queries
- Extracts intent (search, eligibility check, benefit inquiry)
- Identifies entities (scheme names, locations, demographics)
- Handles multilingual queries via translation
- Maintains conversation context for follow-ups

**Response Generation**
- Synthesizes information from retrieved chunks
- Generates simplified, personalized explanations
- Adapts language complexity to user profile
- Includes source attribution and confidence scores
- Formats output for WhatsApp delivery

**LLM Configuration**
- Model: GPT-4 or Claude-3 for high accuracy
- Temperature: 0.3 (balanced creativity and consistency)
- Max tokens: 500 (concise responses)
- System prompt: Optimized for government scheme explanation
- Fallback: GPT-3.5-turbo for cost optimization

#### Geo-Filtering Engine

**Location Detection**
- Extracts location from user profile
- Detects district/state from mobile number prefix
- Supports manual location selection
- Validates location against Indian administrative boundaries
- Caches location data for performance

**Scheme Filtering**
- Filters schemes by user's state and district
- Prioritizes local schemes over national schemes
- Considers scheme applicability rules
- Ranks schemes by relevance to user location
- Handles multi-state schemes appropriately

#### Fraud Detection Engine

**Pattern Analysis**
- Detects duplicate profiles (same phone, Aadhaar)
- Identifies suspicious profile data (impossible age, income)
- Monitors query patterns for bot-like behavior
- Flags rapid-fire queries from single source
- Tracks known fraud keywords and phrases

**Scheme Verification**
- Cross-references schemes with official government database
- Maintains whitelist of verified scheme names
- Flags schemes not found in official sources
- Warns users about common scam schemes
- Updates fraud database weekly

**Risk Scoring**
- Assigns risk score (0-100) to each query/profile
- Triggers manual review for high-risk cases (>80)
- Logs all fraud detection events
- Provides fraud alerts to government dashboard
- Implements machine learning for pattern improvement

#### Eligibility Prediction Engine

**Rule-Based Matching**
- Parses eligibility criteria from scheme documents
- Extracts rules (age range, income limit, occupation)
- Compares user profile against rules
- Calculates match percentage
- Identifies missing information

**ML-Based Prediction**
- Trains on historical beneficiary data (when available)
- Features: age, income, occupation, location, family size
- Model: Random Forest or XGBoost classifier
- Outputs: eligibility probability (0-1)
- Confidence score based on feature completeness

**Benefit Estimation**
- Extracts benefit formulas from scheme documents
- Applies formulas to user profile
- Estimates monetary benefits (subsidies, grants)
- Estimates non-monetary benefits (training, insurance)
- Provides range (min-max) when exact calculation not possible

#### Analytics Dashboard

**Data Collection**
- Logs all user queries with timestamps
- Tracks scheme views and eligibility checks
- Records user demographics and locations
- Monitors system performance metrics
- Captures fraud detection events

**Metrics Computation**
- Real-time aggregation using MongoDB aggregation pipeline
- Cached metrics updated every 5 minutes
- Historical trend analysis (daily, weekly, monthly)
- Comparative analytics (state-wise, scheme-wise)
- Predictive analytics for demand forecasting

**Visualization**
- Interactive charts (line, bar, pie, heat maps)
- Geographic visualization on India map
- Drill-down capabilities (national → state → district)
- Exportable reports in PDF and Excel formats
- Customizable dashboards per user role

---

## 3. End-to-End Data Flow


### User Query Flow (Step-by-Step)

**Step 1: Query Initiation**
- User sends text/voice message via WhatsApp
- WhatsApp webhook forwards message to API Gateway
- API Gateway authenticates request and routes to Query Processor
- System retrieves user profile from MongoDB (or creates new profile)

**Step 2: Input Processing**
- If voice input: AWS Transcribe converts speech to text
- Language detection identifies user's language
- If non-English: AWS Translate converts to English for processing
- Query Processor extracts intent and entities using LLM
- Fraud Detection Engine performs initial risk assessment

**Step 3: Geo-Filtering**
- System retrieves user's district and state from profile
- Geo-Filter Engine identifies applicable schemes for location
- Creates location-based filter for vector search
- Logs query with geographic metadata

**Step 4: RAG Retrieval**
- Query converted to embedding vector using same model as documents
- Vector database performs similarity search with geo-filter
- Returns top 5 most relevant scheme chunks with similarity scores
- Chunks include metadata (scheme name, department, source PDF)
- System logs retrieved schemes for analytics

**Step 5: Eligibility Prediction**
- For each retrieved scheme, Eligibility Engine analyzes criteria
- Compares user profile against eligibility rules
- ML model predicts eligibility probability
- Calculates potential benefit amount
- Ranks schemes by eligibility score × relevance score

**Step 6: Response Generation**
- LLM receives: user query + retrieved chunks + user profile + eligibility data
- System prompt instructs LLM to generate simple, personalized explanation
- LLM synthesizes information into structured response:
  - Top 3 relevant schemes
  - Simplified eligibility criteria
  - Estimated benefits
  - Application steps
  - Required documents
- Response formatted for WhatsApp (short paragraphs, bullets, emojis)

**Step 7: Translation & Voice Conversion**
- If user's language is not English: AWS Translate converts response
- If voice output requested: AWS Polly converts text to speech
- Audio compressed to <100KB for low bandwidth
- Response includes both text and audio options

**Step 8: Delivery**
- Response sent to WhatsApp Business API
- WhatsApp delivers message to user
- System logs response time and delivery status
- Conversation context saved for follow-up queries

**Step 9: Analytics Update**
- Query logged in analytics database
- Metrics updated: query count, scheme views, language usage
- Geographic heat map updated
- Dashboard refreshes with new data (if open)

**Step 10: Follow-up Handling**
- User can ask follow-up questions
- System maintains conversation context (last 5 messages)
- Follow-ups processed faster using cached scheme data
- User can bookmark schemes or request detailed information

---

## 4. RAG Architecture Details

### PDF Ingestion Pipeline

**Upload & Storage**
```
Government PDF → AWS S3 Bucket → Trigger Lambda Function
                                        ↓
                              Document Processing Queue
```

**Processing Steps:**
1. PDF uploaded to S3 bucket (manual or automated)
2. S3 event triggers Lambda function
3. Lambda adds document to processing queue (SQS)
4. Worker service picks up job from queue
5. Document metadata stored in MongoDB

**Metadata Schema:**
```json
{
  "document_id": "scheme_pmkisan_2024",
  "scheme_name": "PM-KISAN",
  "department": "Ministry of Agriculture",
  "state": "All India",
  "category": "Agriculture",
  "upload_date": "2024-01-15",
  "s3_path": "s3://schemes/pmkisan.pdf",
  "status": "processing",
  "page_count": 12
}
```

### Text Extraction

**AWS Textract Integration**
- Asynchronous API for multi-page PDFs
- Extracts text with layout preservation
- Identifies tables, forms, and key-value pairs
- Handles Hindi/English mixed documents
- Returns structured JSON with bounding boxes

**Post-Processing:**
- Remove headers, footers, page numbers
- Merge hyphenated words across lines
- Reconstruct tables into readable format
- Identify section headings (Eligibility, Benefits, etc.)
- Clean special characters and formatting artifacts

**Output Example:**
```json
{
  "document_id": "scheme_pmkisan_2024",
  "extracted_text": "PM-KISAN Scheme provides income support...",
  "sections": {
    "overview": "PM-KISAN is a Central Sector Scheme...",
    "eligibility": "All landholding farmers' families...",
    "benefits": "Financial benefit of Rs. 6000 per year...",
    "application": "Farmers can register through CSC..."
  },
  "tables": [...],
  "extraction_confidence": 0.95
}
```

### Chunking Strategy

**Semantic Chunking**
- Chunk size: 500 tokens (~375 words)
- Overlap: 50 tokens to preserve context
- Respect section boundaries (don't split mid-section)
- Each chunk includes section header for context
- Metadata attached to each chunk

**Chunk Example:**
```json
{
  "chunk_id": "pmkisan_chunk_003",
  "document_id": "scheme_pmkisan_2024",
  "text": "Eligibility Criteria: All landholding farmers...",
  "section": "eligibility",
  "token_count": 487,
  "chunk_index": 3,
  "metadata": {
    "scheme_name": "PM-KISAN",
    "state": "All India",
    "category": "Agriculture"
  }
}
```

### Embedding Generation

**Model Selection**
- Primary: OpenAI text-embedding-ada-002 (1536 dimensions)
- Alternative: Sentence-BERT multilingual model
- Batch size: 100 chunks per API call
- Cost optimization: Cache embeddings, avoid regeneration

**Embedding Process:**
```python
# Pseudocode
for batch in chunks:
    embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[chunk.text for chunk in batch]
    )
    store_embeddings(batch, embeddings)
```

**Storage:**
- Vector database: FAISS (local) or Pinecone (cloud)
- Index type: HNSW (Hierarchical Navigable Small World)
- Distance metric: Cosine similarity
- Metadata filtering: Enabled for geo-filtering

### Vector Storage & Indexing

**FAISS Configuration**
```python
# Index creation
dimension = 1536
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 16

# Add metadata filtering
index_with_metadata = faiss.IndexIDMap(index)
```

**Pinecone Configuration (Cloud Alternative)**
```python
pinecone.init(api_key="...", environment="us-west1-gcp")
index = pinecone.Index("government-schemes")

# Upsert with metadata
index.upsert(vectors=[
    (chunk_id, embedding, metadata)
])
```

### Retrieval Process

**Query Embedding**
```python
# Convert user query to embedding
query_embedding = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=user_query
)
```

**Semantic Search with Filters**
```python
# Search with geo-filtering
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "state": {"$in": [user_state, "All India"]},
        "district": {"$in": [user_district, "All"]}
    }
)
```

**Result Ranking**
```python
# Combine similarity score with eligibility prediction
for result in results:
    relevance_score = result.similarity_score
    eligibility_score = predict_eligibility(result.chunk, user_profile)
    final_score = 0.6 * relevance_score + 0.4 * eligibility_score
    result.final_score = final_score

ranked_results = sorted(results, key=lambda x: x.final_score, reverse=True)
```

### Response Synthesis

**Context Assembly**
```python
context = {
    "retrieved_chunks": top_5_chunks,
    "user_profile": user_profile,
    "eligibility_predictions": eligibility_scores,
    "conversation_history": last_3_messages
}
```

**LLM Prompt Template**
```
System: You are a helpful assistant explaining government schemes to rural citizens in India. Use simple language, avoid jargon, and be encouraging.

Context:
- User Profile: {age}, {occupation}, {location}
- Retrieved Schemes: {scheme_chunks}
- Eligibility Predictions: {predictions}

User Query: {user_query}

Instructions:
1. Explain top 3 relevant schemes in simple terms
2. For each scheme, mention: what it is, who can apply, benefits, how to apply
3. Use bullet points and short sentences
4. Include eligibility prediction with confidence
5. End with encouraging next steps

Response:
```

**Output Formatting**
```python
# Format for WhatsApp
response = format_whatsapp_message(
    llm_response,
    include_emojis=True,
    max_length=1000,
    add_quick_replies=["Tell me more", "Check eligibility", "How to apply"]
)
```

---

## 5. AI Models & Services Used


### Large Language Models (LLMs)

**Primary LLM: GPT-4 / Claude-3**
- **Purpose:** Query understanding, response generation, summarization
- **Use Cases:**
  - Converting complex scheme language to simple explanations
  - Extracting eligibility criteria from unstructured text
  - Generating personalized recommendations
  - Answering follow-up questions with context
- **Configuration:**
  - Temperature: 0.3 (consistent, factual responses)
  - Max tokens: 500 (concise outputs)
  - Top-p: 0.9 (balanced diversity)
- **Cost Optimization:**
  - Use GPT-3.5-turbo for simple queries
  - Cache common responses
  - Batch processing where possible
- **Fallback Strategy:** If primary LLM fails, fallback to GPT-3.5-turbo or local model

**Embedding Model: text-embedding-ada-002**
- **Purpose:** Convert text to vector embeddings for semantic search
- **Specifications:**
  - Dimensions: 1536
  - Context window: 8191 tokens
  - Cost: $0.0001 per 1K tokens
- **Usage:**
  - Embed scheme document chunks
  - Embed user queries for similarity search
  - Batch processing for efficiency

### AWS AI Services

**AWS Textract**
- **Purpose:** Extract text and structure from PDF documents
- **Features Used:**
  - Asynchronous document analysis for multi-page PDFs
  - Table extraction for benefit amount tables
  - Form extraction for application requirements
  - Layout analysis for section identification
- **API Calls:**
  - `StartDocumentAnalysis` for processing
  - `GetDocumentAnalysis` for retrieving results
- **Cost:** ~$1.50 per 1000 pages

**AWS Translate**
- **Purpose:** Multilingual translation for queries and responses
- **Supported Languages:**
  - Hindi (hi)
  - Tamil (ta)
  - Telugu (te)
  - Bengali (bn)
  - Marathi (mr)
- **Features:**
  - Neural machine translation for high accuracy
  - Custom terminology for government terms
  - Batch translation for efficiency
- **API Calls:**
  - `TranslateText` for real-time translation
  - Custom terminology for scheme names
- **Cost:** $15 per million characters

**AWS Polly**
- **Purpose:** Text-to-speech for voice output
- **Voices Used:**
  - Hindi: Aditi (female), Kajal (female)
  - Tamil: Not available (use Hindi with Tamil text)
  - English: Raveena (Indian accent)
- **Features:**
  - Neural TTS for natural-sounding speech
  - SSML support for pronunciation control
  - MP3 output compressed for low bandwidth
- **Configuration:**
  - Output format: MP3
  - Sample rate: 16 kHz (balance quality and size)
  - Speech rate: 90% (slightly slower for clarity)
- **Cost:** $4 per million characters (Neural)

**AWS Transcribe (Optional)**
- **Purpose:** Speech-to-text for voice queries
- **Features:**
  - Support for Hindi and English
  - Custom vocabulary for scheme names
  - Automatic language detection
- **Alternative:** Use WhatsApp's built-in voice-to-text

### Machine Learning Models

**Eligibility Prediction Model**
- **Algorithm:** Random Forest Classifier or XGBoost
- **Features (Input):**
  - User demographics: age, gender, occupation
  - Economic: income bracket, land ownership
  - Geographic: state, district, rural/urban
  - Family: family size, dependents
  - Scheme characteristics: category, target demographic
- **Target (Output):**
  - Binary: Eligible (1) or Not Eligible (0)
  - Probability: Confidence score (0-1)
- **Training Data:**
  - Historical beneficiary data (if available)
  - Synthetic data generated from eligibility rules
  - Continuous learning from user feedback
- **Performance Metrics:**
  - Accuracy: >85%
  - Precision: >80% (minimize false positives)
  - Recall: >90% (minimize false negatives)

**Fraud Detection Model**
- **Algorithm:** Isolation Forest (anomaly detection)
- **Features:**
  - Profile consistency score
  - Query pattern analysis
  - Time-based behavior patterns
  - Device fingerprinting
- **Output:**
  - Fraud risk score (0-100)
  - Anomaly indicators
- **Thresholds:**
  - Low risk: 0-30 (auto-approve)
  - Medium risk: 31-70 (monitor)
  - High risk: 71-100 (flag for review)

### Vector Database

**FAISS (Facebook AI Similarity Search)**
- **Purpose:** Fast similarity search for RAG retrieval
- **Index Type:** HNSW (Hierarchical Navigable Small World)
- **Configuration:**
  - Dimension: 1536 (matching embedding model)
  - M parameter: 32 (number of connections)
  - efConstruction: 40 (index build quality)
  - efSearch: 16 (search quality)
- **Performance:**
  - Search time: <100ms for 10K documents
  - Memory: ~6MB per 1000 documents
- **Deployment:** In-memory on API server

**Pinecone (Cloud Alternative)**
- **Purpose:** Managed vector database with metadata filtering
- **Features:**
  - Automatic scaling
  - Metadata filtering for geo-based search
  - Real-time updates
  - High availability
- **Configuration:**
  - Pod type: p1.x1 (starter)
  - Replicas: 1 (MVP), 3 (production)
  - Metric: Cosine similarity
- **Cost:** $70/month for 1M vectors

---

## 6. Database Design (High-Level Schema)

### MongoDB Collections

**users Collection**
```json
{
  "_id": "user_9876543210",
  "phone": "+919876543210",
  "name": "Ramesh Kumar",
  "age": 35,
  "gender": "male",
  "state": "Uttar Pradesh",
  "district": "Varanasi",
  "occupation": "farmer",
  "income_bracket": "below_2_lakh",
  "family_size": 5,
  "land_ownership": "yes",
  "land_size_acres": 2.5,
  "preferred_language": "hi",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-02-10T14:20:00Z",
  "consent_given": true
}
```

**schemes Collection**
```json
{
  "_id": "scheme_pmkisan_2024",
  "scheme_name": "PM-KISAN",
  "full_name": "Pradhan Mantri Kisan Samman Nidhi",
  "department": "Ministry of Agriculture & Farmers Welfare",
  "category": "agriculture",
  "state": "All India",
  "district": "All",
  "description": "Income support to all landholding farmers",
  "eligibility_criteria": {
    "occupation": ["farmer"],
    "land_ownership": "required",
    "income_limit": null,
    "age_range": [18, 100]
  },
  "benefits": {
    "type": "monetary",
    "amount": 6000,
    "frequency": "annual",
    "installments": 3
  },
  "application_process": "Online through PM-KISAN portal or CSC",
  "required_documents": ["Aadhaar", "Land records", "Bank account"],
  "deadline": null,
  "source_pdf": "s3://schemes/pmkisan.pdf",
  "last_updated": "2024-01-15T00:00:00Z",
  "status": "active"
}
```

**queries Collection**
```json
{
  "_id": "query_abc123",
  "user_id": "user_9876543210",
  "query_text": "What schemes are available for farmers?",
  "query_language": "en",
  "query_type": "voice",
  "timestamp": "2024-02-10T14:25:30Z",
  "user_location": {
    "state": "Uttar Pradesh",
    "district": "Varanasi"
  },
  "retrieved_schemes": [
    "scheme_pmkisan_2024",
    "scheme_pmfby_2024",
    "scheme_kcc_2024"
  ],
  "response_text": "Here are 3 schemes for farmers...",
  "response_language": "hi",
  "response_time_ms": 2340,
  "eligibility_predictions": {
    "scheme_pmkisan_2024": {"eligible": true, "confidence": 0.95},
    "scheme_pmfby_2024": {"eligible": true, "confidence": 0.87}
  },
  "fraud_score": 5,
  "user_feedback": null
}
```

**conversations Collection**
```json
{
  "_id": "conv_xyz789",
  "user_id": "user_9876543210",
  "started_at": "2024-02-10T14:25:00Z",
  "last_message_at": "2024-02-10T14:30:00Z",
  "messages": [
    {
      "role": "user",
      "content": "What schemes are available for farmers?",
      "timestamp": "2024-02-10T14:25:30Z"
    },
    {
      "role": "assistant",
      "content": "Here are 3 schemes...",
      "timestamp": "2024-02-10T14:25:33Z"
    }
  ],
  "context": {
    "current_schemes": ["scheme_pmkisan_2024"],
    "last_intent": "scheme_search"
  },
  "status": "active"
}
```

**fraud_alerts Collection**
```json
{
  "_id": "fraud_alert_001",
  "user_id": "user_1234567890",
  "alert_type": "duplicate_profile",
  "risk_score": 85,
  "details": {
    "reason": "Same Aadhaar used by multiple profiles",
    "duplicate_user_ids": ["user_0987654321"]
  },
  "timestamp": "2024-02-10T15:00:00Z",
  "status": "pending_review",
  "reviewed_by": null,
  "resolution": null
}
```

**analytics_daily Collection**
```json
{
  "_id": "analytics_2024_02_10",
  "date": "2024-02-10",
  "total_queries": 1523,
  "unique_users": 892,
  "queries_by_language": {
    "hi": 678,
    "en": 345,
    "ta": 234,
    "te": 156,
    "bn": 110
  },
  "queries_by_state": {
    "Uttar Pradesh": 345,
    "Tamil Nadu": 234,
    "Bihar": 189
  },
  "top_schemes": [
    {"scheme_id": "scheme_pmkisan_2024", "views": 456},
    {"scheme_id": "scheme_pmay_2024", "views": 234}
  ],
  "avg_response_time_ms": 2150,
  "fraud_alerts": 12,
  "eligibility_checks": 1234
}
```

### Redis Cache Schema

**User Profile Cache**
```
Key: user:{phone_number}
Value: JSON serialized user profile
TTL: 1 hour
```

**Scheme Cache**
```
Key: scheme:{scheme_id}
Value: JSON serialized scheme details
TTL: 24 hours
```

**Query Response Cache**
```
Key: query:{hash(query_text + user_location)}
Value: JSON serialized response
TTL: 1 hour
```

**Analytics Cache**
```
Key: analytics:dashboard:{date}
Value: JSON serialized metrics
TTL: 5 minutes
```

### Database Indexes

**users Collection Indexes**
- `phone` (unique)
- `state, district` (compound, for geo-filtering)
- `created_at` (for analytics)

**schemes Collection Indexes**
- `scheme_name` (text search)
- `state, district` (compound, for geo-filtering)
- `category` (for filtering)
- `status` (for active schemes)

**queries Collection Indexes**
- `user_id, timestamp` (compound, for user history)
- `timestamp` (for analytics)
- `user_location.state` (for geographic analytics)

---

## 7. Security & Privacy Architecture

