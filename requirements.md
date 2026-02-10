# Requirements Document

## Project: InfinityCircle
**Team:** InfinityCircle  
**Version:** 1.0  
**Date:** February 2026

---

## 1. Introduction

### Problem Statement

Rural citizens across India face significant barriers in accessing and understanding government welfare schemes due to:
- Complex PDF documents written in technical bureaucratic language
- Language barriers (schemes often available only in English/Hindi)
- Low digital literacy and limited smartphone capabilities
- Lack of personalized guidance on eligibility and benefits
- Difficulty in navigating multiple schemes across different departments
- Prevalence of fraud and misinformation about government schemes

These barriers result in low scheme adoption rates, exclusion of eligible beneficiaries, and reduced impact of government welfare initiatives in rural areas.

### Vision

To democratize access to government schemes by creating an AI-powered, voice-first platform that breaks down language and complexity barriers, enabling every rural citizen to discover, understand, and apply for schemes they are eligible for.

### Objectives

1. Simplify complex government scheme PDFs into easy-to-understand explanations
2. Provide multilingual support for 5+ Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi)
3. Enable voice-based interaction for low-literacy users
4. Deliver personalized scheme recommendations based on user profile and location
5. Predict eligibility and potential benefits for each scheme
6. Detect and prevent fraud through AI-powered verification
7. Provide WhatsApp-based delivery for maximum reach
8. Offer government agencies real-time analytics on scheme awareness and adoption

---

## 2. Stakeholders

### Primary Stakeholders

**Rural Citizens**
- Role: End users seeking information about government schemes
- Needs: Simple language, voice support, personalized recommendations
- Access: WhatsApp, basic smartphones, low bandwidth

**Government Departments**
- Role: Scheme administrators and policy makers
- Needs: Analytics on reach, adoption rates, citizen queries, fraud detection
- Access: Web-based analytics dashboard

### Secondary Stakeholders

**Field Workers / Gram Sevaks**
- Role: Assist citizens in scheme discovery and application
- Needs: Quick access to scheme information, eligibility checking tools

**NGOs and CSOs**
- Role: Facilitate scheme awareness in rural communities
- Needs: Bulk query capabilities, community-level analytics

---

## 3. Functional Requirements

### 3.1 User Profile & Geo Detection

**FR-1.1:** System shall capture basic user profile information including:
- Name, age, gender
- District and state (mandatory)
- Occupation category (farmer, laborer, student, etc.)
- Income bracket (optional)
- Family size and composition

**FR-1.2:** System shall automatically detect user location based on:
- Mobile number prefix (district-level)
- GPS coordinates (if available)
- Manual selection by user

**FR-1.3:** System shall maintain user profiles in database for personalized recommendations

**FR-1.4:** System shall support profile updates and corrections

### 3.2 Conversational Query System

**FR-2.1:** System shall accept queries in multiple formats:
- Text input (English and 5 Indian languages)
- Voice input (speech-to-text conversion)
- Predefined quick queries ("Show me farmer schemes")

**FR-2.2:** System shall understand natural language queries such as:
- "What schemes are available for farmers in my area?"
- "Am I eligible for PM-KISAN?"
- "How much money can I get from housing scheme?"

**FR-2.3:** System shall maintain conversation context for follow-up questions

**FR-2.4:** System shall handle ambiguous queries by asking clarifying questions

**FR-2.5:** System shall support query history and bookmarking

### 3.3 RAG-Based Scheme Retrieval

**FR-3.1:** System shall ingest government scheme PDFs from multiple sources:
- Central government schemes
- State-specific schemes
- District-level schemes

**FR-3.2:** System shall extract and structure information including:
- Scheme name and department
- Eligibility criteria
- Benefits and subsidies
- Application process
- Required documents
- Deadlines and timelines

**FR-3.3:** System shall create vector embeddings of scheme documents for semantic search

**FR-3.4:** System shall retrieve top 3-5 most relevant schemes based on user query

**FR-3.5:** System shall filter schemes based on user's geo-location (district/state)

**FR-3.6:** System shall rank schemes by relevance score and predicted eligibility

### 3.4 Smart Summarization Engine

**FR-4.1:** System shall generate simplified summaries of complex scheme documents

**FR-4.2:** Summaries shall include:
- One-line scheme description
- Who can apply (eligibility in simple terms)
- What benefits you get (monetary and non-monetary)
- How to apply (step-by-step)
- Documents needed
- Important deadlines

**FR-4.3:** System shall adapt language complexity based on user literacy level

**FR-4.4:** System shall use local examples and context for better understanding

**FR-4.5:** System shall highlight key numbers (subsidy amounts, deadlines) prominently

### 3.5 Eligibility & Benefit Prediction

**FR-5.1:** System shall analyze user profile against scheme eligibility criteria

**FR-5.2:** System shall provide eligibility prediction with confidence score:
- Eligible (>80% confidence)
- Likely Eligible (50-80% confidence)
- Not Eligible (<50% confidence)

**FR-5.3:** System shall explain why user is/isn't eligible for each scheme

**FR-5.4:** System shall predict potential benefit amount based on:
- User profile parameters
- Historical data from similar beneficiaries
- Scheme calculation formulas

**FR-5.5:** System shall suggest missing information needed to improve eligibility prediction

### 3.6 Fraud Detection Module

**FR-6.1:** System shall detect potential fraud patterns:
- Duplicate applications from same user
- Suspicious profile information (age, income inconsistencies)
- Unusual query patterns indicating bot activity
- Known fraud keywords and phrases

**FR-6.2:** System shall verify scheme authenticity:
- Cross-reference with official government databases
- Flag schemes not found in official sources
- Warn users about common scams

**FR-6.3:** System shall provide fraud reporting mechanism for users

**FR-6.4:** System shall maintain fraud alert database updated from government sources

**FR-6.5:** System shall educate users about common fraud tactics

### 3.7 Voice Interaction Module

**FR-7.1:** System shall convert voice input to text using speech recognition

**FR-7.2:** System shall support voice input in 5 Indian languages

**FR-7.3:** System shall handle noisy environments and accented speech

**FR-7.4:** System shall convert text responses to natural-sounding voice output

**FR-7.5:** System shall allow users to replay voice responses

**FR-7.6:** System shall optimize audio quality for low-bandwidth networks

### 3.8 WhatsApp Integration

**FR-8.1:** System shall integrate with WhatsApp Business API

**FR-8.2:** System shall support text and voice message queries via WhatsApp

**FR-8.3:** System shall deliver responses in WhatsApp-friendly format:
- Short paragraphs
- Bullet points
- Emojis for visual clarity
- Quick reply buttons

**FR-8.4:** System shall send scheme summaries as formatted messages

**FR-8.5:** System shall support sharing scheme information with others

**FR-8.6:** System shall send proactive notifications:
- New schemes launched
- Application deadlines approaching
- Scheme updates relevant to user

**FR-8.7:** System shall handle high message volumes during peak times

### 3.9 Government Analytics Dashboard

**FR-9.1:** Dashboard shall display real-time metrics:
- Total queries received (daily/weekly/monthly)
- Most searched schemes
- Geographic distribution of queries
- Language preference distribution
- Eligibility prediction statistics

**FR-9.2:** Dashboard shall provide scheme-specific analytics:
- Query volume per scheme
- Average eligibility rate
- Common user questions
- Drop-off points in application process

**FR-9.3:** Dashboard shall identify gaps and opportunities:
- Underserved districts/demographics
- Schemes with low awareness
- Common eligibility barriers
- Fraud attempt patterns

**FR-9.4:** Dashboard shall support data export for reporting

**FR-9.5:** Dashboard shall provide role-based access control for government users

**FR-9.6:** Dashboard shall visualize trends over time with charts and graphs

---

## 4. Non-Functional Requirements

### 4.1 Scalability

**NFR-1.1:** System shall support 10,000+ concurrent users during MVP phase

**NFR-1.2:** System architecture shall be horizontally scalable to handle 1M+ users

**NFR-1.3:** Vector database shall efficiently handle 1000+ scheme documents

**NFR-1.4:** System shall process queries with auto-scaling based on load

### 4.2 Performance

**NFR-2.1:** System shall respond to text queries within 3 seconds (90th percentile)

**NFR-2.2:** System shall respond to voice queries within 5 seconds (90th percentile)

**NFR-2.3:** RAG retrieval shall complete within 1 second for semantic search

**NFR-2.4:** WhatsApp message delivery shall occur within 2 seconds of response generation

**NFR-2.5:** Dashboard shall load within 2 seconds with cached data

### 4.3 Security

**NFR-3.1:** System shall encrypt all data in transit using TLS 1.3

**NFR-3.2:** System shall encrypt sensitive user data at rest

**NFR-3.3:** System shall implement API authentication and rate limiting

**NFR-3.4:** System shall sanitize all user inputs to prevent injection attacks

**NFR-3.5:** System shall implement role-based access control for admin functions

**NFR-3.6:** System shall maintain audit logs for all data access and modifications

### 4.4 Data Privacy

**NFR-4.1:** System shall comply with Indian data protection regulations

**NFR-4.2:** System shall obtain explicit user consent for data collection

**NFR-4.3:** System shall anonymize data used for analytics

**NFR-4.4:** System shall provide users ability to delete their data

**NFR-4.5:** System shall not share user data with third parties without consent

**NFR-4.6:** System shall store data within Indian geographic boundaries

### 4.5 Accessibility

**NFR-5.1:** System shall be usable by users with low digital literacy

**NFR-5.2:** System shall support voice-only interaction (no reading required)

**NFR-5.3:** System shall work on basic smartphones (Android 8+)

**NFR-5.4:** System shall provide clear error messages in user's language

**NFR-5.5:** System shall use simple, jargon-free language in all communications

### 4.6 Multilingual Support

**NFR-6.1:** System shall support 5 Indian languages: Hindi, Tamil, Telugu, Bengali, Marathi

**NFR-6.2:** System shall maintain translation accuracy >90% for scheme information

**NFR-6.3:** System shall preserve meaning and context during translation

**NFR-6.4:** System shall handle code-mixed queries (e.g., Hindi-English)

**NFR-6.5:** System shall allow users to switch languages mid-conversation

### 4.7 Low-Bandwidth Optimization

**NFR-7.1:** System shall function on 2G/3G networks

**NFR-7.2:** Voice messages shall be compressed to <100KB per minute

**NFR-7.3:** System shall implement progressive loading for dashboard

**NFR-7.4:** System shall cache frequently accessed scheme data

**NFR-7.5:** System shall provide text-only mode for extremely low bandwidth

---

## 5. Constraints & Assumptions

### Constraints

**Technical Constraints:**
- MVP development timeline: 36 hours (hackathon duration)
- Limited access to official government APIs during hackathon
- Budget constraints for cloud services and LLM API calls
- WhatsApp Business API approval process may take time

**Data Constraints:**
- Limited availability of structured scheme data
- PDF quality varies across government departments
- No real-time integration with government beneficiary databases
- Historical beneficiary data not publicly available

**Resource Constraints:**
- Small team size (4-6 members)
- Limited testing with actual rural users during hackathon
- No access to production WhatsApp Business account during MVP

### Assumptions

**User Assumptions:**
- Target users have access to WhatsApp (90%+ smartphone penetration in rural India)
- Users have basic familiarity with voice messaging
- Users trust AI-generated information if clearly sourced
- Users prefer voice over text for complex information

**Technical Assumptions:**
- Government scheme PDFs are available publicly online
- LLM APIs (OpenAI/Claude) provide sufficient accuracy for scheme understanding
- Vector databases can handle semantic search at required scale
- AWS services (Textract, Translate, Polly) are reliable and cost-effective

**Business Assumptions:**
- Government departments will be interested in adoption analytics
- NGOs and field workers will help onboard rural users
- Fraud detection will improve user trust and adoption
- Multilingual support will significantly increase reach

---

## 6. Future Scope

### Phase 2 Enhancements (Post-Hackathon)

**Advanced Features:**
- Integration with government beneficiary databases for real-time eligibility verification
- AI-powered application form filling assistance
- Document upload and verification (Aadhaar, income certificate, etc.)
- Application status tracking across multiple schemes
- Peer support community for scheme beneficiaries

**Expanded Coverage:**
- Support for 15+ Indian languages including regional dialects
- Coverage of 500+ central and state schemes
- Integration with local government schemes (panchayat level)
- Sector-specific modules (agriculture, education, health, housing)

**Enhanced AI Capabilities:**
- Personalized scheme recommendations using collaborative filtering
- Predictive analytics for scheme success probability
- Chatbot personality customization based on user preferences
- Multi-modal input (image-based queries, document scanning)

**Platform Expansion:**
- Mobile app (Android/iOS) for richer user experience
- IVRS (Interactive Voice Response System) for feature phone users
- SMS-based fallback for non-WhatsApp users
- Integration with India Stack (Aadhaar, DigiLocker, UPI)

**Government Integration:**
- Direct application submission to government portals
- Real-time scheme updates from government APIs
- Beneficiary feedback loop to government departments
- Integration with PM-JAY, PM-KISAN, and other major schemes

**Analytics & Insights:**
- Predictive modeling for scheme demand forecasting
- Sentiment analysis of citizen feedback
- Geographic heat maps for scheme awareness gaps
- AI-powered policy recommendations based on citizen needs

### Long-term Vision

- Become the primary digital interface between rural citizens and government schemes
- Expand to other government services (certificates, grievances, payments)
- Partner with state governments for official deployment
- Scale to 100M+ rural users across India
- Reduce scheme adoption time from months to days
- Achieve 80%+ accuracy in eligibility prediction
- Prevent 90%+ of fraud attempts through AI detection

---

## Appendix

### Glossary

- **RAG:** Retrieval-Augmented Generation - AI technique combining document retrieval with language generation
- **Vector Database:** Database optimized for storing and searching high-dimensional embeddings
- **LLM:** Large Language Model - AI model trained on vast text data for natural language understanding
- **Geo-filtering:** Filtering content based on geographic location
- **Semantic Search:** Search based on meaning rather than exact keyword matching

### References

- Government of India Scheme Portal: https://www.india.gov.in
- Digital India Initiative: https://digitalindia.gov.in
- WhatsApp Business API Documentation
- AWS AI Services Documentation
- Smart India Hackathon Guidelines 2026
