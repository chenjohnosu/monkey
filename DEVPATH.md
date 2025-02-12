
## 0.6 Code Clean Up & Redevelopement

The current monkey implementation was part of the learning process for developing with LLMS.  Learning more about LLM, RAG, and Topic Modeling, I realized what I have is not very elegant and a brute force application of what I intended, so I will seek to refine the code base as well as add new functions/features.

1. Keep current capabilities as "RAG" general capability.
2. Improve removal of stop words and improving creating of vector databases.  
3. Implement gensim to more sophisticated theme and topic detection rather than rely simply on RAG/LLM Output.
4. Uncouple multiple steps in the modes: create vector database, merge two (or more) vector databases, do better job at providing GUIDE options, dedicated analysis of quantitative data
5. Provide more granular control of embedding, context window
6. Simplify usage
7. Simplify tool chain and LLM/RAG tools
8. Macro: if command issued with a directory; chain creating vector database and provide topic summary
9. Be able to list/locate monkey databases with a .monkey meta file.

### New Architecture

A. Create/Merge Vector Database(s)

B. Retrieval (Modes)

1. Topic Modeling (Unstructured)
2. RAG - Qualitative
3. RAG - Quantitative (pandasai)
4. GENSIM - Qualitative

C. Generate Answer (LLM)

## 0.7 Scale Enable for HPC clusters for very large models


