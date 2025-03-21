The difference between `/run themes` and `/run topic` lies in their analytical approaches and focus areas, despite some overlapping terminology:

### `/run themes [all|nfm|net|key|lsa|cluster]`

This focuses on **thematic analysis**, which identifies meaningful patterns and concepts across documents through various techniques:

- **nfm**: Named entity-focused method that extracts named entities and relationships between them
- **net**: Content network analysis that maps document relationships based on content similarity
- **key**: Keyword extraction that identifies significant terms across the corpus
- **lsa**: Latent Semantic Analysis that finds hidden semantic structures 
- **cluster**: Document clustering to identify natural document groupings

Thematic analysis is more concerned with content semantics, entity relationships, and extracting meaningful narrative threads. Your implementation emphasizes documents that "talk about the same things."

### `/run topic [all|lda|nmf|cluster]`

This focuses on **topic modeling**, which uses statistical models to discover abstract "topics" that occur in documents:

- **lda**: Latent Dirichlet Allocation, a generative statistical model that assumes documents are mixtures of topics
- **nmf**: Non-negative Matrix Factorization, which finds latent features through matrix decomposition
- **cluster**: Clustering-based topic identification (similar approach but with different implementation)

Topic modeling is more concerned with the statistical distribution of terms and discovering underlying topic structures. It views documents as probability distributions over topics.

### Key Differences:

1. **Analytical Focus**:
   - Themes: More focused on semantics, relationships, and narrative elements
   - Topics: More focused on statistical distributions and latent structures

2. **Implementation**:
   - Themes: Uses multiple complementary techniques including network analysis and entity extraction
   - Topics: Primarily uses statistical modeling techniques from the field of machine learning

3. **Results Interpretation**:
   - Themes: Often more directly interpretable as they map to named entities and explicit concepts
   - Topics: Can be more abstract, representing statistical co-occurrence patterns

These commands complement each other rather than replace one another. For comprehensive document analysis, you might want to run both to get different perspectives on your document collection.