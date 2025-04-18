"""
Enhanced diagnostic tools for theme analysis
"""

import os
import datetime
from collections import Counter, defaultdict
from core.engine.utils import ensure_dir
from core.engine.logging import debug


def enhance_theme_analysis_with_diagnostics(self, workspace, method='all', original_results=None):
    """
    Analyze themes in a workspace with enhanced diagnostics output

    Args:
        workspace (str): The workspace to analyze
        method (str): Analysis method ('all', 'nfm', 'net', 'key', 'lsa', 'cluster')

    Returns:
        Dict: Analysis results with enhanced diagnostics
    """
    debug(self.config, f"Analyzing themes in workspace '{workspace}' with enhanced diagnostics")

    # Make a copy of the original analyze method
    original_analyze = self.analyze

    # Run the original analysis to get the base results
    results = original_results if original_results is not None else original_analyze(workspace, method)

    # Generate enhanced diagnostics based on the method
    diagnostic_results = {}

    if method in ['all', 'nfm']:
        diagnostic_results['nfm'] = self._generate_entity_diagnostics(results.get('nfm', {}), workspace)

    if method in ['all', 'net']:
        diagnostic_results['net'] = self._generate_network_diagnostics(results.get('net', {}), workspace)

    if method in ['all', 'key']:
        diagnostic_results['key'] = self._generate_keyword_diagnostics(results.get('key', {}), workspace)

    # Output the enhanced diagnostics
    self._output_enhanced_diagnostics(workspace, diagnostic_results, method)

    # Return the original results plus diagnostics
    return {
        'original_results': results,
        'diagnostics': diagnostic_results
    }


def _generate_entity_diagnostics(self, entity_results, workspace):
    """
    Generate enhanced diagnostics for named entity analysis

    Args:
        entity_results (Dict): Named entity analysis results
        workspace (str): Workspace being analyzed

    Returns:
        Dict: Enhanced diagnostics information
    """
    print("\n=== ENHANCED ENTITY ANALYSIS DIAGNOSTICS ===")

    # Initialize diagnostics
    diagnostics = {
        'summary': {},
        'entity_distribution': {},
        'theme_analysis': {},
        'interpretation': {},
        'visualization_data': {}
    }

    # Check if we have valid results
    if not entity_results or 'themes' not in entity_results:
        print("No valid entity analysis results to diagnose")
        return diagnostics

    # 1. Generate summary statistics
    themes = entity_results.get('themes', [])
    entity_count = entity_results.get('entity_count', 0)
    significant_entities = entity_results.get('significant_entities', 0)

    # Basic summary
    print(f"Total entities extracted: {entity_count}")
    print(
        f"Significant entities identified: {significant_entities} ({(significant_entities / entity_count * 100):.1f}% of total)")
    print(f"Themes generated: {len(themes)}")

    # Store in diagnostic summary
    diagnostics['summary'] = {
        'total_entities': entity_count,
        'significant_entities': significant_entities,
        'significant_percentage': round(significant_entities / entity_count * 100 if entity_count else 0, 1),
        'theme_count': len(themes),
        'average_theme_size': sum(t.get('entity_count', 0) for t in themes) / len(themes) if themes else 0,
        'average_theme_docs': sum(t.get('document_count', 0) for t in themes) / len(themes) if themes else 0
    }

    # 2. Analyze theme distribution
    print("\nTheme Distribution Analysis:")

    # Calculate theme size distribution
    theme_sizes = [t.get('entity_count', 0) for t in themes]
    theme_docs = [t.get('document_count', 0) for t in themes]
    theme_frequencies = [t.get('frequency', 0) for t in themes]

    # Simple statistics
    if theme_sizes:
        print(f"  Theme size range: {min(theme_sizes)} to {max(theme_sizes)} entities per theme")
        print(f"  Theme document coverage: {min(theme_docs)} to {max(theme_docs)} documents per theme")

        # Group themes by size buckets
        size_buckets = {
            'small (2-5 entities)': len([s for s in theme_sizes if 2 <= s <= 5]),
            'medium (6-15 entities)': len([s for s in theme_sizes if 6 <= s <= 15]),
            'large (16+ entities)': len([s for s in theme_sizes if s >= 16])
        }

        print("  Theme Size Distribution:")
        for bucket, count in size_buckets.items():
            print(f"    {bucket}: {count} themes ({count / len(themes) * 100:.1f}%)")

        # Store in diagnostics
        diagnostics['entity_distribution'] = {
            'theme_size_range': [min(theme_sizes), max(theme_sizes)],
            'theme_doc_range': [min(theme_docs), max(theme_docs)],
            'size_buckets': size_buckets,
            'avg_frequency': sum(theme_frequencies) / len(theme_frequencies)
        }

    # 3. Analyze theme content
    print("\nTheme Content Analysis:")

    # Collect all keywords across themes for overlap analysis
    all_keywords = []
    for theme in themes:
        all_keywords.extend(theme.get('keywords', []))

    # Count keyword occurrences
    from collections import Counter
    keyword_counts = Counter(all_keywords)

    # Find overlapping keywords (appearing in multiple themes)
    overlapping = [k for k, c in keyword_counts.items() if c > 1]
    overlap_percentage = len(overlapping) / len(keyword_counts) * 100 if keyword_counts else 0

    print(f"  Unique keywords across all themes: {len(keyword_counts)}")
    print(f"  Keywords appearing in multiple themes: {len(overlapping)} ({overlap_percentage:.1f}%)")

    # Top overlapping keywords
    if overlapping:
        top_overlapping = sorted([(k, c) for k, c in keyword_counts.items() if c > 1],
                                 key=lambda x: x[1], reverse=True)[:5]
        print("  Most overlapping keywords:")
        for keyword, count in top_overlapping:
            print(f"    '{keyword}' appears in {count} themes")

    # Store in diagnostics
    diagnostics['theme_analysis'] = {
        'unique_keywords': len(keyword_counts),
        'overlapping_keywords': len(overlapping),
        'overlap_percentage': round(overlap_percentage, 1),
        'top_overlapping': dict(sorted([(k, c) for k, c in keyword_counts.items() if c > 1],
                                       key=lambda x: x[1], reverse=True)[:10])
    }

    # 4. Generate interpretation
    print("\nInterpretation Guidance:")

    # Theme coherence assessment
    avg_overlap = overlap_percentage
    if avg_overlap < 10:
        coherence = "high"
        print("  Theme coherence appears HIGH - themes are well-separated with minimal keyword overlap")
    elif avg_overlap < 25:
        coherence = "moderate"
        print("  Theme coherence appears MODERATE - some keyword overlap between themes")
    else:
        coherence = "low"
        print("  Theme coherence appears LOW - significant keyword overlap between themes")

    # Theme coverage assessment
    total_docs = max(sum(t.get('document_count', 0) for t in themes), 1)
    if total_docs > 0 and len(themes) > 0:
        coverage_ratio = (sum(t.get('document_count', 0) for t in themes[:3]) / total_docs) * 100

        if coverage_ratio > 70:
            coverage = "concentrated"
            print(f"  Theme coverage is CONCENTRATED - top 3 themes cover {coverage_ratio:.1f}% of documents")
        elif coverage_ratio > 40:
            coverage = "balanced"
            print(f"  Theme coverage is BALANCED - top 3 themes cover {coverage_ratio:.1f}% of documents")
        else:
            coverage = "distributed"
            print(f"  Theme coverage is DISTRIBUTED - top 3 themes cover only {coverage_ratio:.1f}% of documents")
    else:
        coverage = "unknown"
        coverage_ratio = 0

    # Store interpretation
    diagnostics['interpretation'] = {
        'coherence': coherence,
        'coverage': coverage,
        'coverage_ratio': round(coverage_ratio, 1),
        'analysis_quality': "high" if coherence == "high" and len(themes) >= 3 else
        "moderate" if coherence == "moderate" or len(themes) >= 2 else "low"
    }

    # 5. Generate visualization data
    print("\nVisualization Data:")

    # Theme network visualization data
    theme_nodes = []
    theme_edges = []

    # Create nodes for each theme
    for i, theme in enumerate(themes):
        theme_nodes.append({
            'id': f"theme_{i}",
            'label': theme.get('name', f"Theme {i + 1}").replace("Theme: ", ""),
            'size': theme.get('document_count', 1),
            'frequency': theme.get('frequency', 0)
        })

    # Create edges between themes that share keywords
    theme_keyword_map = {}
    for i, theme in enumerate(themes):
        theme_keywords = set(theme.get('keywords', []))
        theme_keyword_map[i] = theme_keywords

    # Compare themes for shared keywords
    for i in range(len(themes)):
        for j in range(i + 1, len(themes)):
            shared_keywords = theme_keyword_map[i].intersection(theme_keyword_map[j])
            if shared_keywords:  # If they share any keywords, create an edge
                theme_edges.append({
                    'source': f"theme_{i}",
                    'target': f"theme_{j}",
                    'weight': len(shared_keywords),
                    'shared': list(shared_keywords)
                })

    print(f"  Generated visualization data for {len(theme_nodes)} themes with {len(theme_edges)} connections")

    # Store visualization data
    diagnostics['visualization_data'] = {
        'theme_nodes': theme_nodes,
        'theme_edges': theme_edges
    }

    # 6. Prepare suggestions for refining analysis (optional)
    if coherence == "low":
        print("\nSuggestions for improving analysis:")
        print("  - Consider increasing entity filtering to reduce noise")
        print("  - Try adjusting thresholds for significance to create more distinct themes")

    return diagnostics


def _generate_network_diagnostics(self, network_results, workspace):
    """
    Generate enhanced diagnostics for content network analysis

    Args:
        network_results (Dict): Content network analysis results
        workspace (str): Workspace being analyzed

    Returns:
        Dict: Enhanced diagnostics information
    """
    print("\n=== ENHANCED CONTENT NETWORK DIAGNOSTICS ===")

    # Initialize diagnostics
    diagnostics = {
        'network_stats': {},
        'community_analysis': {},
        'centrality_metrics': {},
        'document_coverage': {},
        'interpretation': {},
        'visualization_data': {}
    }

    # Check if we have valid results
    if not network_results or 'themes' not in network_results:
        print("No valid network analysis results to diagnose")
        return diagnostics

    # Get themes from results
    themes = network_results.get('themes', [])

    # 1. Network Statistics
    print("Network Structure Analysis:")

    # Count total documents and nodes
    total_docs = sum(len(theme.get('nodes', [])) for theme in themes)
    total_communities = len(themes)

    print(f"  Total documents in network: {total_docs}")
    print(f"  Total communities detected: {total_communities}")

    # Calculate community size distribution
    community_sizes = [len(theme.get('nodes', [])) for theme in themes]
    if community_sizes:
        print(f"  Community size range: {min(community_sizes)} to {max(community_sizes)} documents")
        print(f"  Average community size: {sum(community_sizes) / len(community_sizes):.1f} documents")

    # Store in diagnostics
    diagnostics['network_stats'] = {
        'total_documents': total_docs,
        'total_communities': total_communities,
        'community_size_range': [min(community_sizes), max(community_sizes)] if community_sizes else [0, 0],
        'avg_community_size': sum(community_sizes) / len(community_sizes) if community_sizes else 0,
        'community_size_distribution': {
            'small (1-3 docs)': len([s for s in community_sizes if 1 <= s <= 3]),
            'medium (4-10 docs)': len([s for s in community_sizes if 4 <= s <= 10]),
            'large (11+ docs)': len([s for s in community_sizes if s >= 11])
        }
    }

    # 2. Centrality Analysis
    print("\nCentrality Analysis:")

    # Extract centrality values
    centrality_values = [theme.get('centrality', 0) for theme in themes]
    if centrality_values:
        print(f"  Centrality score range: {min(centrality_values):.2f} to {max(centrality_values):.2f}")
        print(f"  Average centrality: {sum(centrality_values) / len(centrality_values):.2f}")

        # Identify central vs peripheral communities
        central_threshold = 0.7
        central_communities = len([c for c in centrality_values if c >= central_threshold])
        print(f"  Highly central communities (score ≥ {central_threshold}): {central_communities}")
        print(f"  Peripheral communities (score < {central_threshold}): {len(centrality_values) - central_communities}")

        # Store in diagnostics
        diagnostics['centrality_metrics'] = {
            'centrality_range': [min(centrality_values), max(centrality_values)],
            'avg_centrality': sum(centrality_values) / len(centrality_values),
            'central_communities': central_communities,
            'peripheral_communities': len(centrality_values) - central_communities
        }

    # 3. Keyword Analysis
    print("\nKeyword Analysis:")

    # Collect keywords from all themes
    all_keywords = []
    for theme in themes:
        all_keywords.extend(theme.get('keywords', []))

    # Count keyword frequencies
    from collections import Counter
    keyword_counts = Counter(all_keywords)

    print(f"  Unique keywords across communities: {len(keyword_counts)}")

    # Top keywords
    if keyword_counts:
        top_keywords = keyword_counts.most_common(5)
        print("  Most frequent keywords:")
        for keyword, count in top_keywords:
            print(f"    '{keyword}' appears in {count} communities")

    # Store in diagnostics
    diagnostics['community_analysis'] = {
        'unique_keywords': len(keyword_counts),
        'top_keywords': dict(keyword_counts.most_common(10))
    }

    # 4. Document Coverage
    print("\nDocument Coverage Analysis:")

    # Analyze theme overlap for documents
    doc_themes = {}  # Maps documents to the themes they appear in

    for i, theme in enumerate(themes):
        for doc in theme.get('nodes', []):
            if doc not in doc_themes:
                doc_themes[doc] = []
            doc_themes[doc].append(i)

    # Count documents by number of themes they appear in
    theme_appearances = Counter([len(themes) for doc, themes in doc_themes.items()])

    # Calculate unique document percentage
    unique_docs = theme_appearances.get(1, 0)
    overlapping_docs = sum(count for themes, count in theme_appearances.items() if themes > 1)
    unique_percentage = (unique_docs / total_docs * 100) if total_docs > 0 else 0

    print(f"  Documents appearing in exactly one community: {unique_docs} ({unique_percentage:.1f}%)")
    print(f"  Documents appearing in multiple communities: {overlapping_docs} ({100 - unique_percentage:.1f}%)")

    # Document distribution
    if theme_appearances:
        print("  Document distribution by community count:")
        for theme_count, doc_count in sorted(theme_appearances.items()):
            print(f"    In {theme_count} communities: {doc_count} documents")

    # Store in diagnostics
    diagnostics['document_coverage'] = {
        'unique_docs': unique_docs,
        'unique_percentage': round(unique_percentage, 1),
        'overlapping_docs': overlapping_docs,
        'theme_appearances': dict(theme_appearances)
    }

    # 5. Generate interpretation
    print("\nInterpretation Guidance:")

    # Network structure interpretation
    if total_communities == 1:
        structure = "monolithic"
        print("  Network structure is MONOLITHIC - documents form a single cohesive group")
    elif unique_percentage > 80:
        structure = "segregated"
        print("  Network structure is SEGREGATED - documents form distinct, separate communities")
    elif unique_percentage > 50:
        structure = "clustered"
        print("  Network structure is CLUSTERED - documents form distinguishable communities with some overlap")
    else:
        structure = "interconnected"
        print("  Network structure is INTERCONNECTED - significant overlap between document communities")

    # Centrality interpretation
    if 'centrality_metrics' in diagnostics:
        avg_centrality = diagnostics['centrality_metrics'].get('avg_centrality', 0)
        if avg_centrality > 0.7:
            centrality_pattern = "star-like"
            print("  Centrality pattern is STAR-LIKE - communities have high internal centrality")
        elif avg_centrality > 0.4:
            centrality_pattern = "balanced"
            print("  Centrality pattern is BALANCED - communities have moderate internal structure")
        else:
            centrality_pattern = "distributed"
            print("  Centrality pattern is DISTRIBUTED - communities have diffuse internal structure")
    else:
        centrality_pattern = "unknown"

    # Store interpretation
    diagnostics['interpretation'] = {
        'network_structure': structure,
        'centrality_pattern': centrality_pattern,
        'document_distribution': "highly overlapping" if unique_percentage < 40 else
        "moderately overlapping" if unique_percentage < 70 else "mostly unique",
        'analysis_quality': "high" if total_communities >= 3 and structure != "monolithic" else
        "moderate" if total_communities >= 2 else "limited"
    }

    # 6. Generate visualization data
    print("\nVisualization Data:")

    # Community network visualization data
    community_nodes = []
    community_edges = []

    # Create nodes for each community
    for i, theme in enumerate(themes):
        community_nodes.append({
            'id': f"community_{i}",
            'label': theme.get('name', f"Community {i + 1}"),
            'size': len(theme.get('nodes', [])),
            'centrality': theme.get('centrality', 0),
            'keywords': theme.get('keywords', [])
        })

    # Create edges between communities that share documents
    community_docs = {}
    for i, theme in enumerate(themes):
        community_docs[i] = set(theme.get('nodes', []))

    # Compare communities for shared documents
    for i in range(len(themes)):
        for j in range(i + 1, len(themes)):
            shared_docs = community_docs[i].intersection(community_docs[j])
            if shared_docs:  # If they share any documents, create an edge
                community_edges.append({
                    'source': f"community_{i}",
                    'target': f"community_{j}",
                    'weight': len(shared_docs),
                    'shared_count': len(shared_docs)
                })

    print(
        f"  Generated visualization data for {len(community_nodes)} communities with {len(community_edges)} connections")

    # Store visualization data
    diagnostics['visualization_data'] = {
        'community_nodes': community_nodes,
        'community_edges': community_edges
    }

    return diagnostics


def _generate_keyword_diagnostics(self, keyword_results, workspace):
    """
    Generate enhanced diagnostics for keyword analysis

    Args:
        keyword_results (Dict): Keyword analysis results
        workspace (str): Workspace being analyzed

    Returns:
        Dict: Enhanced diagnostics information
    """
    print("\n=== ENHANCED KEYWORD ANALYSIS DIAGNOSTICS ===")

    # Initialize diagnostics
    diagnostics = {
        'keyword_stats': {},
        'distribution_analysis': {},
        'semantic_groups': {},
        'document_coverage': {},
        'interpretation': {},
        'visualization_data': {}
    }

    # Check if we have valid results
    if not keyword_results or 'themes' not in keyword_results:
        print("No valid keyword analysis results to diagnose")
        return diagnostics

    # Get keywords from results
    keywords = keyword_results.get('themes', [])

    # 1. Keyword Statistics
    print("Keyword Statistics:")

    # Basic counts
    total_keywords = len(keywords)

    print(f"  Total keywords extracted: {total_keywords}")

    # Score distribution
    if keywords:
        keyword_scores = [k.get('score', 0) for k in keywords]
        print(f"  Score range: {min(keyword_scores):.2f} to {max(keyword_scores):.2f}")
        print(f"  Average score: {sum(keyword_scores) / len(keyword_scores):.2f}")

        # Group by score ranges
        high_scores = len([s for s in keyword_scores if s >= 0.5])
        medium_scores = len([s for s in keyword_scores if 0.2 <= s < 0.5])
        low_scores = len([s for s in keyword_scores if s < 0.2])

        print("  Score distribution:")
        print(f"    High (≥0.5): {high_scores} keywords ({high_scores / total_keywords * 100:.1f}%)")
        print(f"    Medium (0.2-0.5): {medium_scores} keywords ({medium_scores / total_keywords * 100:.1f}%)")
        print(f"    Low (<0.2): {low_scores} keywords ({low_scores / total_keywords * 100:.1f}%)")

    # Store in diagnostics
    diagnostics['keyword_stats'] = {
        'total_keywords': total_keywords,
        'score_range': [min(keyword_scores), max(keyword_scores)] if keywords else [0, 0],
        'avg_score': sum(keyword_scores) / len(keyword_scores) if keywords else 0,
        'score_distribution': {
            'high': high_scores,
            'medium': medium_scores,
            'low': low_scores
        } if keywords else {}
    }

    # 2. Document Coverage Analysis
    print("\nDocument Coverage Analysis:")

    # Analyze how many documents each keyword appears in
    if keywords:
        doc_counts = [k.get('documents', 0) for k in keywords]

        if doc_counts:
            print(f"  Document appearance range: {min(doc_counts)} to {max(doc_counts)} documents per keyword")
            print(f"  Average documents per keyword: {sum(doc_counts) / len(doc_counts):.1f}")

            # Document coverage distribution
            widespread = len([d for d in doc_counts if d >= 5])
            moderate = len([d for d in doc_counts if 2 <= d < 5])
            single = len([d for d in doc_counts if d < 2])

            print("  Coverage distribution:")
            print(f"    Widespread (≥5 docs): {widespread} keywords ({widespread / total_keywords * 100:.1f}%)")
            print(f"    Moderate (2-4 docs): {moderate} keywords ({moderate / total_keywords * 100:.1f}%)")
            print(f"    Limited (1 doc): {single} keywords ({single / total_keywords * 100:.1f}%)")

            # Store in diagnostics
            diagnostics['document_coverage'] = {
                'doc_count_range': [min(doc_counts), max(doc_counts)],
                'avg_docs_per_keyword': sum(doc_counts) / len(doc_counts),
                'coverage_distribution': {
                    'widespread': widespread,
                    'moderate': moderate,
                    'limited': single
                }
            }

    # 3. Try to group semantically related keywords
    print("\nSemantic Grouping Analysis:")

    try:
        # Use simple character-based similarity for grouping
        keyword_texts = [k.get('keyword', '') for k in keywords]
        semantic_groups = self._group_similar_keywords(keyword_texts)

        print(f"  Identified {len(semantic_groups)} potential semantic groups")

        # Show the top 3 groups
        for i, group in enumerate(list(semantic_groups.values())[:3]):
            print(f"  Group {i + 1} example: {', '.join(group[:3])}{' ...' if len(group) > 3 else ''}")

        # Store in diagnostics
        diagnostics['semantic_groups'] = {
            'group_count': len(semantic_groups),
            'top_groups': [list(group)[:5] for group in list(semantic_groups.values())[:5]],
            'avg_group_size': sum(len(group) for group in semantic_groups.values()) / len(
                semantic_groups) if semantic_groups else 0
        }
    except Exception as e:
        print(f"  Error in semantic grouping: {str(e)}")

    # 4. Generate interpretation
    print("\nInterpretation Guidance:")

    # Keyword diversity assessment
    if keywords and 'semantic_groups' in diagnostics:
        group_count = diagnostics['semantic_groups'].get('group_count', 0)
        if group_count / total_keywords > 0.5:
            diversity = "high"
            print("  Keyword diversity appears HIGH - many distinct semantic groups")
        elif group_count / total_keywords > 0.3:
            diversity = "moderate"
            print("  Keyword diversity appears MODERATE - some semantic grouping")
        else:
            diversity = "low"
            print("  Keyword diversity appears LOW - keywords fall into few semantic groups")
    else:
        diversity = "unknown"

    # Keyword significance assessment
    if keywords and high_scores / total_keywords > 0.3:
        significance = "high"
        print("  Keyword significance appears HIGH - many keywords with strong scores")
    elif keywords and (high_scores + medium_scores) / total_keywords > 0.5:
        significance = "moderate"
        print("  Keyword significance appears MODERATE - mix of strong and medium-scoring keywords")
    else:
        significance = "low"
        print("  Keyword significance appears LOW - few keywords with strong scores")

    # Store interpretation
    diagnostics['interpretation'] = {
        'diversity': diversity,
        'significance': significance,
        'coverage': "high" if widespread > moderate + single else
        "moderate" if widespread + moderate > single else "low",
        'analysis_quality': "high" if diversity != "low" and significance != "low" else
        "moderate" if diversity != "low" or significance != "low" else "limited"
    }

    # 5. Generate visualization data
    print("\nVisualization Data:")

    # Keyword network visualization data
    keyword_nodes = []
    keyword_edges = []

    # Create nodes for each keyword
    for i, kw in enumerate(keywords):
        keyword_nodes.append({
            'id': f"keyword_{i}",
            'label': kw.get('keyword', f"Keyword {i + 1}"),
            'score': kw.get('score', 0),
            'documents': kw.get('documents', 0)
        })

    # Create edges between keywords that appear in the same documents
    # First, create document-to-keyword mapping
    doc_keywords = defaultdict(list)

    for i, kw in enumerate(keywords):
        # Check if 'doc_sources' exists and is not empty
        if hasattr(self, 'keyword_doc_mapping') and kw.get('keyword', '') in self.keyword_doc_mapping:
            doc_sources = self.keyword_doc_mapping[kw.get('keyword', '')]
            for doc in doc_sources:
                doc_keywords[doc].append(i)

    # Then create edges based on document co-occurrence
    keyword_pairs = Counter()
    for doc, kw_indices in doc_keywords.items():
        for i in range(len(kw_indices)):
            for j in range(i + 1, len(kw_indices)):
                key = tuple(sorted([kw_indices[i], kw_indices[j]]))
                keyword_pairs[key] += 1

    # Add top co-occurring pairs as edges
    for (i, j), count in keyword_pairs.most_common(min(100, len(keyword_pairs))):
        if count > 1:  # Only add edges with multiple document co-occurrences
            keyword_edges.append({
                'source': f"keyword_{i}",
                'target': f"keyword_{j}",
                'weight': count
            })

    print(f"  Generated visualization data for {len(keyword_nodes)} keywords with {len(keyword_edges)} connections")

    # Store visualization data
    diagnostics['visualization_data'] = {
        'keyword_nodes': keyword_nodes,
        'keyword_edges': keyword_edges
    }

    return diagnostics


def _group_similar_keywords(self, keywords):
    """
    Group similar keywords based on character-level similarity

    Args:
        keywords (List[str]): List of keywords

    Returns:
        Dict: Groups of similar keywords
    """
    # Use a simple approach that identifies:
    # 1. Keywords that are substrings of each other
    # 2. Keywords with high character overlap

    # Initialize groups
    groups = {}
    assigned = set()

    # Helper for character-level similarity
    def char_similarity(a, b):
        a_chars = set(a.lower())
        b_chars = set(b.lower())
        if not a_chars or not b_chars:
            return 0
        intersection = len(a_chars.intersection(b_chars))
        union = len(a_chars.union(b_chars))
        return intersection / union if union > 0 else 0

    # Process each keyword
    for i, kw1 in enumerate(keywords):
        if kw1 in assigned or not kw1:
            continue

        # Start a new group
        group_key = len(groups)
        groups[group_key] = [kw1]
        assigned.add(kw1)

        # Look for similar keywords
        for kw2 in keywords[i + 1:]:
            if kw2 in assigned or not kw2:
                continue

            # Check if one is substring of the other
            is_substring = kw1.lower() in kw2.lower() or kw2.lower() in kw1.lower()

            # Check character similarity
            similarity = char_similarity(kw1, kw2)

            # Group if similar enough
            if is_substring or similarity > 0.6:
                groups[group_key].append(kw2)
                assigned.add(kw2)

    # Filter out singleton groups
    return {k: v for k, v in groups.items() if len(v) > 1}


def _output_enhanced_diagnostics(self, workspace, diagnostic_results, method):
    """
    Output enhanced diagnostics to file

    Args:
        workspace (str): Workspace being analyzed
        diagnostic_results (Dict): Enhanced diagnostic results
        method (str): Analysis method
    """
    print("\n=== SAVING ENHANCED DIAGNOSTICS ===")

    # Get output format
    output_format = self.config.get('system.output_format', 'json')

    # Create output directory
    output_dir = os.path.join('logs', workspace)
    ensure_dir(output_dir)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostics_{method}_{timestamp}.{output_format}"
    filepath = os.path.join(output_dir, filename)

    # Save diagnostics based on format
    if output_format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            import json
            json.dump(diagnostic_results, f, indent=2)
    else:
        # Default to text format
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== THEME ANALYSIS DIAGNOSTICS - {workspace} ===\n")
            f.write(f"Method: {method}\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")

            # Write each analysis type
            for analysis_type, results in diagnostic_results.items():
                f.write(f"--- {analysis_type.upper()} DIAGNOSTICS ---\n\n")

                # Write each section
                for section, data in results.items():
                    f.write(f"{section.replace('_', ' ').title()}:\n")

                    # Handle different data types
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"  {key}: {value}\n")
                    elif isinstance(data, list):
                        for item in data:
                            f.write(f"  - {item}\n")
                    else:
                        f.write(f"  {data}\n")

                    f.write("\n")

                f.write("\n")

    print(f"Enhanced diagnostics saved to: {filepath}")
    return filepath


def send_diagnostics_to_llm(self, diagnostics_path, query=None):
    """
    Send diagnostics to LLM for interpretation

    Args:
        diagnostics_path (str): Path to diagnostics file
        query (str, optional): Specific question to ask about the diagnostics

    Returns:
        str: LLM interpretation
    """
    print("\n=== SENDING DIAGNOSTICS TO LLM FOR INTERPRETATION ===")

    # Check if LLM connector is available
    if not hasattr(self, 'llm_connector') or not self.llm_connector:
        # Create LLM connector if needed
        try:
            from core.connectors.connector_factory import ConnectorFactory
            factory = ConnectorFactory(self.config)
            self.llm_connector = factory.get_llm_connector()
        except Exception as e:
            print(f"Error creating LLM connector: {str(e)}")
            return "Could not connect to LLM for interpretation."

    try:
        # Load diagnostics file
        with open(diagnostics_path, 'r', encoding='utf-8') as f:
            diagnostics_content = f.read()

        # Limit content size to avoid token limits
        max_content_size = 8000
        if len(diagnostics_content) > max_content_size:
            print(
                f"Diagnostics content too large ({len(diagnostics_content)} chars), truncating to {max_content_size} chars")
            diagnostics_content = diagnostics_content[:max_content_size] + "...[truncated]"

        # Create prompt
        if query:
            prompt = f"""You are an expert document and text analysis assistant. 
The user has performed theme analysis on a document collection and has the following question about the results:

{query}

Below is the diagnostic output from the analysis:

{diagnostics_content}

Please provide a detailed interpretation focused specifically on the user's question."""
        else:
            prompt = f"""You are an expert document and text analysis assistant.
The user has performed theme analysis on a document collection.
Below is the diagnostic output from the analysis:

{diagnostics_content}

Please provide:
1. A concise summary of the key findings
2. An interpretation of the document relationships and themes
3. 2-3 insights about what these themes reveal about the document collection
4. Suggestions for how the user might further explore or refine the analysis"""

        # Send to LLM
        print("Sending diagnostics to LLM for interpretation...")
        model = self.config.get('llm.default_model')
        response = self.llm_connector.generate(prompt, model=model, max_tokens=1000)

        # Save LLM interpretation alongside diagnostics
        output_dir = os.path.dirname(diagnostics_path)
        interpretation_path = os.path.join(output_dir,
                                           f"{os.path.splitext(os.path.basename(diagnostics_path))[0]}_llm_interpretation.txt")

        with open(interpretation_path, 'w', encoding='utf-8') as f:
            f.write("=== LLM INTERPRETATION OF THEME ANALYSIS ===\n\n")
            if query:
                f.write(f"Question: {query}\n\n")
            f.write(response)

        print(f"LLM interpretation saved to: {interpretation_path}")

        # Return the response
        return response

    except Exception as e:
        print(f"Error sending diagnostics to LLM: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating LLM interpretation: {str(e)}"