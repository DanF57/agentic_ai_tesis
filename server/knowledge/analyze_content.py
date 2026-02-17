# analyze_content.py
import json
import re
from collections import Counter
from typing import List, Dict
import random

# --- CONFIGURATION ---
INPUT_JSON_PATH = 'processed_documents_v2.json'
SAMPLE_SIZE = 10  # Number of random examples to show

def load_documents(file_path: str) -> List[Dict]:
    """Load processed documents from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

def analyze_content_patterns(documents: List[Dict]) -> Dict:
    """Analyze documents for patterns, common phrases, and potential noise."""
    
    posts = [doc for doc in documents if doc['metadata']['type'] == 'post']
    comments = [doc for doc in documents if doc['metadata']['type'] == 'comment']
    
    # Word frequency analysis
    all_words = []
    short_phrases = []
    
    for doc in documents:
        text = doc['document'].lower()
        words = re.findall(r'\b\w+\b', text)
        all_words.extend(words)
        
        # Extract 2-3 word phrases
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            short_phrases.append(phrase)
    
    # Common noise patterns to detect
    noise_patterns = {
        'thanks': r'\b(thanks?|thank you|thx|ty)\b',
        'upvote': r'\b(upvote|downvote|karma)\b',
        'edit': r'\b(edit:|edited|update:)\b',
        'deleted': r'\[(deleted|removed)\]',
        'meta': r'\b(post|comment|thread|subreddit)\b',
        'low_effort': r'^(this|lol|nice|cool|same|agreed?|exactly)[\s\.\!]*$',
        'questions_only': r'^\?+$|^what\?*$|^how\?*$|^why\?*$',
    }
    
    noise_matches = {key: 0 for key in noise_patterns.keys()}
    
    for doc in documents:
        text = doc['document'].lower()
        for noise_type, pattern in noise_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                noise_matches[noise_type] += 1
    
    return {
        'total_docs': len(documents),
        'posts': len(posts),
        'comments': len(comments),
        'word_freq': Counter(all_words),
        'phrase_freq': Counter(short_phrases),
        'noise_matches': noise_matches,
        'sample_posts': random.sample(posts, min(SAMPLE_SIZE, len(posts))),
        'sample_comments': random.sample(comments, min(SAMPLE_SIZE, len(comments)))
    }

def analyze_length_distribution(documents: List[Dict]) -> Dict:
    """Analyze document length distributions."""
    
    post_lengths = []
    comment_lengths = []
    
    for doc in documents:
        doc_length = len(doc['document'].split())
        if doc['metadata']['type'] == 'post':
            post_lengths.append(doc_length)
        else:
            comment_lengths.append(doc_length)
    
    def get_stats(lengths):
        if not lengths:
            return {}
        return {
            'min': min(lengths),
            'max': max(lengths),
            'avg': sum(lengths) / len(lengths),
            'median': sorted(lengths)[len(lengths) // 2]
        }
    
    return {
        'posts': get_stats(post_lengths),
        'comments': get_stats(comment_lengths)
    }

def identify_low_quality(documents: List[Dict]) -> List[Dict]:
    """Identify potentially low-quality documents."""
    
    low_quality = []
    
    for doc in documents:
        text = doc['document']
        word_count = len(text.split())
        
        # Criteria for low quality
        is_short = word_count < 10
        is_just_thanks = re.match(r'^(thanks?|thank you|thx)[\s\.\!]*$', text.lower().strip())
        is_one_word = word_count == 1
        has_no_letters = not re.search(r'[a-zA-Z]', text)
        is_just_link = text.strip().startswith('http')
        
        if any([is_short and doc['metadata']['type'] == 'post', 
                is_just_thanks, 
                is_one_word, 
                has_no_letters,
                is_just_link]):
            low_quality.append({
                'id': doc['id'],
                'type': doc['metadata']['type'],
                'text': text[:100],
                'word_count': word_count,
                'reason': 'short' if is_short else 'low_effort'
            })
    
    return low_quality

def print_report(analysis: Dict, length_stats: Dict, low_quality: List[Dict]):
    """Print comprehensive analysis report."""
    
    print("=" * 80)
    print("DOCUMENT ANALYSIS REPORT")
    print("=" * 80)
    
    # Basic Stats
    print(f"\n📊 BASIC STATISTICS")
    print(f"{'Total Documents:':<30} {analysis['total_docs']:>10,}")
    print(f"{'Posts:':<30} {analysis['posts']:>10,}")
    print(f"{'Comments:':<30} {analysis['comments']:>10,}")
    
    # Length Distribution
    print(f"\n📏 LENGTH DISTRIBUTION (words)")
    print(f"\n  Posts:")
    for key, val in length_stats['posts'].items():
        print(f"    {key.capitalize():<10} {val:>10.1f}")
    
    print(f"\n  Comments:")
    for key, val in length_stats['comments'].items():
        print(f"    {key.capitalize():<10} {val:>10.1f}")
    
    # Most Common Words
    print(f"\n📝 TOP 30 MOST COMMON WORDS")
    for word, count in analysis['word_freq'].most_common(30):
        if len(word) > 2:  # Skip very short words
            print(f"  {word:<20} {count:>6,}")
    
    # Common Phrases
    print(f"\n💬 TOP 20 MOST COMMON 3-WORD PHRASES")
    for phrase, count in analysis['phrase_freq'].most_common(20):
        print(f"  {phrase:<40} {count:>6,}")
    
    # Noise Detection
    print(f"\n🔍 POTENTIAL NOISE PATTERNS DETECTED")
    for noise_type, count in sorted(analysis['noise_matches'].items(), 
                                     key=lambda x: x[1], 
                                     reverse=True):
        if count > 0:
            print(f"  {noise_type:<20} {count:>6,} documents")
    
    # Low Quality
    print(f"\n⚠️  LOW QUALITY DOCUMENTS")
    print(f"  Total: {len(low_quality):,}")
    if low_quality:
        print(f"\n  Sample (first 10):")
        for item in low_quality[:10]:
            print(f"    [{item['type']}] {item['id']}")
            print(f"      Words: {item['word_count']} | Reason: {item['reason']}")
            print(f"      Text: {item['text'][:80]}...")
            print()
    
    # Sample Posts
    print(f"\n📄 SAMPLE POSTS (random {SAMPLE_SIZE})")
    print("-" * 80)
    for i, post in enumerate(analysis['sample_posts'][:5], 1):
        print(f"\n{i}. Score: {post['metadata']['votes']} | Subreddit: {post['metadata']['subreddit']}")
        print(f"   {post['document'][:200]}...")
    
    # Sample Comments
    print(f"\n💭 SAMPLE COMMENTS (random {SAMPLE_SIZE})")
    print("-" * 80)
    for i, comment in enumerate(analysis['sample_comments'][:5], 1):
        print(f"\n{i}. Votes: {comment['metadata']['votes']} | Parent: {comment['metadata']['post_title'][:50]}")
        print(f"   {comment['document'][:200]}...")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR NOISE REDUCTION")
    print("-" * 80)
    
    recommendations = []
    
    if analysis['noise_matches']['thanks'] > 100:
        recommendations.append("• Filter out 'thank you' only comments")
    
    if analysis['noise_matches']['low_effort'] > 100:
        recommendations.append("• Remove single-word responses (lol, nice, cool, etc.)")
    
    if len(low_quality) > 500:
        recommendations.append(f"• Remove {len(low_quality):,} low-quality documents (< 10 words)")
    
    avg_comment_len = length_stats['comments'].get('avg', 0)
    if avg_comment_len < 20:
        recommendations.append("• Consider increasing MIN_WORDS_COMMENT threshold")
    
    if analysis['noise_matches']['meta'] > 1000:
        recommendations.append("• Filter meta-discussion about Reddit itself")
    
    recommendations.append("• Remove comments with score < 2 or 3 for better quality")
    recommendations.append("• Filter out comments that are just questions without context")
    recommendations.append("• Remove posts/comments that are purely links without explanation")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 80)

def main():
    print("Loading documents...")
    documents = load_documents(INPUT_JSON_PATH)
    
    if not documents:
        return
    
    print(f"Loaded {len(documents):,} documents. Analyzing...\n")
    
    # Run analyses
    analysis = analyze_content_patterns(documents)
    length_stats = analyze_length_distribution(documents)
    low_quality = identify_low_quality(documents)
    
    # Print report
    print_report(analysis, length_stats, low_quality)
    
    # Save detailed low-quality list
    if low_quality:
        output_file = 'low_quality_documents.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(low_quality, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed low-quality list saved to: {output_file}")

if __name__ == "__main__":
    main()