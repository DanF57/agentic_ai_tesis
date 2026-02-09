# generate_json_v3.py
import json
import re
import uuid
import os
from typing import List, Dict, Tuple

# --- CONFIGURATION ---
JSON_FILE_PATH = 'reddit_dataset_final_1492_posts.json'  # Input file
OUTPUT_JSON_PATH = 'processed_documents_v3.json'  # Output file
MIN_SCORE_COMMENT = 1  # Minimum votes for comments
MIN_WORDS_COMMENT = 5  # Minimum words in comment
MIN_WORDS_POST_CONTENT = 10  # Minimum words in post content (excluding title)

# Patterns for filtering
url_pattern = re.compile(r'(https?://\S+)')

# Emoji pattern for removal
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001F900-\U0001F9FF"  # supplemental symbols
    u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "]+", 
    flags=re.UNICODE
)

# Career/HR related keywords to EXCLUDE
CAREER_KEYWORDS = [
    r'\bsalary\b', r'\bsalaries\b', r'\bhiring\s+manager\b', r'\binterview\b',
    r'\bresume\b', r'\bcv\b', r'\bjob\s+offer\b', r'\bjob\s+search\b',
    r'\bbootcamp\b', r'\btuition\b', r'\bcareer\s+advice\b', r'\bcareer\s+path\b',
    r'\brecruiter\b', r'\brecruiting\b', r'\bjob\s+hunt\b', r'\bjob\s+application\b',
    r'\bcompensation\b', r'\bbenefits\b', r'\b401k\b', r'\bstock\s+options\b',
    r'\bhire\s+me\b', r'\bhiring\b', r'\bapplying\s+for\b', r'\bcareer\s+change\b',
    r'\bnetworking\s+event\b', r'\blinkedin\b', r'\bcover\s+letter\b'
]

# Meta-Reddit keywords to EXCLUDE
META_REDDIT_KEYWORDS = [
    r'\bupvote\b', r'\bdownvote\b', r'\bkarma\b', r'\bsubreddit\s+rules\b',
    r'\bmod\s+team\b', r'\bmoderator\b', r'\breport\s+this\b', r'\bban\b',
    r'\bsticky\b', r'\bpinned\s+post\b'
]

# Low-effort responses to EXCLUDE
LOW_EFFORT_PATTERNS = [
    r'^(this|lol|nice|cool|same|agreed?|exactly|yes|no|nope|yep|yup|ok|okay)[\s\.\!]*$',
    r'^(thanks?|thank\s+you|thx|ty)[\s\.\!]*$',
    r'^(\?\?+|\!+)$',
    r'^(haha|lmao|rofl)[\s\.\!]*$',
]

# Meme-related keywords to EXCLUDE
MEME_KEYWORDS = [
    r'\bmeme\b', r'\bmemes\b', r'\bfunny\b', r'\blol\b', r'\blmao\b',
    r'\bmorning\s+coffee\b', r'\bcoffee\s+meme\b'
]

# Thanks with minimal content patterns (thanks + very short additional text)
def is_thanks_noise(text: str) -> bool:
    """
    Detect if comment is primarily just thanks/appreciation with minimal content.
    Examples:
    - "this is very good, thank you" ❌
    - "thanks! this helped" ❌
    - "thank you for sharing" ❌
    - "thanks! This explanation of linear regression really clarified..." ✅ (substantial)
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Check if it starts with thanks/thank you
    starts_with_thanks = re.match(r'^(thanks?|thank\s+you|thx|ty|appreciate)', text_lower)
    
    if starts_with_thanks:
        # If it starts with thanks and is short, it's noise
        word_count = len(text.split())
        if word_count < 15:  # Less than 15 words total
            return True
    
    # Check if it ends with thanks and is short overall
    ends_with_thanks = re.search(r'(thanks?|thank\s+you|thx|ty)[\s\.\!]*$', text_lower)
    if ends_with_thanks:
        word_count = len(text.split())
        # Remove the thanks part and check what's left
        text_without_thanks = re.sub(r'(thanks?|thank\s+you|thx|ty)[\s\.\!]*$', '', text_lower, flags=re.IGNORECASE).strip()
        text_without_thanks = re.sub(r'^(this\s+is\s+)?(very\s+)?(good|great|helpful|useful|awesome|nice|cool)[\s\,\.\!]*', '', text_without_thanks).strip()
        
        remaining_words = len(text_without_thanks.split())
        if remaining_words < 5:  # Less than 5 meaningful words
            return True
    
    return False

def has_broken_links(text: str) -> bool:
    """
    Detect if text has broken/incomplete markdown links like [ or [text](
    These indicate URLs were removed but left broken formatting.
    """
    # Pattern for broken markdown links
    broken_link_patterns = [
        r'\[\s*\]',  # Empty brackets []
        r'\[\s*\(',  # Opening bracket with paren [(
        r'\]\s*\[',  # Closing then opening bracket ][
        r'\(\s*\)',  # Empty parentheses ()
        r'(?<!\w)\[\s*(?!\[)',  # Single opening bracket not followed by another bracket
        r'(?<!\])\s*\](?!\])',  # Single closing bracket not preceded by another bracket
    ]
    
    for pattern in broken_link_patterns:
        if re.search(pattern, text):
            return True
    
    # Also check for "outlined:" or "approaches:" followed by broken links
    if re.search(r'(outlined|approaches|here|link):\s*\[', text, re.IGNORECASE):
        return True
    
    return False

def remove_emojis(text: str) -> str:
    """Remove all emojis from text."""
    return emoji_pattern.sub('', text).strip()

def clean_and_extract_urls(text: str) -> Tuple[str, List[str]]:
    """Removes URLs from text and returns cleaned text with extracted URLs."""
    if not text:
        return "", []
    urls = url_pattern.findall(text)
    clean_text = url_pattern.sub('', text).replace('  ', ' ').strip()
    return clean_text, urls

def contains_pattern(text: str, patterns: List[str]) -> bool:
    """Check if text contains any of the given regex patterns."""
    if not text:
        return False
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def is_low_effort(text: str) -> bool:
    """Check if text is a low-effort response."""
    if not text:
        return True
    text_clean = text.strip().lower()
    for pattern in LOW_EFFORT_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True
    return False

def preprocess_dataset(file_path: str) -> List[Dict]:
    """Preprocesses Reddit dataset with enhanced filtering for academic content."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    
    stats = {
        'posts_processed': 0,
        'posts_added': 0,
        'posts_rejected_no_content': 0,
        'posts_rejected_career': 0,
        'posts_rejected_meta': 0,
        'posts_rejected_meme': 0,
        'comments_processed': 0,
        'comments_added': 0,
        'comments_rejected_deleted': 0,
        'comments_rejected_low_score': 0,
        'comments_rejected_too_short': 0,
        'comments_rejected_career': 0,
        'comments_rejected_meta': 0,
        'comments_rejected_low_effort': 0,
        'comments_rejected_thanks_noise': 0,
        'comments_rejected_meme': 0,
        'comments_rejected_broken_links': 0,
    }
    
    print(f"Starting preprocessing of {len(data)} posts...")

    for post in data:
        stats['posts_processed'] += 1
        
        # Use existing post_id or generate one
        post_id = post.get('post_id', str(uuid.uuid4()))
        p_title = post.get('title', '').strip()
        p_text = post.get('text', '')
        p_permalink = post.get('permalink', '')
        
        # Skip deleted/removed
        if p_text in ['[deleted]', '[removed]', 'comment deleted', 'thank you']:
            stats['posts_rejected_deleted'] = stats.get('posts_rejected_deleted', 0) + 1
            continue

        p_clean_text, p_urls = clean_and_extract_urls(p_text)
        
        # Remove emojis from post
        p_title = remove_emojis(p_title)
        p_clean_text = remove_emojis(p_clean_text)
        
        # Combined text for filtering
        p_full_text = f"{p_title} {p_clean_text}"
        
        # Filter career/HR content
        if contains_pattern(p_full_text, CAREER_KEYWORDS):
            stats['posts_rejected_career'] += 1
            continue
        
        # Filter meta-Reddit content
        if contains_pattern(p_full_text, META_REDDIT_KEYWORDS):
            stats['posts_rejected_meta'] += 1
            continue
        
        # Filter meme content
        if contains_pattern(p_full_text, MEME_KEYWORDS):
            stats['posts_rejected_meme'] += 1
            continue
        
        # Check if post has meaningful content (not just title)
        post_has_content = len(p_clean_text.split()) >= MIN_WORDS_POST_CONTENT
        
        # Only add post if it has meaningful content
        if post_has_content and p_title:
            post_doc_content = f"Title: {p_title}\nContent: {p_clean_text}".strip()
            
            documents.append({
                "id": f"post_{post_id}",
                "document": post_doc_content,
                "metadata": {
                    "type": "post",
                    "subreddit": post.get('subreddit', 'unknown'),
                    "votes": post.get('score', 0),
                    "author": post.get('author', 'unknown'),
                    "permalink": p_permalink,
                    "post_title": p_title
                }
            })
            stats['posts_added'] += 1
        elif not post_has_content:
            stats['posts_rejected_no_content'] += 1
            # Continue processing comments even if post is rejected

        # Process comments (even if post was rejected for no content)
        if 'comments' in post and isinstance(post['comments'], list):
            for comment in post['comments']:
                stats['comments_processed'] += 1
                
                c_text = comment.get('text', '')
                c_score = comment.get('score', 0)
                c_id = comment.get('comment_id', str(uuid.uuid4()))
                # Comments inherit the permalink from their parent post
                c_permalink = p_permalink

                # Skip deleted/removed
                if c_text in ['[deleted]', '[removed]']:
                    stats['comments_rejected_deleted'] += 1
                    continue
                
                # Check score threshold
                if c_score < MIN_SCORE_COMMENT:
                    stats['comments_rejected_low_score'] += 1
                    continue
                
                # Check minimum words
                if len(c_text.split()) < MIN_WORDS_COMMENT:
                    stats['comments_rejected_too_short'] += 1
                    continue

                c_clean_text, c_urls = clean_and_extract_urls(c_text)
                
                # Remove emojis from comment
                c_clean_text = remove_emojis(c_clean_text)
                
                if not c_clean_text:
                    stats['comments_rejected_too_short'] += 1
                    continue
                
                # Check for broken links (incomplete markdown)
                if has_broken_links(c_clean_text):
                    stats['comments_rejected_broken_links'] += 1
                    continue
                
                # Filter low-effort responses
                if is_low_effort(c_clean_text):
                    stats['comments_rejected_low_effort'] += 1
                    continue
                
                # Filter thanks with minimal content (NEW)
                if is_thanks_noise(c_clean_text):
                    stats['comments_rejected_thanks_noise'] += 1
                    continue
                
                # Filter career/HR content
                if contains_pattern(c_clean_text, CAREER_KEYWORDS):
                    stats['comments_rejected_career'] += 1
                    continue
                
                # Filter meta-Reddit content
                if contains_pattern(c_clean_text, META_REDDIT_KEYWORDS):
                    stats['comments_rejected_meta'] += 1
                    continue
                
                # Filter meme content (NEW)
                if contains_pattern(c_clean_text, MEME_KEYWORDS):
                    stats['comments_rejected_meme'] += 1
                    continue

                # Context Injection - Keep the format for RAG
                # Clean title for context (remove emojis)
                p_title_clean = remove_emojis(p_title)
                full_comment_doc = f"Context (Post Title): {p_title_clean}\nComment: {c_clean_text}"

                documents.append({
                    "id": f"comment_{c_id}",
                    "document": full_comment_doc,
                    "metadata": {
                        "type": "comment",
                        "parent_id": post_id,
                        "subreddit": post.get('subreddit', 'unknown'),
                        "votes": c_score,
                        "author": comment.get('author', 'unknown'),
                        "permalink": c_permalink,
                        "post_title": p_title_clean
                    }
                })
                stats['comments_added'] += 1

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE - DETAILED STATS")
    print("="*80)
    
    print(f"\n📊 POSTS:")
    print(f"  Total processed:           {stats['posts_processed']:>6,}")
    print(f"  ✅ Added:                  {stats['posts_added']:>6,}")
    print(f"  ❌ Rejected (no content):  {stats['posts_rejected_no_content']:>6,}")
    print(f"  ❌ Rejected (career/HR):   {stats['posts_rejected_career']:>6,}")
    print(f"  ❌ Rejected (meta-Reddit): {stats['posts_rejected_meta']:>6,}")
    print(f"  ❌ Rejected (meme):        {stats['posts_rejected_meme']:>6,}")
    print(f"  ❌ Rejected (deleted):     {stats.get('posts_rejected_deleted', 0):>6,}")
    
    print(f"\n💬 COMMENTS:")
    print(f"  Total processed:           {stats['comments_processed']:>6,}")
    print(f"  ✅ Added:                  {stats['comments_added']:>6,}")
    print(f"  ❌ Rejected (deleted):     {stats['comments_rejected_deleted']:>6,}")
    print(f"  ❌ Rejected (low score):   {stats['comments_rejected_low_score']:>6,}")
    print(f"  ❌ Rejected (too short):   {stats['comments_rejected_too_short']:>6,}")
    print(f"  ❌ Rejected (low effort):  {stats['comments_rejected_low_effort']:>6,}")
    print(f"  ❌ Rejected (thanks only): {stats['comments_rejected_thanks_noise']:>6,}")
    print(f"  ❌ Rejected (career/HR):   {stats['comments_rejected_career']:>6,}")
    print(f"  ❌ Rejected (meta-Reddit): {stats['comments_rejected_meta']:>6,}")
    print(f"  ❌ Rejected (meme):        {stats['comments_rejected_meme']:>6,}")
    print(f"  ❌ Rejected (broken links):{stats['comments_rejected_broken_links']:>6,}")
    
    print(f"\n📈 TOTALS:")
    print(f"  Final document count:      {len(documents):>6,}")
    print(f"  Posts in final set:        {stats['posts_added']:>6,}")
    print(f"  Comments in final set:     {stats['comments_added']:>6,}")
    
    rejection_rate_posts = (stats['posts_processed'] - stats['posts_added']) / stats['posts_processed'] * 100 if stats['posts_processed'] > 0 else 0
    rejection_rate_comments = (stats['comments_processed'] - stats['comments_added']) / stats['comments_processed'] * 100 if stats['comments_processed'] > 0 else 0
    
    print(f"\n📉 REJECTION RATES:")
    print(f"  Posts rejected:            {rejection_rate_posts:>5.1f}%")
    print(f"  Comments rejected:         {rejection_rate_comments:>5.1f}%")
    print("="*80 + "\n")
    
    return documents

def main():
    # Process Data
    documents = preprocess_dataset(JSON_FILE_PATH)

    if not documents:
        print("No documents to export.")
        return

    # Save to JSON
    print(f"💾 Saving {len(documents):,} documents to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Successfully saved to {OUTPUT_JSON_PATH}")
    print(f"✅ Total documents: {len(documents):,}\n")

if __name__ == "__main__":
    main()