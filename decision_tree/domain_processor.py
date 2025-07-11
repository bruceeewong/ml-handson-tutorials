"""
Domain Processor for Application Classification
Adapted from Dom2Vec approach for splitting domain names into meaningful tokens
"""

import re
import string
from typing import List, Set
import nltk
from nltk.corpus import words

class DomainProcessor:
    def __init__(self):
        """Initialize the domain processor with word dictionaries."""
        # Download NLTK words corpus if not already present
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        
        # Load English words corpus
        self.english_words = set(words.words())
        
        # Add common tech/domain words not in standard dictionary
        self.tech_words = {
            'api', 'cdn', 'www', 'mail', 'ftp', 'blog', 'shop', 'store', 
            'app', 'mobile', 'web', 'admin', 'user', 'client', 'server',
            'static', 'assets', 'media', 'img', 'images', 'video', 'audio',
            'secure', 'ssl', 'auth', 'login', 'signup', 'account',
            'netflix', 'youtube', 'spotify', 'zoom', 'teams', 'slack',
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta',
            'github', 'gitlab', 'docker', 'kubernetes', 'googleapis',
            'nflx', 'goog', 'msft', 'amzn', 'fb', 'ig', 'yt', 'fonts',
            'heroku', 'herokuapp'  # Removed user123 since pattern recognition handles it
        }
        
        # Common TLDs (Top-Level Domains)
        self.tlds = {
            'com', 'net', 'org', 'edu', 'gov', 'mil', 'int',
            'io', 'co', 'tv', 'me', 'ai', 'app', 'dev', 'tech',
            'info', 'biz', 'name', 'mobi', 'travel', 'jobs',
            'uk', 'de', 'fr', 'jp', 'cn', 'ru', 'br', 'ca', 'au',
            'local', 'ddns', 'dyndns', 'localhost'
        }
        
        # Combine word sets
        self.all_words = self.english_words.union(self.tech_words)
        
        # Convert to lowercase for matching
        self.all_words = {word.lower() for word in self.all_words}
        
        # Add TLDs to words so they're recognized as complete units
        self.all_words.update(self.tlds)
    
    def clean_domain(self, domain: str) -> str:
        """Clean and normalize domain name."""
        # Remove protocol if present
        domain = re.sub(r'^https?://', '', domain)
        
        # Convert to lowercase
        domain = domain.lower()
        
        # Remove trailing slash
        domain = domain.rstrip('/')
        
        # Extract just the domain part (remove path)
        domain = domain.split('/')[0]
        
        return domain
    
    def split_by_separators(self, text: str) -> List[str]:
        """Split text by common separators."""
        # Split by dots, hyphens, underscores
        tokens = re.split(r'[.\-_]', text)
        return [token for token in tokens if token]
    
    def split_camel_case(self, text: str) -> List[str]:
        """Split camelCase and PascalCase words."""
        # Insert space before uppercase letters (except at start)
        text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        return text.split()
    
    def is_likely_dga_token(self, token: str) -> bool:
        """Check if a token looks like it's from a Domain Generation Algorithm."""
        # DGA tokens are typically:
        # - Long (> 8 characters)
        # - Have few vowels
        # - Have unusual character patterns
        # - Don't match known words
        
        if len(token) < 8:  # Lowered threshold from 6 to 8
            return False
        
        # Skip known words and TLDs
        if token.lower() in self.all_words or token.lower() in self.tlds:
            return False
        
        # Count vowels
        vowels = sum(1 for c in token.lower() if c in 'aeiou')
        vowel_ratio = vowels / len(token)
        
        # DGA domains typically have low vowel ratios and unusual patterns
        if vowel_ratio < 0.25 and len(token) > 12:  # Adjusted thresholds
            return True
        
        # Check for very long tokens with mixed alphanumeric (common in DGA)
        if len(token) > 15 and any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            return True
        
        # Check for repetitive patterns (common in DGA)
        if len(set(token)) < len(token) * 0.6 and len(token) > 10:  # Adjusted ratio
            return True
        
        # Check for very long tokens that aren't meaningful words
        if len(token) > 20:
            return True
        
        return False
    
    def extract_words_from_token(self, token: str) -> List[str]:
        """Extract known words from a token using longest match strategy."""
        token_lower = token.lower()
        
        # If it's a known TLD, return as-is
        if token_lower in self.tlds:
            return [token_lower]
        
        # If it's already a known word, return as-is
        if token_lower in self.all_words:
            return [token_lower]
        
        # Check if it's a meaningful token based on patterns
        if self.is_meaningful_token(token):
            return [token_lower]
        
        # Check if it looks like a DGA token
        if self.is_likely_dga_token(token):
            # For DGA tokens, try to extract meaningful chunks instead of single chars
            chunks = []
            i = 0
            while i < len(token):
                # Try to find 3-4 character chunks first
                chunk_found = False
                for chunk_size in [4, 3, 2]:
                    if i + chunk_size <= len(token):
                        chunk = token[i:i + chunk_size]
                        if chunk.lower() in self.all_words:
                            chunks.append(chunk.lower())
                            i += chunk_size
                            chunk_found = True
                            break
                
                if not chunk_found:
                    # Take 2-3 character chunks for DGA tokens instead of single chars
                    chunk_size = min(3, len(token) - i)
                    chunks.append(token[i:i + chunk_size].lower())
                    i += chunk_size
            
            return chunks
        
        # For mixed alphanumeric tokens, use smart splitting first
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            return self.smart_split_alphanumeric(token)
        
        # Regular word extraction for non-DGA tokens
        words_found = []
        i = 0
        
        while i < len(token):
            # Try to find the longest matching word starting at position i
            longest_match = ""
            longest_match_end = i
            
            for j in range(i + 1, len(token) + 1):
                substring = token[i:j].lower()
                if substring in self.all_words and len(substring) > len(longest_match):
                    longest_match = substring
                    longest_match_end = j
            
            if longest_match:
                words_found.append(longest_match)
                i = longest_match_end
            else:
                # If no word found, add 2-character sequence or single character
                if i + 1 < len(token) and token[i:i+2].isalpha():
                    words_found.append(token[i:i+2].lower())
                    i += 2
                else:
                    words_found.append(token[i].lower())
                    i += 1
        
        return words_found
    
    def is_meaningful_token(self, token: str) -> bool:
        """Check if a token should be kept as-is based on patterns."""
        token_lower = token.lower()
        
        # Known words and TLDs
        if token_lower in self.all_words or token_lower in self.tlds:
            return True
        
        # Pattern-based recognition for meaningful tokens
        patterns = [
            r'^[a-z]+\d{1,4}$',      # word followed by 1-4 digits (user123, app2024)
            r'^\d{1,4}[a-z]+$',      # 1-4 digits followed by word (2024app)
            r'^\d{4}$',              # 4-digit numbers (years: 1900-2099)
            r'^\d{1,3}$',            # 1-3 digit numbers (versions, ports)
            r'^v\d+(\.\d+)*$',       # version patterns (v1, v2.1, v1.2.3)
            r'^[a-z]{2,}\d[a-z]\d$', # pattern like 'word1a2'
        ]
        
        for pattern in patterns:
            if re.match(pattern, token_lower):
                return True
        
        return False
    
    def smart_split_alphanumeric(self, token: str) -> List[str]:
        """Intelligently split alphanumeric tokens preserving meaningful patterns."""
        # First check if the whole token should be kept as-is
        if self.is_meaningful_token(token):
            return [token.lower()]
        
        # Split on letter-number transitions, but preserve meaningful sequences
        parts = re.findall(r'[a-zA-Z]+|[0-9]+', token)
        
        result = []
        i = 0
        while i < len(parts):
            current_part = parts[i]
            
            # If current part is letters and next is numbers, try to combine meaningfully
            if (i + 1 < len(parts) and 
                current_part.isalpha() and 
                parts[i + 1].isdigit()):
                
                combined = current_part + parts[i + 1]
                if self.is_meaningful_token(combined):
                    result.append(combined.lower())
                    i += 2  # Skip next part since we combined it
                    continue
            
            # Handle the current part
            if current_part.isdigit():
                if len(current_part) <= 4:  # Keep short numbers together
                    result.append(current_part)
                else:  # Split very long numbers into chunks
                    for j in range(0, len(current_part), 3):
                        result.append(current_part[j:j+3])
            else:
                result.append(current_part.lower())
            
            i += 1
        
        return result
    
    def process_domain(self, domain: str) -> List[str]:
        """
        Process a domain name and split it into meaningful tokens.
        
        Args:
            domain: Domain name to process
            
        Returns:
            List of tokens representing the domain
        """
        # Clean the domain
        domain = self.clean_domain(domain)
        
        # Split by separators (dots, hyphens, underscores)
        initial_tokens = self.split_by_separators(domain)
        
        final_tokens = []
        
        for token in initial_tokens:
            if not token:
                continue
            
            # Skip very short tokens unless they're meaningful
            if len(token) <= 1 and token not in ['a', 'i']:
                final_tokens.append(token)
                continue
            
            # Check if it's a known TLD first
            if token.lower() in self.tlds:
                final_tokens.append(token.lower())
                continue
            
            # Try to extract known words from the token
            if token.lower() in self.all_words:
                # Token is already a known word
                final_tokens.append(token.lower())
            else:
                # Try to split camelCase first
                camel_parts = self.split_camel_case(token)
                
                for part in camel_parts:
                    if not part:
                        continue
                    
                    # Try to extract words from each part
                    words = self.extract_words_from_token(part)
                    
                    if len(words) == 1 and words[0] == part.lower() and not part.lower() in self.all_words:
                        # Couldn't split further and it's not a known word, try alphanumeric split
                        alpha_parts = self.smart_split_alphanumeric(part)
                        final_tokens.extend([p.lower() for p in alpha_parts])
                    else:
                        final_tokens.extend(words)
        
        # Filter out empty tokens and convert to lowercase
        final_tokens = [token.lower() for token in final_tokens if token and len(token) > 0]
        
        return final_tokens
    
    def get_domain_features(self, domain: str) -> dict:
        """Extract additional features from domain for classification."""
        clean_domain = self.clean_domain(domain)
        tokens = self.process_domain(domain)
        
        # Check for DGA characteristics in both original tokens and final tokens
        original_tokens = self.split_by_separators(clean_domain)
        has_dga_tokens = (
            any(self.is_likely_dga_token(token) for token in original_tokens) or
            any(self.is_likely_dga_token(token) for token in tokens) or
            self._has_dga_pattern(tokens)
        )
        
        features = {
            'domain_length': len(clean_domain),
            'subdomain_count': clean_domain.count('.'),
            'has_numbers': any(c.isdigit() for c in clean_domain),
            'has_hyphens': '-' in clean_domain,
            'has_underscores': '_' in clean_domain,
            'token_count': len(tokens),
            'avg_token_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'has_dga_tokens': has_dga_tokens,
            'vowel_ratio': self._calculate_vowel_ratio(clean_domain),
            'char_diversity': len(set(clean_domain)) / len(clean_domain) if clean_domain else 0,
        }
        
        return features
    
    def _calculate_vowel_ratio(self, text: str) -> float:
        """Calculate the ratio of vowels to total characters."""
        if not text:
            return 0
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        return vowels / len(text)

    def _has_dga_pattern(self, tokens: List[str]) -> bool:
        """Check if the token pattern suggests DGA generation."""
        if len(tokens) < 5:
            return False
        
        # Count short random-looking tokens
        short_random_tokens = 0
        for token in tokens:
            # Skip known words and TLDs
            if token in self.all_words or token in self.tlds:
                continue
            # Count tokens that are short and contain mixed alphanumeric
            if (len(token) <= 4 and 
                any(c.isdigit() for c in token) and 
                any(c.isalpha() for c in token)):
                short_random_tokens += 1
        
        # If many short random tokens, likely DGA
        if short_random_tokens >= 3:
            return True
        
        return False

# Example usage and testing
if __name__ == "__main__":
    processor = DomainProcessor()
    
    # Test domains
    test_domains = [
        "netflix.com",
        "nflxvideo.net", 
        "api.spotify.com",
        "teams.microsoft.com",
        "youtube.com",
        "fonts.googleapis.com",
        "cvyh1po636avyrsxebwbkn7.ddns.net",  # Random DGA-like
        "mortiscontrastatim.com",  # Dictionary-based DGA
        # Additional test cases
        "subdomain.example.org",  # Normal subdomain
        "very-long-but-legitimate-domain-name.com",  # Long but legitimate
        "a1b2c3d4e5f6g7h8i9j0.malware.net",  # Clear DGA pattern
        "user123.github.io",  # Normal with numbers
        "my-app-2024.herokuapp.com"  # Normal app domain
    ]
    
    print("Domain Processing Examples:")
    print("=" * 50)
    
    for domain in test_domains:
        tokens = processor.process_domain(domain)
        features = processor.get_domain_features(domain)
        
        print(f"\nDomain: {domain}")
        print(f"Tokens: {tokens}")
        print(f"Key features: domain_length={features['domain_length']}, has_dga_tokens={features['has_dga_tokens']}") 