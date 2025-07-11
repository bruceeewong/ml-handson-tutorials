"""Domain synthesizer for generating synthetic training data."""

import random
import re
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path

from .templates import (
    DOMAIN_TEMPLATES, SERVICE_VARIATIONS, VALID_TLDS, 
    COUNTRY_CODES, LANGUAGE_CODES, VERSION_PATTERNS, NUMBER_PATTERNS
)
from .mutations import (
    MUTATIONS, REALISTIC_VARIANTS, AMBIGUOUS_PATTERNS, NOISE_PATTERNS
)


class DomainSynthesizer:
    """Generate synthetic domains for training data augmentation."""
    
    def __init__(self, real_domains: Dict[str, List[str]], seed: int = 42):
        """Initialize the synthesizer with real domain data.
        
        Args:
            real_domains: Dictionary mapping application names to lists of real domains
            seed: Random seed for reproducibility
        """
        self.real_domains = real_domains
        self.generated_domains = set()
        self.random = random.Random(seed)
        
        # Extract patterns from real domains
        self._analyze_real_domains()
        
    def _analyze_real_domains(self):
        """Analyze real domains to extract common patterns."""
        self.real_patterns = {
            'tld_frequency': {},
            'token_frequency': {},
            'length_distribution': [],
            'subdomain_counts': []
        }
        
        for app, domains in self.real_domains.items():
            for domain in domains:
                # TLD analysis
                parts = domain.split('.')
                if len(parts) >= 2:
                    tld = parts[-1]
                    self.real_patterns['tld_frequency'][tld] = \
                        self.real_patterns['tld_frequency'].get(tld, 0) + 1
                
                # Token analysis
                for token in parts:
                    self.real_patterns['token_frequency'][token] = \
                        self.real_patterns['token_frequency'].get(token, 0) + 1
                
                # Length and structure
                self.real_patterns['length_distribution'].append(len(domain))
                self.real_patterns['subdomain_counts'].append(len(parts) - 1)
    
    def generate_from_template(self, app_name: str, template: str, count: int = 10) -> List[str]:
        """Generate domains using template substitution.
        
        Args:
            app_name: Application name (e.g., 'netflix', 'spotify')
            template: Template string with placeholders
            count: Number of domains to generate
            
        Returns:
            List of generated domain strings
        """
        domains = []
        variations = SERVICE_VARIATIONS.get(app_name, {})
        
        if not variations:
            return domains
        
        attempts = 0
        max_attempts = count * 10
        
        while len(domains) < count and attempts < max_attempts:
            attempts += 1
            
            # Start with the template
            domain = template
            
            # Replace placeholders
            replacements = {
                '{service}': self.random.choice(variations.get('base', [app_name])),
                '{tld}': self.random.choice(variations.get('tlds', ['com'])),
                '{region}': self.random.choice(variations.get('regions', ['us'])),
                '{country_code}': self.random.choice(list(COUNTRY_CODES.keys())),
                '{country}': self.random.choice(list(COUNTRY_CODES.keys())),
                '{lang}': self.random.choice(LANGUAGE_CODES),
                '{function}': self.random.choice(variations.get('functions', ['app'])),
                '{subdomain}': self.random.choice(variations.get('subdomains', ['www'])),
                '{version}': self.random.choice(VERSION_PATTERNS),
                '{number}': self.random.choice(NUMBER_PATTERNS),
                '{env}': self.random.choice(variations.get('envs', ['prod'])),
            }
            
            # Special handling for CDN terms
            if '{service}' in domain and self.random.random() < 0.3:
                cdn_terms = variations.get('cdn_terms', [])
                if cdn_terms:
                    replacements['{service}'] = self.random.choice(cdn_terms)
            
            # Apply replacements
            for placeholder, value in replacements.items():
                domain = domain.replace(placeholder, value)
            
            # Validate and add
            if self.validate_domain(domain) and domain not in self.generated_domains:
                domains.append(domain)
                self.generated_domains.add(domain)
        
        return domains
    
    def apply_mutations(self, domain: str, max_mutations: int = 3) -> List[str]:
        """Apply realistic mutations to existing domains.
        
        Args:
            domain: Original domain string
            max_mutations: Maximum number of mutations to generate
            
        Returns:
            List of mutated domain strings
        """
        mutated = []
        
        # Abbreviation mutations
        for word, abbrevs in MUTATIONS['abbreviations'].items():
            if word in domain:
                for abbrev in abbrevs[:max_mutations]:
                    new_domain = domain.replace(word, abbrev)
                    if self.validate_domain(new_domain):
                        mutated.append(new_domain)
        
        # Synonym replacements
        tokens = domain.split('.')
        for i, token in enumerate(tokens):
            for word, synonyms in MUTATIONS['synonyms'].items():
                if word == token:
                    for synonym in self.random.sample(synonyms, min(len(synonyms), 2)):
                        new_tokens = tokens.copy()
                        new_tokens[i] = synonym
                        new_domain = '.'.join(new_tokens)
                        if self.validate_domain(new_domain):
                            mutated.append(new_domain)
        
        # Prefix/suffix mutations
        if len(tokens) > 1:
            # Add prefixes
            for prefix_type, prefixes in MUTATIONS['prefixes'].items():
                if self.random.random() < 0.3:
                    prefix = self.random.choice(prefixes)
                    new_domain = prefix + domain
                    if self.validate_domain(new_domain):
                        mutated.append(new_domain)
            
            # Add suffixes (to subdomain, not TLD)
            for suffix_type, suffixes in MUTATIONS['suffixes'].items():
                if self.random.random() < 0.3:
                    suffix = self.random.choice(suffixes)
                    parts = domain.rsplit('.', 1)
                    new_domain = parts[0] + suffix + '.' + parts[1]
                    if self.validate_domain(new_domain):
                        mutated.append(new_domain)
        
        # Realistic variants
        for variant_type, variants in REALISTIC_VARIANTS.items():
            if self.random.random() < 0.2:
                variant = self.random.choice(variants)
                # Add as subdomain
                new_domain = f"{variant}.{domain}"
                if self.validate_domain(new_domain):
                    mutated.append(new_domain)
        
        return list(set(mutated))[:max_mutations]
    
    def generate_ambiguous_domains(self, app_name: str, count: int = 10) -> List[str]:
        """Generate domains that could belong to multiple applications.
        
        Args:
            app_name: Application to bias towards
            count: Number of domains to generate
            
        Returns:
            List of ambiguous domain strings
        """
        domains = []
        variations = SERVICE_VARIATIONS.get(app_name, {})
        
        for _ in range(count * 2):
            template = self.random.choice(AMBIGUOUS_PATTERNS)
            
            # Sometimes use generic terms, sometimes app-specific
            if self.random.random() < 0.5:
                # Generic terms
                domain = template.replace('{tld}', self.random.choice(VALID_TLDS))
            else:
                # App-specific but ambiguous
                base_term = self.random.choice(variations.get('base', [app_name]))
                domain = template.replace('{tld}', self.random.choice(VALID_TLDS))
                
                # Add app-specific touch to ambiguous pattern
                if self.random.random() < 0.3:
                    domain = domain.replace('-', f'-{base_term}-')
                elif self.random.random() < 0.3:
                    parts = domain.split('.')
                    if len(parts) > 1:
                        parts.insert(-1, base_term)
                        domain = '.'.join(parts)
            
            if self.validate_domain(domain) and domain not in self.generated_domains:
                domains.append(domain)
                self.generated_domains.add(domain)
                
                if len(domains) >= count:
                    break
        
        return domains
    
    def add_realistic_noise(self, domain: str) -> str:
        """Add realistic noise to a domain (typos, variations).
        
        Args:
            domain: Original domain
            
        Returns:
            Domain with realistic noise
        """
        if self.random.random() < 0.1:  # 10% chance of noise
            parts = domain.split('.')
            
            # Apply noise to a random part (not TLD)
            if len(parts) > 1:
                idx = self.random.randint(0, len(parts) - 2)
                part = parts[idx]
                
                # Choose noise type
                noise_type = self.random.choice(['typos', 'variations'])
                noise_funcs = list(NOISE_PATTERNS[noise_type].values())
                noise_func = self.random.choice(noise_funcs)
                
                try:
                    parts[idx] = noise_func(part)
                    return '.'.join(parts)
                except:
                    return domain
        
        return domain
    
    def validate_domain(self, domain: str) -> bool:
        """Validate that a domain is realistic and well-formed.
        
        Args:
            domain: Domain string to validate
            
        Returns:
            True if domain is valid, False otherwise
        """
        # Length constraints
        if len(domain) < 4 or len(domain) > 63:
            return False
        
        # Character constraints
        if not re.match(r'^[a-z0-9.-]+$', domain.lower()):
            return False
        
        # Structure constraints
        parts = domain.split('.')
        if len(parts) < 2:
            return False
        
        # Each part should be non-empty
        for part in parts:
            if not part or len(part) > 63:
                return False
            # No leading/trailing hyphens
            if part.startswith('-') or part.endswith('-'):
                return False
        
        # Valid TLD
        tld = parts[-1]
        if tld not in VALID_TLDS and not tld in self.real_patterns['tld_frequency']:
            return False
        
        # Not already in real domains
        for app_domains in self.real_domains.values():
            if domain in app_domains:
                return False
        
        return True
    
    def generate_synthetic_dataset(self, domains_per_app: int = 1000) -> Dict[str, List[str]]:
        """Generate complete synthetic dataset.
        
        Args:
            domains_per_app: Target number of domains per application
            
        Returns:
            Dictionary mapping app names to lists of synthetic domains
        """
        synthetic_data = {}
        
        for app_name in self.real_domains.keys():
            print(f"Generating synthetic domains for {app_name}...")
            app_domains = []
            
            # Phase 1: Template-based generation (70%)
            template_target = int(domains_per_app * 0.7)
            templates_per_type = template_target // len(DOMAIN_TEMPLATES)
            
            for template_type, templates in DOMAIN_TEMPLATES.items():
                for template in templates:
                    count = templates_per_type // len(templates)
                    domains = self.generate_from_template(app_name, template, count)
                    app_domains.extend(domains)
            
            # Phase 2: Mutation-based generation (20%)
            mutation_target = int(domains_per_app * 0.2)
            real_app_domains = self.real_domains[app_name]
            mutations_per_domain = max(1, mutation_target // len(real_app_domains))
            
            for real_domain in real_app_domains:
                mutated = self.apply_mutations(real_domain, mutations_per_domain)
                app_domains.extend(mutated)
            
            # Phase 3: Ambiguous patterns (10%)
            ambiguous_target = int(domains_per_app * 0.1)
            ambiguous = self.generate_ambiguous_domains(app_name, ambiguous_target)
            app_domains.extend(ambiguous)
            
            # Add realistic noise to some domains
            app_domains = [self.add_realistic_noise(d) if self.random.random() < 0.05 else d 
                          for d in app_domains]
            
            # Remove duplicates and limit to target
            app_domains = list(set(app_domains))
            self.random.shuffle(app_domains)
            synthetic_data[app_name] = app_domains[:domains_per_app]
            
            print(f"  Generated {len(synthetic_data[app_name])} domains for {app_name}")
        
        return synthetic_data
    
    def save_synthetic_data(self, synthetic_data: Dict[str, List[str]], output_path: str):
        """Save synthetic data to JSON file.
        
        Args:
            synthetic_data: Dictionary of synthetic domains
            output_path: Path to save JSON file
        """
        output = {
            'metadata': {
                'total_domains': sum(len(domains) for domains in synthetic_data.values()),
                'applications': list(synthetic_data.keys()),
                'domains_per_app': {app: len(domains) for app, domains in synthetic_data.items()},
                'generation_method': 'DomainSynthesizer v1.1'
            },
            'domains': synthetic_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSynthetic data saved to {output_path}")
        print(f"Total domains generated: {output['metadata']['total_domains']}")


if __name__ == "__main__":
    # Example usage
    from sample_data import APPLICATION_DOMAINS
    
    synthesizer = DomainSynthesizer(APPLICATION_DOMAINS)
    synthetic_data = synthesizer.generate_synthetic_dataset(domains_per_app=100)
    
    # Print sample
    for app, domains in synthetic_data.items():
        print(f"\n{app} sample domains:")
        for domain in domains[:5]:
            print(f"  - {domain}")