"""Domain templates for synthetic data generation."""

DOMAIN_TEMPLATES = {
    'api_endpoints': [
        'api.{service}.{tld}',
        'api-{region}.{service}.{tld}',
        '{service}-api.{tld}',
        'api.{subdomain}.{service}.{tld}',
        '{version}.api.{service}.{tld}',
        'rest.{service}.{tld}',
        'graphql.{service}.{tld}',
        'api-gateway.{service}.{tld}',
        '{service}.api.{tld}',
        'developer.{service}.{tld}'
    ],
    'cdn_patterns': [
        'cdn.{service}.{tld}',
        '{service}cdn.{tld}',
        'static.{service}.{tld}',
        'assets.{service}.{tld}',
        '{region}-cdn.{service}.{tld}',
        'cdn{number}.{service}.{tld}',
        'edge.{service}.{tld}',
        'cache.{service}.{tld}',
        'media.{service}.{tld}',
        'content.{service}.{tld}',
        'img.{service}.{tld}',
        'images.{service}.{tld}',
        '{service}-static.{tld}',
        'static-{region}.{service}.{tld}'
    ],
    'regional': [
        '{service}.{country_code}',
        '{region}.{service}.{tld}',
        '{service}-{region}.{tld}',
        '{country}.{service}.{tld}',
        '{service}.{region}.{tld}',
        '{lang}.{service}.{tld}'
    ],
    'services': [
        '{function}.{service}.{tld}',
        '{service}-{function}.{tld}',
        '{function}-{service}.{tld}',
        '{function}.{subdomain}.{service}.{tld}',
        '{service}.{function}.{tld}'
    ],
    'mobile': [
        'm.{service}.{tld}',
        'mobile.{service}.{tld}',
        'app.{service}.{tld}',
        '{service}-mobile.{tld}',
        'ios.{service}.{tld}',
        'android.{service}.{tld}',
        'apps.{service}.{tld}'
    ],
    'auth': [
        'auth.{service}.{tld}',
        'login.{service}.{tld}',
        'accounts.{service}.{tld}',
        'signin.{service}.{tld}',
        'sso.{service}.{tld}',
        'oauth.{service}.{tld}',
        '{service}-auth.{tld}',
        'id.{service}.{tld}'
    ],
    'data': [
        'data.{service}.{tld}',
        'analytics.{service}.{tld}',
        'metrics.{service}.{tld}',
        'logs.{service}.{tld}',
        'telemetry.{service}.{tld}',
        'stats.{service}.{tld}'
    ],
    'infrastructure': [
        '{env}.{service}.{tld}',
        '{service}-{env}.{tld}',
        'prod.{service}.{tld}',
        'staging.{service}.{tld}',
        'dev.{service}.{tld}',
        'test.{service}.{tld}',
        'beta.{service}.{tld}',
        'alpha.{service}.{tld}'
    ]
}

SERVICE_VARIATIONS = {
    'netflix': {
        'base': ['netflix', 'nflx', 'netflx'],
        'functions': ['watch', 'stream', 'video', 'movies', 'shows', 'play', 'media', 'content', 'tv'],
        'cdn_terms': ['nflxvideo', 'nflximg', 'nflxext', 'nflxcdn', 'netflixcdn'],
        'regions': ['us', 'eu', 'asia', 'latam', 'uk', 'ca', 'au', 'jp', 'kr', 'br'],
        'subdomains': ['api', 'secure', 'account', 'help', 'devices'],
        'tlds': ['com', 'net', 'tv', 'io'],
        'envs': ['prod', 'production', 'live']
    },
    'spotify': {
        'base': ['spotify', 'sptfy', 'spot', 'spoti'],
        'functions': ['music', 'audio', 'podcast', 'play', 'stream', 'radio', 'artist', 'playlist', 'player'],
        'cdn_terms': ['scdn', 'spotifycdn', 'spoticdn', 'sptfycdn'],
        'regions': ['us', 'eu', 'uk', 'br', 'jp', 'au', 'ca', 'se', 'de', 'fr'],
        'subdomains': ['api', 'accounts', 'open', 'partner', 'developer'],
        'tlds': ['com', 'co', 'fm', 'net'],
        'envs': ['prod', 'live', 'stable']
    },
    'youtube': {
        'base': ['youtube', 'ytube', 'yt', 'youtu'],
        'functions': ['video', 'watch', 'stream', 'live', 'music', 'tv', 'kids', 'studio', 'creators'],
        'cdn_terms': ['ytimg', 'googlevideo', 'ytcdn', 'youtubecdn'],
        'regions': ['us', 'uk', 'ca', 'au', 'in', 'br', 'jp', 'de', 'fr', 'mx'],
        'subdomains': ['api', 'm', 'music', 'tv', 'studio', 'kids'],
        'tlds': ['com', 'be', 'tv', 'co'],
        'envs': ['prod', 'live']
    },
    'microsoft_teams': {
        'base': ['teams', 'msteams', 'microsoftteams', 'teamslive'],
        'functions': ['chat', 'meeting', 'call', 'collab', 'workspace', 'conference', 'video', 'voice'],
        'cdn_terms': ['teamscdn', 'msftcdn', 'teamsassets', 'teamsstatic'],
        'regions': ['us', 'eu', 'asia', 'uk', 'au', 'ca', 'in', 'jp'],
        'subdomains': ['api', 'login', 'web', 'app', 'admin'],
        'tlds': ['com', 'live.com', 'microsoft.com', 'net'],
        'envs': ['prod', 'production']
    },
    'zoom': {
        'base': ['zoom', 'zoomus', 'zm'],
        'functions': ['meeting', 'webinar', 'conference', 'video', 'call', 'chat', 'phone', 'rooms'],
        'cdn_terms': ['zmcdn', 'zoomcdn', 'zoomassets', 'zmstatic'],
        'regions': ['us', 'eu', 'asia', 'uk', 'au', 'ca', 'in', 'sg', 'jp'],
        'subdomains': ['api', 'app', 'web', 'download', 'support'],
        'tlds': ['us', 'com', 'net', 'co'],
        'envs': ['prod', 'live']
    },
    'google_services': {
        'base': ['google', 'goog', 'googl'],
        'functions': ['search', 'mail', 'drive', 'docs', 'maps', 'cloud', 'ads', 'analytics', 'play'],
        'cdn_terms': ['gstatic', 'googleusercontent', 'googleapis', 'googlecdn'],
        'regions': ['us', 'eu', 'asia', 'uk', 'au', 'ca', 'in', 'br', 'jp'],
        'subdomains': ['api', 'accounts', 'mail', 'drive', 'docs', 'cloud'],
        'tlds': ['com', 'net', 'co', 'io'],
        'envs': ['prod', 'production']
    },
    'facebook_meta': {
        'base': ['facebook', 'fb', 'meta', 'instagram', 'insta', 'whatsapp', 'wa'],
        'functions': ['social', 'connect', 'share', 'messenger', 'chat', 'stories', 'reels', 'marketplace'],
        'cdn_terms': ['fbcdn', 'fbstatic', 'metacdn', 'igcdn', 'instagramcdn'],
        'regions': ['us', 'eu', 'asia', 'uk', 'br', 'in', 'mx', 'ca', 'au'],
        'subdomains': ['api', 'graph', 'developers', 'business', 'm'],
        'tlds': ['com', 'net', 'co', 'me'],
        'envs': ['prod', 'live']
    },
    'amazon_aws': {
        'base': ['amazon', 'aws', 'amzn', 'amazn'],
        'functions': ['s3', 'ec2', 'lambda', 'rds', 'dynamo', 'cloudfront', 'elastic', 'compute'],
        'cdn_terms': ['cloudfront', 'awscdn', 'amazonaws', 'awsstatic'],
        'regions': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'sa-east-1'],
        'subdomains': ['console', 'api', 's3', 'ec2', 'lambda'],
        'tlds': ['com', 'amazonaws.com', 'aws', 'net'],
        'envs': ['prod', 'production']
    },
    'apple_services': {
        'base': ['apple', 'icloud', 'itunes', 'appstore'],
        'functions': ['store', 'music', 'tv', 'photos', 'mail', 'drive', 'backup', 'find'],
        'cdn_terms': ['mzstatic', 'applecdn', 'icloudcdn', 'itunescdn'],
        'regions': ['us', 'uk', 'ca', 'au', 'jp', 'cn', 'de', 'fr', 'kr'],
        'subdomains': ['api', 'accounts', 'developer', 'support', 'store'],
        'tlds': ['com', 'apple', 'me', 'co'],
        'envs': ['prod', 'live']
    },
    'github': {
        'base': ['github', 'gh', 'gthb'],
        'functions': ['code', 'repo', 'git', 'actions', 'packages', 'pages', 'gist', 'raw'],
        'cdn_terms': ['githubusercontent', 'githubassets', 'ghcdn', 'githubcdn'],
        'regions': ['us', 'eu', 'asia', 'uk', 'au', 'ca', 'in'],
        'subdomains': ['api', 'gist', 'raw', 'pages', 'actions'],
        'tlds': ['com', 'io', 'dev', 'net'],
        'envs': ['prod', 'production']
    },
    'slack': {
        'base': ['slack', 'slck', 'slak'],
        'functions': ['chat', 'workspace', 'team', 'channel', 'message', 'call', 'huddle', 'connect'],
        'cdn_terms': ['slackcdn', 'slack-edge', 'slackassets', 'slackstatic'],
        'regions': ['us', 'eu', 'asia', 'uk', 'au', 'ca', 'jp'],
        'subdomains': ['api', 'app', 'files', 'status', 'admin'],
        'tlds': ['com', 'net', 'co', 'io'],
        'envs': ['prod', 'production']
    }
}

VALID_TLDS = [
    'com', 'net', 'org', 'io', 'co', 'tv', 'me', 'dev', 'app',
    'us', 'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'in', 'br',
    'cloud', 'online', 'tech', 'digital', 'media', 'live', 'fm'
]

COUNTRY_CODES = {
    'us': 'United States',
    'uk': 'United Kingdom', 
    'ca': 'Canada',
    'au': 'Australia',
    'de': 'Germany',
    'fr': 'France',
    'jp': 'Japan',
    'kr': 'South Korea',
    'cn': 'China',
    'in': 'India',
    'br': 'Brazil',
    'mx': 'Mexico',
    'es': 'Spain',
    'it': 'Italy',
    'nl': 'Netherlands',
    'se': 'Sweden',
    'sg': 'Singapore',
    'nz': 'New Zealand'
}

LANGUAGE_CODES = ['en', 'es', 'fr', 'de', 'pt', 'it', 'ja', 'ko', 'zh', 'hi', 'ar', 'ru']

VERSION_PATTERNS = ['v1', 'v2', 'v3', 'v4', 'v5', '2023', '2024', '2025', 'latest', 'stable']

NUMBER_PATTERNS = ['1', '2', '3', '01', '02', '03', '001', '002', '003']