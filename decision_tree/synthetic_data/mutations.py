"""Mutation patterns for synthetic domain generation."""

MUTATIONS = {
    'abbreviations': {
        'service': ['svc', 'srv', 'serv'],
        'content': ['cnt', 'cont', 'cntnt'],
        'delivery': ['del', 'dlv', 'dlvr'],
        'network': ['net', 'nw', 'netwrk'],
        'application': ['app', 'apl', 'applic'],
        'platform': ['plat', 'pltfrm', 'pform'],
        'production': ['prod', 'prd', 'pro'],
        'development': ['dev', 'devel', 'dvlp'],
        'management': ['mgmt', 'mngmt', 'mgnt'],
        'infrastructure': ['infra', 'inf', 'infr']
    },
    'synonyms': {
        'api': ['rest', 'graphql', 'endpoint', 'gateway', 'service'],
        'cdn': ['edge', 'cache', 'static', 'assets', 'content'],
        'media': ['content', 'stream', 'video', 'audio', 'files'],
        'auth': ['login', 'signin', 'sso', 'oauth', 'accounts'],
        'data': ['analytics', 'metrics', 'telemetry', 'stats', 'logs'],
        'mobile': ['app', 'ios', 'android', 'portable'],
        'cloud': ['saas', 'paas', 'iaas', 'hosted'],
        'user': ['member', 'account', 'profile', 'customer']
    },
    'compounds': {
        'patterns': [
            '{word1}-{word2}',
            '{word1}{word2}',
            '{w1}.{w2}',
            '{w1}_{w2}'
        ],
        'common_compounds': [
            ('web', 'app'),
            ('mobile', 'app'),
            ('api', 'gateway'),
            ('cdn', 'edge'),
            ('user', 'content'),
            ('media', 'server'),
            ('auth', 'server'),
            ('data', 'center')
        ]
    },
    'prefixes': {
        'environment': ['dev-', 'test-', 'qa-', 'uat-', 'prod-', 'staging-'],
        'version': ['v1-', 'v2-', 'beta-', 'alpha-', 'rc-'],
        'region': ['us-', 'eu-', 'asia-', 'global-'],
        'protocol': ['http-', 'https-', 'wss-', 'ftp-'],
        'security': ['secure-', 'safe-', 'protected-']
    },
    'suffixes': {
        'environment': ['-dev', '-test', '-qa', '-prod', '-staging'],
        'version': ['-v1', '-v2', '-beta', '-alpha', '-stable'],
        'type': ['-api', '-cdn', '-app', '-web', '-mobile'],
        'status': ['-live', '-active', '-online', '-ready']
    }
}

REALISTIC_VARIANTS = {
    'versioning': [
        'v1', 'v2', 'v3', 'v4', 'v5',
        '1.0', '2.0', '3.0',
        'beta', 'alpha', 'rc', 'stable', 'latest',
        '2023', '2024', '2025',
        'new', 'old', 'legacy', 'next'
    ],
    'environments': [
        'prod', 'production', 'prd',
        'dev', 'development', 'develop',
        'test', 'testing', 'tst',
        'qa', 'quality',
        'uat', 'staging', 'stage', 'stg',
        'demo', 'sandbox', 'preview'
    ],
    'load_balancing': [
        'lb1', 'lb2', 'lb3',
        'node1', 'node2', 'node3', 'node4', 'node5',
        'cluster1', 'cluster2', 'cluster3',
        'server1', 'server2', 'server3',
        'instance1', 'instance2', 'instance3',
        'pod1', 'pod2', 'pod3',
        'shard1', 'shard2', 'shard3'
    ],
    'random_ids': [
        'xyz123', 'abc456', 'def789',
        'tmp001', 'tmp002', 'tmp003',
        'id123', 'id456', 'id789',
        'ref100', 'ref200', 'ref300',
        'key111', 'key222', 'key333'
    ],
    'geographic': [
        'east', 'west', 'north', 'south', 'central',
        'ne', 'nw', 'se', 'sw',
        'primary', 'secondary', 'tertiary',
        'main', 'backup', 'fallback',
        'zone1', 'zone2', 'zone3',
        'region1', 'region2', 'region3'
    ],
    'protocols': [
        'http', 'https', 'ws', 'wss',
        'tcp', 'udp', 'grpc', 'rest',
        'soap', 'graphql', 'websocket'
    ],
    'security': [
        'secure', 'safe', 'protected',
        'auth', 'authenticated', 'authorized',
        'ssl', 'tls', 'encrypted',
        'private', 'public', 'internal', 'external'
    ],
    'performance': [
        'fast', 'quick', 'speed',
        'cache', 'cached', 'optimized',
        'accelerated', 'turbo', 'boost',
        'lite', 'mini', 'compact'
    ]
}

AMBIGUOUS_PATTERNS = [
    'streaming.{tld}',
    'media-cdn.{tld}',
    'api-gateway.{tld}',
    'cloud-services.{tld}',
    'user-content.{tld}',
    'static-assets.{tld}',
    'web-app.{tld}',
    'mobile-api.{tld}',
    'data-analytics.{tld}',
    'auth-service.{tld}',
    'content-delivery.{tld}',
    'platform-api.{tld}',
    'app-backend.{tld}',
    'media-server.{tld}',
    'file-storage.{tld}',
    'api.content.{tld}',
    'cdn.media.{tld}',
    'login.portal.{tld}',
    'secure.gateway.{tld}',
    'global-cdn.{tld}'
]

NOISE_PATTERNS = {
    'typos': {
        'double_letters': lambda s: s[0] + s[0] + s[1:] if len(s) > 1 else s,
        'swap_letters': lambda s: s[1] + s[0] + s[2:] if len(s) > 2 else s,
        'drop_letter': lambda s: s[:-1] if len(s) > 3 else s,
        'add_letter': lambda s: s + s[-1] if len(s) > 2 else s
    },
    'variations': {
        'pluralize': lambda s: s + 's' if not s.endswith('s') else s,
        'remove_plural': lambda s: s[:-1] if s.endswith('s') and len(s) > 4 else s,
        'add_er': lambda s: s + 'er' if not s.endswith('er') else s,
        'add_ing': lambda s: s + 'ing' if not s.endswith('ing') else s
    }
}