"""
Sample Training Data for Application Classification
Contains real-world domain examples for different applications/services
"""

# Training data organized by application/service
APPLICATION_DOMAINS = {
    'netflix': [
        'netflix.com',
        'nflxvideo.net',
        'nflximg.net', 
        'nflxext.com',
        'nflxso.net',
        'fast.com',  # Netflix speed test
        'assets.nflxext.com',
        'api.netflix.com',
        'secure.netflix.com',
        'help.netflix.com',
        'media.netflix.com',
        'cdn.netflix.com',
        'images.netflix.com'
    ],
    
    'youtube': [
        'youtube.com',
        'youtu.be',
        'yt3.ggpht.com',
        'ytimg.com',
        'youtubei.googleapis.com',
        'youtube-nocookie.com',
        'googlevideo.com',
        'gvt1.com', 
        'gvt2.com',
        'gvt3.com',
        'ytimg.l.google.com',
        'i.ytimg.com',
        's.ytimg.com'
    ],
    
    'spotify': [
        'spotify.com',
        'scdn.co',
        'spotifycdn.com',
        'audio-ak-spotify-com.akamaized.net',
        'api.spotify.com',
        'accounts.spotify.com',
        'login.spotify.com',
        'open.spotify.com',
        'audio4-ak.spotify.com',
        'heads-ak.spotify.com',
        'i.scdn.co',
        'mosaic.scdn.co',
        'static.spotify.com'
    ],
    
    'microsoft_teams': [
        'teams.microsoft.com',
        'teams.live.com',
        'statics.teams.microsoft.com',
        'teams.events.data.microsoft.com',
        'config.teams.microsoft.com',
        'teams.events.data.trafficmanager.net',
        'teams.microsoft.com',
        'api.teams.skype.com',
        'presence.teams.microsoft.com',
        'notifications.teams.microsoft.com',
        'chatsvcagg.teams.microsoft.com',
        'amer.ng.msg.teams.microsoft.com'
    ],
    
    'zoom': [
        'zoom.us',
        'zoomgov.com',
        'zmcdn.com',
        'zoom.com',
        'zoomcdn.com',
        'zoom.com.cn',
        'd2k5nsl2zxldvw.cloudfront.net',  # Zoom CDN
        'assets.zoom.us',
        'api.zoom.us',
        'marketplace.zoom.us',
        'support.zoom.us',
        'blog.zoom.us',
        'explore.zoom.us'
    ],
    
    'google_services': [
        'google.com',
        'googleapis.com',
        'googleusercontent.com',
        'gstatic.com',
        'googlesyndication.com',
        'googletagmanager.com',
        'googleanalytics.com',
        'google-analytics.com',
        'fonts.googleapis.com',
        'fonts.gstatic.com',
        'maps.googleapis.com',
        'translate.googleapis.com',
        'ajax.googleapis.com',
        'accounts.google.com',
        'mail.google.com',
        'drive.google.com',
        'docs.google.com',
        'calendar.google.com'
    ],
    
    'facebook_meta': [
        'facebook.com',
        'fbcdn.net',
        'fb.com',
        'messenger.com',
        'instagram.com',
        'cdninstagram.com',
        'whatsapp.com',
        'whatsapp.net',
        'connect.facebook.net',
        'graph.facebook.com',
        'edge-chat.facebook.com',
        'scontent.cdninstagram.com',
        'static.cdninstagram.com',
        'meta.com',
        'workplace.com'
    ],
    
    'amazon_aws': [
        'amazon.com',
        'amazonaws.com',
        'cloudfront.net',
        'aws.amazon.com',
        'console.aws.amazon.com',
        's3.amazonaws.com',
        'ec2.amazonaws.com',
        'elasticloadbalancing.com',
        'amazonwebservices.com',
        'awsstatic.com',
        'ssl-images-amazon.com',
        'media-amazon.com',
        'primevideo.com',
        'amazonvideo.com'
    ],
    
    'apple_services': [
        'apple.com',
        'icloud.com',
        'itunes.com',
        'app-store.com',
        'mzstatic.com',
        'apple-dns.net',
        'cdn-apple.com',
        'apple.news',
        'is1-ssl.mzstatic.com',
        'is2-ssl.mzstatic.com',
        'is3-ssl.mzstatic.com',
        'configuration.apple.com',
        'init.itunes.apple.com'
    ],
    
    'github': [
        'github.com',
        'githubusercontent.com',
        'githubassets.com',
        'github.io',
        'api.github.com',
        'avatars.githubusercontent.com',
        'raw.githubusercontent.com',
        'codeload.github.com',
        'assets-cdn.github.com',
        'desktop.github.com'
    ],
    
    'slack': [
        'slack.com',
        'slack-edge.com',
        'slack-imgs.com',
        'slack-redir.net',
        'slackb.com',
        'api.slack.com',
        'hooks.slack.com',
        'files.slack.com',
        'a.slack-edge.com',
        'b.slack-edge.com'
    ]
}

# Additional domains that might appear but are more general
GENERAL_DOMAINS = {
    'cdn_services': [
        'cloudflare.com',
        'fastly.com',
        'akamai.net',
        'akamaized.net',
        'edgecastcdn.net',
        'maxcdn.com',
        'bootstrapcdn.com',
        'jsdelivr.net'
    ],
    
    'cloud_storage': [
        'dropbox.com',
        'box.com',
        'onedrive.com',
        'sharepoint.com',
        'live.com',
        'outlook.com'
    ],
    
    'development_tools': [
        'gitlab.com',
        'bitbucket.org',
        'npmjs.com',
        'pypi.org',
        'docker.io',
        'kubernetes.io'
    ]
}

def get_all_training_data():
    """Get all training data as a flat list of (domain, app_label) tuples."""
    training_data = []
    
    for app_name, domains in APPLICATION_DOMAINS.items():
        for domain in domains:
            training_data.append((domain, app_name))
    
    return training_data

def get_app_domains(app_name):
    """Get domains for a specific application."""
    return APPLICATION_DOMAINS.get(app_name, [])

def get_available_apps():
    """Get list of available application names."""
    return list(APPLICATION_DOMAINS.keys())

def print_data_summary():
    """Print a summary of the training data."""
    total_domains = sum(len(domains) for domains in APPLICATION_DOMAINS.values())
    
    print("Training Data Summary:")
    print("=" * 40)
    print(f"Total applications: {len(APPLICATION_DOMAINS)}")
    print(f"Total domains: {total_domains}")
    print()
    
    for app_name, domains in APPLICATION_DOMAINS.items():
        print(f"{app_name:20}: {len(domains):3} domains")
        # Show first few examples
        examples = domains[:3]
        print(f"{'':22} Examples: {', '.join(examples)}")
        if len(domains) > 3:
            print(f"{'':22} ... and {len(domains) - 3} more")
        print()

if __name__ == "__main__":
    print_data_summary()
    
    # Test data access
    print("\nTesting data access:")
    netflix_domains = get_app_domains('netflix')
    print(f"Netflix domains: {netflix_domains[:3]}...")
    
    all_data = get_all_training_data()
    print(f"Total training samples: {len(all_data)}")
    print(f"First few samples: {all_data[:3]}") 