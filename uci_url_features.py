"""
PhishGuard Live Demo - URL-Only Feature Extractor (9 UCI Lexical Features)
Author: Aditya Bhatt (URN: 2481184)
"""

import re
from urllib.parse import urlparse

SHORTENING_SERVICES = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|"
    r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|"
    r"vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net"
)

UCI_URL_FEATURE_NAMES = [
    "UsingIP",
    "LongURL",
    "ShortURL",
    "Symbol@",
    "Redirecting//",
    "PrefixSuffix-",
    "SubDomains",
    "NonStdPort",
    "HTTPSDomainURL",
]


def extract_uci_url_features(url: str) -> list[int]:
    """
    Extract the 9 UCI features computable purely from the URL string without
    making network requests or visiting the suspect webpage.
    Returns values encoded as -1 (phishing), 0 (suspicious), or +1 (legitimate).
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # 1. UsingIP
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$|^0x[0-9a-fA-F]+$"
    using_ip = -1 if re.match(ip_pattern, hostname) else 1

    # 2. LongURL
    url_len = len(url)
    long_url = 1 if url_len < 54 else (0 if url_len <= 75 else -1)

    # 3. ShortURL
    short_url = -1 if re.search(SHORTENING_SERVICES, url, flags=re.IGNORECASE) else 1

    # 4. Symbol@
    symbol_at = -1 if "@" in url else 1

    # 5. Redirecting//
    last_double_slash = url.rfind("//")
    redirecting = -1 if last_double_slash > 6 else 1

    # 6. PrefixSuffix-
    prefix_suffix = -1 if "-" in hostname else 1

    # 7. SubDomains
    clean_host = re.sub(r"^www\.", "", hostname)
    dot_count = clean_host.count(".")
    subdomains = 1 if dot_count == 1 else (0 if dot_count == 2 else -1)

    # 8. NonStdPort
    port = parsed.port
    non_std_port = -1 if (port is not None and port not in (80, 443)) else 1

    # 9. HTTPSDomainURL
    https_in_domain = -1 if "https" in hostname.lower() else 1

    return [
        using_ip,
        long_url,
        short_url,
        symbol_at,
        redirecting,
        prefix_suffix,
        subdomains,
        non_std_port,
        https_in_domain,
    ]
