//! Public-address policy for requests to URLs that a remote party chose
//! (`source.url` imports, generation results and their redirects).
//!
//! A host is resolved before anything is sent, every answer must be a public
//! unicast address (a mixed public/private answer is refused, which also
//! defeats DNS rebinding), and the caller pins the connection to the checked
//! address instead of letting the HTTP client resolve the name again.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use opentake_media::MediaCancelToken;

/// How long one DNS lookup may take.
pub(crate) const DNS_LOOKUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Why a URL's host was refused. Callers map these to their own messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublicTargetError {
    MissingHost,
    NonPublicAddress,
    NoAddresses,
    LookupFailed,
    LookupTimedOut,
    Cancelled,
}

/// Resolve `url`'s host and check every address. Returns the host and, for a
/// DNS name, the address to pin the connection to (`None` for an IP literal,
/// which the client connects to directly).
pub(crate) async fn resolve_public_target<F, Fut>(
    url: &reqwest::Url,
    cancel: &MediaCancelToken,
    lookup: F,
) -> Result<(String, Option<SocketAddr>), PublicTargetError>
where
    F: FnOnce(String, u16) -> Fut,
    Fut: Future<Output = std::io::Result<Vec<SocketAddr>>>,
{
    let host = url
        .host_str()
        .filter(|host| !host.is_empty())
        .ok_or(PublicTargetError::MissingHost)?
        .to_string();
    let port = url.port_or_known_default().unwrap_or(443);
    if let Some(ip) = literal_host_ip(&host) {
        ensure_public_ip(ip)?;
        return Ok((host, None));
    }
    let lookup = lookup(host.clone(), port);
    tokio::pin!(lookup);
    let timeout = tokio::time::sleep(DNS_LOOKUP_TIMEOUT);
    tokio::pin!(timeout);
    let addresses = tokio::select! {
        result = &mut lookup => result.map_err(|_| PublicTargetError::LookupFailed)?,
        () = wait_for_cancel(cancel) => return Err(PublicTargetError::Cancelled),
        () = &mut timeout => return Err(PublicTargetError::LookupTimedOut),
    };
    Ok((host, Some(pin_public_address(addresses)?)))
}

/// The system resolver.
pub(crate) async fn system_lookup(host: String, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    Ok(tokio::net::lookup_host((host.as_str(), port))
        .await?
        .collect())
}

/// Pick the connection address from a DNS answer in which every address must
/// be public.
pub(crate) fn pin_public_address(
    mut addresses: Vec<SocketAddr>,
) -> Result<SocketAddr, PublicTargetError> {
    addresses.sort_unstable();
    addresses.dedup();
    if addresses.is_empty() {
        return Err(PublicTargetError::NoAddresses);
    }
    // Reject mixed public/private answers rather than choosing the public one:
    // this makes split-horizon and rebinding responses fail closed.
    for address in &addresses {
        ensure_public_ip(address.ip())?;
    }
    Ok(addresses[0])
}

pub(crate) fn ensure_public_ip(ip: IpAddr) -> Result<(), PublicTargetError> {
    if public_ip(ip) {
        Ok(())
    } else {
        Err(PublicTargetError::NonPublicAddress)
    }
}

pub(crate) fn literal_host_ip(host: &str) -> Option<IpAddr> {
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
        .parse()
        .ok()
}

pub(crate) fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_ipv4(ip),
        IpAddr::V6(ip) => public_ipv6(ip),
    }
}

fn public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !matches!(
        (a, b, c),
        (0, _, _)
            | (10, _, _)
            | (100, 64..=127, _)
            | (127, _, _)
            | (169, 254, _)
            | (172, 16..=31, _)
            | (192, 0, 0)
            | (192, 0, 2)
            | (192, 88, 99)
            | (192, 168, _)
            | (198, 18..=19, _)
            | (198, 51, 100)
            | (203, 0, 113)
            | (224..=255, _, _)
    )
}

fn public_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(ipv4) = ip.to_ipv4() {
        return public_ipv4(ipv4);
    }
    let segments = ip.segments();
    let first = segments[0];
    if ip.is_unspecified()
        || ip.is_loopback()
        || (first & 0xfe00) == 0xfc00 // unique-local
        || (first & 0xfe00) == 0xfe00 // link/site-local and reserved
        || (first & 0xff00) == 0xff00 // multicast
        || (first & 0xe000) != 0x2000
    // fail closed outside global unicast 2000::/3
    {
        return false;
    }
    let is_special_purpose = matches!(
        (segments[0], segments[1]),
        (0x2001, 0x0000) // Teredo
            | (0x2001, 0x0002) // benchmarking
            | (0x2001, 0x0db8) // documentation
            | (0x2002, _) // 6to4 transition
    ) || (segments[0] == 0x2001
        && matches!(segments[1] & 0xfff0, 0x0010 | 0x0020));
    !is_special_purpose
}

async fn wait_for_cancel(cancel: &MediaCancelToken) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_and_reserved_ranges_are_not_public() {
        for ip in [
            "240.0.0.1",
            "192.0.0.8",
            "192.88.99.1",
            "64:ff9b::a00:1",
            "2002:a00:1::1",
            "2001::1",
            "::a00:1",
            "::ffff:10.0.0.1",
        ] {
            let ip = ip.parse::<IpAddr>().unwrap();
            assert!(!public_ip(ip), "{ip} must not be treated as public");
        }
        for ip in ["93.184.216.34", "2606:4700:4700::1111"] {
            assert!(public_ip(ip.parse().unwrap()), "{ip} is public");
        }
    }

    #[tokio::test]
    async fn names_resolve_once_and_every_answer_must_be_public() {
        let url = reqwest::Url::parse("https://cdn.example.test/result.png").unwrap();
        let cancel = MediaCancelToken::new();
        let public: SocketAddr = "93.184.216.34:443".parse().unwrap();
        let (host, pinned) = resolve_public_target(&url, &cancel, |host, port| async move {
            assert_eq!((host.as_str(), port), ("cdn.example.test", 443));
            Ok(vec![public])
        })
        .await
        .unwrap();
        assert_eq!((host.as_str(), pinned), ("cdn.example.test", Some(public)));

        let mixed = resolve_public_target(&url, &cancel, |_, _| async move {
            Ok(vec![public, "10.0.0.1:443".parse().unwrap()])
        })
        .await;
        assert_eq!(mixed, Err(PublicTargetError::NonPublicAddress));

        let literal = reqwest::Url::parse("https://127.0.0.1/result.png").unwrap();
        let refused = resolve_public_target(&literal, &cancel, |_, _| async move {
            panic!("a literal address is never looked up")
        })
        .await;
        assert_eq!(refused, Err(PublicTargetError::NonPublicAddress));

        cancel.cancel();
        let cancelled = resolve_public_target(&url, &cancel, |_, _| std::future::pending()).await;
        assert_eq!(cancelled, Err(PublicTargetError::Cancelled));
    }
}
