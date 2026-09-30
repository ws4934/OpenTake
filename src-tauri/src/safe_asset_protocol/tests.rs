use super::*;

fn local_tempdir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("safe-asset-")
        .tempdir_in(std::env::current_dir().unwrap())
        .unwrap()
}

#[test]
fn serves_only_a_bounded_single_range_from_the_retained_file() {
    let directory = local_tempdir();
    let path = directory.path().join("clip.mp4");
    std::fs::write(&path, b"0123456789").unwrap();
    let range = tauri::http::HeaderValue::from_static("bytes=2-5");

    let response = serve_open_file(&path, None, false, Some(&range)).unwrap();

    assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(response.headers()[CONTENT_RANGE], "bytes 2-5/10");
    assert!(response.headers().contains_key(ETAG));
    assert_eq!(response.body(), b"2345");
}

#[test]
fn stale_if_range_never_splices_bytes_from_a_different_identity() {
    let directory = local_tempdir();
    let path = directory.path().join("clip.mp4");
    std::fs::write(&path, b"0123456789").unwrap();
    let range = tauri::http::HeaderValue::from_static("bytes=2-5");
    let stale_identity = tauri::http::HeaderValue::from_static("\"stale-file\"");
    let (file, final_path) = open_retained_regular_file(&path).unwrap();

    let response = serve_opened_file(
        file,
        &final_path,
        false,
        Some(&range),
        Some(&stale_identity),
    )
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.body(), b"0123456789");
}

#[cfg(unix)]
#[test]
fn nofollow_nonblocking_open_rejects_fifo_and_symlink() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;

    let directory = local_tempdir();
    let regular = directory.path().join("regular.jpg");
    let link = directory.path().join("link.jpg");
    let fifo = directory.path().join("pipe.jpg");
    std::fs::write(&regular, b"jpeg").unwrap();
    symlink(&regular, &link).unwrap();
    let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: fifo_c is a live NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);

    assert!(open_retained_regular_file(&regular).is_ok());
    assert!(open_retained_regular_file(&link).is_err());
    assert!(open_retained_regular_file(&fifo).is_err());
}

#[cfg(unix)]
#[test]
fn final_handle_path_authorization_rejects_a_symlinked_ancestor_escape() {
    use std::os::unix::fs::symlink;
    use tauri::Manager;

    let directory = local_tempdir();
    let outside = local_tempdir();
    std::fs::write(outside.path().join("outside.jpg"), b"outside").unwrap();
    let alias = directory.path().join("alias");
    symlink(outside.path(), &alias).unwrap();
    let requested = alias.join("outside.jpg");

    let (_, final_path) = open_retained_regular_file(&requested).unwrap();
    assert_eq!(final_path, outside.path().join("outside.jpg"));

    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    scope.allow_file(&requested).unwrap();
    scope.forbid_file(&final_path).unwrap();
    assert!(scope_allows_lexical_path(&scope, &requested));
    assert!(!scope_allows_lexical_path(&scope, &final_path));
    assert_eq!(
        serve_open_file(&requested, Some(&scope), false, None)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
}

#[test]
fn rejects_multi_range_and_oversized_full_body() {
    let directory = local_tempdir();
    let path = directory.path().join("clip.bin");
    let file = File::create(&path).unwrap();
    file.set_len(MAX_FULL_BODY_BYTES + 1).unwrap();
    let multi = tauri::http::HeaderValue::from_static("bytes=0-1,4-5");

    assert_eq!(
        serve_open_file(&path, None, false, Some(&multi))
            .unwrap()
            .status(),
        StatusCode::RANGE_NOT_SATISFIABLE
    );
    assert_eq!(
        serve_open_file(&path, None, false, None).unwrap().status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert_eq!(
        serve_open_file(&path, None, true, None).unwrap().status(),
        StatusCode::OK
    );
    assert_eq!(
        serve_open_file(&path, None, true, None).unwrap().body(),
        &Vec::<u8>::new()
    );
}

#[test]
fn response_headers_are_origin_bound_and_inert() {
    let directory = local_tempdir();
    let path = directory.path().join("frame.jpg");
    std::fs::write(&path, b"jpeg").unwrap();

    let response = serve_open_file(&path, None, false, None).unwrap();

    assert_eq!(
        response.headers()[ACCESS_CONTROL_ALLOW_ORIGIN],
        asset_origin()
    );
    assert_eq!(response.headers()["x-content-type-options"], "nosniff");
    assert_eq!(
        response.headers()["content-security-policy"],
        "default-src 'none'; sandbox"
    );
}

#[cfg(target_os = "windows")]
#[test]
fn project_helper_blocks_an_ambient_bundle_replacement_while_retained() {
    let directory = local_tempdir();
    let selected = directory.path().join("Selected.opentake");
    std::fs::create_dir_all(selected.join("media")).unwrap();
    std::fs::write(selected.join("media/clip.mp4"), b"project-a").unwrap();
    let retained = ProjectRoot::open(&selected).unwrap();

    assert!(std::fs::rename(&selected, directory.path().join("Retained-A.opentake")).is_err());

    drop(retained);
    std::fs::rename(&selected, directory.path().join("Retained-A.opentake")).unwrap();
    assert_eq!(
        std::fs::read(directory.path().join("Retained-A.opentake/media/clip.mp4")).unwrap(),
        b"project-a"
    );
}

#[cfg(unix)]
#[test]
fn home_thumbnail_validation_rejects_symlinked_bundles_and_leaves() {
    use std::os::unix::fs::symlink;

    let directory = local_tempdir();
    let target = directory.path().join("Target.opentake");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("thumbnail.jpg"), b"jpeg").unwrap();
    let alias = directory.path().join("Alias.opentake");
    symlink(&target, &alias).unwrap();
    assert!(validate_resident_home_thumbnail(&alias.join("thumbnail.jpg")).is_err());

    let linked_leaf = directory.path().join("Leaf.opentake");
    std::fs::create_dir(&linked_leaf).unwrap();
    symlink(
        target.join("thumbnail.jpg"),
        linked_leaf.join("thumbnail.jpg"),
    )
    .unwrap();
    assert!(validate_resident_home_thumbnail(&linked_leaf.join("thumbnail.jpg")).is_err());

    std::fs::write(target.join("cover.jpg"), b"jpeg").unwrap();
    assert!(
        validate_resident_home_thumbnail(&target.join("cover.jpg")).is_err(),
        "only the bundle's thumbnail.jpg leaf qualifies"
    );
    let plain = directory.path().join("Plain");
    std::fs::create_dir(&plain).unwrap();
    std::fs::write(plain.join("thumbnail.jpg"), b"jpeg").unwrap();
    assert!(
        validate_resident_home_thumbnail(&plain.join("thumbnail.jpg")).is_err(),
        "a thumbnail outside a .opentake bundle is not a Home cover"
    );
    assert!(validate_resident_home_thumbnail(&target.join("thumbnail.jpg")).is_ok());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn helper_rejects_a_parent_that_is_not_the_same_executable() {
    // The Rust test harness is launched by Cargo, so its live parent is a
    // different executable. A self-issued token/PID pair is insufficient.
    let parent_pid = actual_parent_process_id().unwrap();
    assert!(!parent_is_same_executable(parent_pid).unwrap());
}

/// Run one request through the helper's open, parent authorization and serve
/// steps in-process (the pooled path runs the same functions in a helper).
fn helper_exchange(request: &HelperRequest) -> (OpenedMetadata, IsolatedResponse) {
    let opened = open_helper_asset(request);
    let metadata = opened_metadata(request, &opened);
    let response = match opened {
        Ok(opened) => serve_helper_asset(request, opened),
        Err(_) => IsolatedResponse {
            metadata: helper::HelperResponseMetadata {
                token: request.token.clone(),
                final_path: None,
                project_root_identity: None,
                status: 0,
                headers: Vec::new(),
                body_length: 0,
                error_kind: metadata.error_kind,
            },
            body: Vec::new(),
        },
    };
    (metadata, response)
}

fn external_request(token: &str, path: &Path) -> HelperRequest {
    HelperRequest {
        token: token.to_owned(),
        parent_pid: std::process::id(),
        path: NativePath::new(path).to_wire(),
        head_only: false,
        range: None,
        if_range: None,
        project: None,
    }
}

#[cfg(target_os = "linux")]
#[test]
fn native_names_survive_helper_ipc_and_scope_revocation() {
    use std::os::unix::ffi::OsStrExt;
    use tauri::Manager;
    let directory = local_tempdir();
    let raw = directory
        .path()
        .join(std::ffi::OsStr::from_bytes(b"clip-\xff.mp4"));
    let shadow = PathBuf::from(raw.to_string_lossy().as_ref());
    std::fs::write(&raw, b"original").unwrap();
    std::fs::write(&shadow, b"replaced").unwrap();
    let app = tauri::test::mock_app();
    app.manage(
        crate::native_read_scope::NativeReadScope::load(directory.path().join("grants.json"))
            .unwrap(),
    );
    crate::native_read_scope::allow_file(app.handle(), &raw).unwrap();
    let scope = asset_scope_snapshot(app.handle());
    assert!(scope.allows(&raw));
    assert!(!scope.allows(&shadow));
    assert!(!app.handle().asset_protocol_scope().is_allowed(&shadow));

    let request = external_request("native-file", &raw);
    let request: HelperRequest =
        serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    let (opened, response) = helper_exchange(&request);
    assert!(opened.error_kind.is_none());
    let final_path =
        opentake_domain::native_path::decode(opened.final_path.as_deref().unwrap()).unwrap();
    assert!(paths_equal_for_authority(&final_path, &raw));
    assert_eq!(response.body, b"original");
    assert!(scope.allows(&final_path));
    crate::native_read_scope::forbid_file(app.handle(), &raw).unwrap();
    assert!(!asset_scope_snapshot(app.handle()).allows(&raw));
}

const TEST_HELPER_ENV: &str = "OPENTAKE_TEST_ASSET_HELPER";
const TEST_HELPER_MARKER_ENV: &str = "OPENTAKE_TEST_ASSET_HELPER_MARKER";

/// Child-process entry for the pool tests below: the test binary re-runs
/// itself filtered to this test with `TEST_HELPER_ENV` set. In a normal test
/// run the variable is absent and this returns immediately.
#[test]
fn pooled_helper_process_entry() {
    let Ok(mode) = std::env::var(TEST_HELPER_ENV) else {
        return;
    };
    let result = match mode.as_str() {
        "serve" => helper::run_helper_stdio(),
        "hang-once" => {
            let marker = PathBuf::from(std::env::var_os(TEST_HELPER_MARKER_ENV).unwrap());
            if std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker)
                .is_ok()
            {
                misbehaving_helper(|_, _| loop {
                    std::thread::park();
                })
            } else {
                helper::run_helper_stdio()
            }
        }
        "wrong-token" => misbehaving_helper(|request, stdout| {
            helper::write_helper_reply(
                stdout,
                &helper::HelperReply::Opened(OpenedMetadata {
                    token: format!("{}-other", request.token),
                    final_path: Some(request.path.clone()),
                    etag: Some("\"forged\"".to_owned()),
                    project_root_identity: None,
                    error_kind: None,
                }),
                &[],
            )
        }),
        _ => Err(std::io::Error::other("unknown test helper mode")),
    };
    std::process::exit(i32::from(result.is_err()));
}

/// Handshake, read one request, then misbehave with `respond`.
fn misbehaving_helper(
    respond: impl FnOnce(&HelperRequest, &mut std::io::StdoutLock<'static>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(helper::HELPER_HANDSHAKE)?;
    stdout.flush()?;
    let Some(helper::HelperFrame::Request { request, .. }) =
        helper::read_helper_frame(&mut std::io::stdin().lock())?
    else {
        return Err(std::io::Error::other("expected a request"));
    };
    respond(&request, &mut stdout)?;
    // Keep the pipe open so only the parent's deadline or kill ends this.
    loop {
        std::thread::park();
    }
}

fn test_pool(mode: &str, deadline: Duration, extra_env: &[(&str, &Path)]) -> Arc<HelperPool> {
    test_pool_with_slots(mode, deadline, extra_env, pool::HELPER_POOL_SIZE)
}

fn test_pool_with_slots(
    mode: &str,
    deadline: Duration,
    extra_env: &[(&str, &Path)],
    slots: usize,
) -> Arc<HelperPool> {
    let mut env = vec![(TEST_HELPER_ENV.into(), mode.into())];
    env.extend(
        extra_env
            .iter()
            .map(|(key, value)| ((*key).into(), value.as_os_str().to_owned())),
    );
    Arc::new(HelperPool::with_slots(
        pool::HelperLauncher {
            program: Some(std::env::current_exe().unwrap()),
            args: [
                "--exact",
                "safe_asset_protocol::tests::pooled_helper_process_entry",
                "--nocapture",
                "--test-threads=1",
            ]
            .into_iter()
            .map(Into::into)
            .collect(),
            env,
            // libtest prints its banner before the entry runs.
            max_preamble_bytes: 4096,
        },
        deadline,
        slots,
    ))
}

fn serving_pool() -> Arc<HelperPool> {
    test_pool("serve", Duration::from_secs(30), &[])
}

fn multi_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

#[cfg(unix)]
fn get_request_for(path: &Path) -> Request<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt;
    let encoded = percent_encoding::percent_encode(
        path.as_os_str().as_bytes(),
        percent_encoding::NON_ALPHANUMERIC,
    );
    Request::builder()
        .method(Method::GET)
        .uri(format!("http://opentake.local/{encoded}"))
        .body(Vec::new())
        .unwrap()
}

#[cfg(not(unix))]
fn get_request_for(path: &Path) -> Request<Vec<u8>> {
    let encoded = percent_encoding::percent_encode(
        path.to_str().unwrap().as_bytes(),
        percent_encoding::NON_ALPHANUMERIC,
    );
    Request::builder()
        .method(Method::GET)
        .uri(format!("http://opentake.local/{encoded}"))
        .body(Vec::new())
        .unwrap()
}

#[cfg(unix)]
#[test]
fn response_for_request_rejects_scope_only_alias_that_resolves_outside_scope() {
    use std::os::unix::fs::symlink;
    use tauri::Manager;

    let app = tauri::test::mock_app();
    app.manage(AppCore::new());
    let cache_root = app.path().app_cache_dir().unwrap();
    std::fs::create_dir_all(&cache_root).unwrap();
    let cache_directory = tempfile::Builder::new()
        .prefix("safe-asset-cache-alias-")
        .tempdir_in(&cache_root)
        .unwrap();
    let outside = local_tempdir();
    let final_path = outside.path().join("outside.jpg");
    std::fs::write(&final_path, b"outside").unwrap();
    let alias = cache_directory.path().join("alias");
    symlink(outside.path(), &alias).unwrap();
    let requested = alias.join("outside.jpg");

    let scope = app.handle().asset_protocol_scope();
    scope.allow_file(&requested).unwrap();
    scope.forbid_file(&final_path).unwrap();
    assert!(scope_allows_lexical_path(&scope, &requested));
    assert!(!scope_allows_lexical_path(&scope, &final_path));

    // The helper reports the out-of-scope final path before reading; the
    // parent aborts the read.
    let request = external_request("scope-only-alias", &requested);
    let opened = opened_metadata(&request, &open_helper_asset(&request));
    let expected = non_project_asset_authority(
        app.handle(),
        &AppCore::new(),
        &asset_scope_snapshot(app.handle()),
        &requested,
    )
    .expect("the requested alias itself is lexically approved");
    assert!(authorize_opened_asset(
        app.handle(),
        &AppCore::new(),
        None,
        Some(&expected),
        &request.token,
        &opened,
    )
    .is_err());

    let pool = serving_pool();
    let response = multi_thread_runtime().block_on(response_for_request(
        app.handle(),
        get_request_for(&requested),
        &pool,
    ));
    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "a ScopeOnly request must reject an out-of-scope retained final path"
    );
    assert!(response.body() != b"outside");
}

#[cfg(unix)]
#[test]
fn response_for_request_rejects_project_media_ancestor_symlink_escape() {
    use opentake_core::ProbedMedia;
    use std::os::unix::fs::symlink;
    use tauri::Manager;

    let approved = local_tempdir();
    let outside = local_tempdir();
    let final_path = outside.path().join("outside.mp4");
    std::fs::write(&final_path, b"outside-project-media").unwrap();
    let alias = approved.path().join("selected-source");
    symlink(outside.path(), &alias).unwrap();
    let requested = alias.join("outside.mp4");

    let core = AppCore::new();
    core.save_project(Some(approved.path().join("Escape.opentake")))
        .unwrap();
    core.import_media_file(&requested, "outside", &ProbedMedia::default())
        .unwrap();
    let app = tauri::test::mock_app();
    app.manage(core);
    let scope = app.handle().asset_protocol_scope();
    scope.allow_directory(approved.path(), true).unwrap();
    assert!(scope_allows_lexical_path(&scope, &requested));
    assert!(!scope_allows_lexical_path(&scope, &final_path));

    let pool = serving_pool();
    let response = multi_thread_runtime().block_on(response_for_request(
        app.handle(),
        get_request_for(&requested),
        &pool,
    ));

    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "project media must not use a recursive lexical grant to escape through an ancestor symlink"
    );
    assert!(response.body() != b"outside-project-media");
}

#[cfg(unix)]
#[test]
fn project_helper_rejects_an_ambient_bundle_replacement() {
    let directory = local_tempdir();
    let selected = directory.path().join("Selected.opentake");
    std::fs::create_dir_all(selected.join("media")).unwrap();
    std::fs::write(selected.join("media/clip.mp4"), b"project-a").unwrap();
    let retained = ProjectRoot::open(&selected).unwrap();
    let expected_identity = retained.stable_identity();

    std::fs::rename(&selected, directory.path().join("Retained-A.opentake")).unwrap();
    std::fs::create_dir_all(selected.join("media")).unwrap();
    std::fs::write(selected.join("media/clip.mp4"), b"project-b").unwrap();

    let request = HelperRequest {
        project: Some(HelperProjectAuthority {
            project_epoch: 7,
            project_path: selected.to_string_lossy().into_owned(),
            root_identity: expected_identity,
        }),
        ..external_request("test-token", &selected.join("media/clip.mp4"))
    };

    let (opened, response) = helper_exchange(&request);
    assert!(matches!(
        opened.error_kind,
        Some(WireIoErrorKind::PermissionDenied)
    ));
    assert!(response.body.is_empty());
}

#[test]
fn current_project_authority_allows_nested_media_without_recursive_scope() {
    use tauri::Manager;

    let directory = local_tempdir();
    let bundle = directory.path().join("ExactRootGrant.opentake");
    let core = AppCore::new();
    core.save_project(Some(bundle.clone())).unwrap();
    std::fs::create_dir_all(bundle.join("media")).unwrap();
    let media = bundle.join("media/clip.mp4");
    std::fs::write(&media, b"project-media").unwrap();
    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    scope.allow_file(&bundle).unwrap();
    assert!(!scope_allows_lexical_path(&scope, &media));

    let authority = project_request_authority(&core, &asset_scope_snapshot(app.handle()), &media)
        .unwrap()
        .expect("current retained project is nested-media authority");
    let request = HelperRequest {
        project: Some(HelperProjectAuthority::from_core(&authority)),
        ..external_request("test-token", &media)
    };
    let (opened, response) = helper_exchange(&request);
    assert!(authorize_opened_asset(
        app.handle(),
        &core,
        Some(&authority),
        None,
        &request.token,
        &opened
    )
    .is_ok());
    assert_eq!(response.metadata.status, StatusCode::OK.as_u16());
    assert_eq!(response.body, b"project-media");
}

/// Build a project whose opened path runs through a symlinked ancestor
/// (`link -> real`, like macOS `/tmp -> /private/tmp` or a linked volume) and
/// return the pieces of one authorized project-asset read of `media/a.png`.
#[cfg(unix)]
fn symlinked_ancestor_project_read(
    directory: &Path,
) -> (
    AppCore,
    PathBuf,
    PathBuf,
    ProjectAssetAuthority,
    HelperRequest,
) {
    use std::os::unix::fs::symlink;

    let real = directory.join("real");
    std::fs::create_dir(&real).unwrap();
    let link = directory.join("link");
    symlink(&real, &link).unwrap();
    let bundle = link.join("P.opentake");
    let core = AppCore::new();
    core.save_project(Some(bundle.clone())).unwrap();
    std::fs::create_dir_all(bundle.join("media")).unwrap();
    let media = bundle.join("media/a.png");
    std::fs::write(&media, b"png-bytes").unwrap();

    let app = tauri::test::mock_app();
    let authority = project_request_authority(&core, &asset_scope_snapshot(app.handle()), &media)
        .unwrap()
        .expect("the opened project path is the nested-media authority");
    assert_eq!(authority.project_path, bundle);
    let request = HelperRequest {
        project: Some(HelperProjectAuthority::from_core(&authority)),
        ..external_request("symlinked-ancestor-token", &media)
    };
    (core, real, media, authority, request)
}

#[cfg(unix)]
#[test]
fn project_assets_behind_a_symlinked_ancestor_are_served() {
    let directory = local_tempdir();
    let (core, real, _, authority, request) = symlinked_ancestor_project_read(directory.path());
    let (opened, isolated) = helper_exchange(&request);
    let final_path = PathBuf::from(isolated.metadata.final_path.clone().unwrap());
    assert_eq!(
        final_path,
        real.join("P.opentake/media/a.png"),
        "the retained handle reports the resolved path, not the opened alias"
    );
    let app = tauri::test::mock_app();

    let response = isolated_response_to_http(
        app.handle(),
        &core,
        Some(&authority),
        None,
        &request.token,
        &opened,
        isolated,
    );

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a symlinked ancestor must not turn the retained root into a 403"
    );
    assert_eq!(response.body(), b"png-bytes");
}

#[cfg(unix)]
#[test]
fn project_assets_require_the_retained_root_identity_after_the_read() {
    let directory = local_tempdir();
    let (core, _, _, authority, request) = symlinked_ancestor_project_read(directory.path());
    let app = tauri::test::mock_app();
    let other_identity = ProjectRootIdentity {
        volume: authority.root_identity.volume,
        file: authority.root_identity.file.wrapping_add(1),
    };

    for identity in [Some(other_identity), None] {
        // A helper that reports a different root before the read is refused
        // before any byte is read...
        let (mut opened, _) = helper_exchange(&request);
        opened.project_root_identity = identity;
        assert!(authorize_opened_asset(
            app.handle(),
            &core,
            Some(&authority),
            None,
            &request.token,
            &opened
        )
        .is_err());

        // ...and bytes served through any other root are never published,
        // whether the helper reports it consistently or only after the read.
        let (mut opened, mut consistent) = helper_exchange(&request);
        opened.project_root_identity = identity;
        consistent.metadata.project_root_identity = identity;
        let (opened_honestly, mut switched) = helper_exchange(&request);
        switched.metadata.project_root_identity = identity;
        for (opened, isolated) in [(opened, consistent), (opened_honestly, switched)] {
            let response = isolated_response_to_http(
                app.handle(),
                &core,
                Some(&authority),
                None,
                &request.token,
                &opened,
                isolated,
            );
            assert_eq!(
                response.status(),
                StatusCode::FORBIDDEN,
                "bytes read through any root other than the retained one must not be published"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn project_asset_read_rejects_a_bundle_replaced_behind_the_symlinked_ancestor() {
    let directory = local_tempdir();
    let (core, real, _, authority, request) = symlinked_ancestor_project_read(directory.path());
    std::fs::rename(real.join("P.opentake"), real.join("Parked.opentake")).unwrap();
    std::fs::create_dir_all(real.join("P.opentake/media")).unwrap();
    std::fs::write(real.join("P.opentake/media/a.png"), b"replacement").unwrap();
    let app = tauri::test::mock_app();

    let (opened, isolated) = helper_exchange(&request);
    assert!(isolated.body.is_empty());
    assert!(authorize_opened_asset(
        app.handle(),
        &core,
        Some(&authority),
        None,
        &request.token,
        &opened
    )
    .is_err());
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        Some(&authority),
        None,
        &request.token,
        &opened,
        isolated,
    );

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert!(response.body() != b"replacement");
}

#[cfg(unix)]
#[test]
fn home_thumbnail_behind_a_symlinked_ancestor_is_authorized_exactly() {
    use std::os::unix::fs::symlink;
    use tauri::Manager;

    let directory = local_tempdir();
    let real = directory.path().join("real");
    std::fs::create_dir_all(real.join("Recent.opentake")).unwrap();
    let link = directory.path().join("link");
    symlink(&real, &link).unwrap();
    let thumbnail = link.join("Recent.opentake/thumbnail.jpg");
    std::fs::write(&thumbnail, b"jpeg").unwrap();

    let final_path = validate_resident_home_thumbnail(&thumbnail).unwrap();
    assert_eq!(final_path, real.join("Recent.opentake/thumbnail.jpg"));

    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    scope.allow_file(&thumbnail).unwrap();
    scope.allow_file(&final_path).unwrap();
    let expected = non_project_asset_authority(
        app.handle(),
        &AppCore::new(),
        &asset_scope_snapshot(app.handle()),
        &thumbnail,
    )
    .expect("the exact thumbnail grants authorize the Home cover");
    let request = external_request("home-thumbnail-token", &thumbnail);
    let (opened, isolated) = helper_exchange(&request);
    let response = isolated_response_to_http(
        app.handle(),
        &AppCore::new(),
        None,
        Some(&expected),
        &request.token,
        &opened,
        isolated,
    );
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.body(), b"jpeg");
    assert!(
        !scope_allows_lexical_path(&scope, &link.join("Recent.opentake/project.json")),
        "the thumbnail grant must stay exact"
    );
}

#[test]
fn home_thumbnail_exception_requires_an_exact_file_grant() {
    use tauri::Manager;

    let directory = local_tempdir();
    let bundle = directory.path().join("Recent.opentake");
    std::fs::create_dir_all(&bundle).unwrap();
    let thumbnail = bundle.join("thumbnail.jpg");
    std::fs::write(&thumbnail, b"jpeg").unwrap();
    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();

    scope.allow_directory(&bundle, true).unwrap();
    assert!(!is_home_thumbnail_exception(
        &asset_scope_snapshot(app.handle()),
        &thumbnail,
        &bundle
    ));
    scope.allow_file(&thumbnail).unwrap();
    let snapshot = asset_scope_snapshot(app.handle());
    assert!(is_home_thumbnail_exception(&snapshot, &thumbnail, &bundle));
    assert!(matches!(
        non_project_asset_authority(app.handle(), &AppCore::new(), &snapshot, &thumbnail),
        Some(NonProjectAssetAuthority::ScopeOnly {
            kind: ScopeOnlyKind::HomeThumbnail,
            requested_path,
        }) if requested_path == normalized_path(&thumbnail)
    ));
}

#[test]
fn exact_external_grants_follow_the_active_project_while_static_roots_remain_available() {
    use opentake_core::ProbedMedia;
    use tauri::Manager;

    let directory = local_tempdir();
    let source_a = directory.path().join("project-a.mp4");
    let source_b = directory.path().join("project-b.mp4");
    std::fs::write(&source_a, b"project-a").unwrap();
    std::fs::write(&source_b, b"project-b").unwrap();

    let core = AppCore::new();
    let bundle_a = directory.path().join("Project-A.opentake");
    core.save_project(Some(bundle_a)).unwrap();
    core.import_media_file(&source_a, "project-a", &ProbedMedia::default())
        .unwrap();
    core.save_project(None).unwrap();

    let replacement = AppCore::new();
    let bundle_b = directory.path().join("Project-B.opentake");
    replacement.save_project(Some(bundle_b.clone())).unwrap();
    replacement
        .import_media_file(&source_b, "project-b", &ProbedMedia::default())
        .unwrap();
    replacement.save_project(None).unwrap();

    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    scope.allow_directory(directory.path(), true).unwrap();
    let authority = |path: &Path| {
        non_project_asset_authority(
            app.handle(),
            &core,
            &asset_scope_snapshot(app.handle()),
            path,
        )
    };
    let epoch_a = core.project_revision().project_epoch;
    assert!(matches!(
        authority(&source_a),
        Some(NonProjectAssetAuthority::ProjectMedia {
            project_epoch,
            requested_path,
        }) if project_epoch == epoch_a && requested_path == normalized_path(&source_a)
    ));
    let unreferenced_sibling = directory.path().join("unreferenced.mp4");
    std::fs::write(&unreferenced_sibling, b"unreferenced").unwrap();
    assert!(
        authority(&unreferenced_sibling).is_none(),
        "a recursive dialog grant must not expose a sibling absent from the active manifest"
    );

    core.open_project(bundle_b).unwrap();
    assert!(
        authority(&source_a).is_none(),
        "persisted exact grants from project A must not remain active after opening B"
    );
    assert!(authority(&source_b).is_some());

    let cache = app.path().app_cache_dir().unwrap();
    std::fs::create_dir_all(&cache).unwrap();
    let derived = cache.join("poster.png");
    std::fs::write(&derived, b"png").unwrap();
    scope.allow_directory(&cache, true).unwrap();
    assert!(
        authority(&derived).is_some(),
        "application cache/resource roots must not be coupled to the project media set"
    );
}

/// External project media shared by the identity tests below: `requested`
/// is imported into a saved project and exactly granted.
fn external_media_fixture(
    directory: &Path,
    requested: &Path,
) -> (tauri::App<tauri::test::MockRuntime>, AppCore) {
    use opentake_core::ProbedMedia;
    use tauri::Manager;

    let core = AppCore::new();
    core.save_project(Some(directory.join("External.opentake")))
        .unwrap();
    core.import_media_file(requested, "selected", &ProbedMedia::default())
        .unwrap();
    let app = tauri::test::mock_app();
    app.handle()
        .asset_protocol_scope()
        .allow_file(requested)
        .unwrap();
    (app, core)
}

#[cfg(unix)]
#[test]
fn a_symlink_swapped_in_during_a_read_never_publishes_the_other_file() {
    use opentake_core::ProbedMedia;
    use std::os::unix::fs::symlink;
    use tauri::Manager;

    let directory = local_tempdir();
    let source_a_dir = directory.path().join("source-a");
    let source_b_dir = directory.path().join("source-b");
    std::fs::create_dir_all(&source_a_dir).unwrap();
    std::fs::create_dir_all(&source_b_dir).unwrap();
    let source_a = source_a_dir.join("clip.mp4");
    let source_b = source_b_dir.join("clip.mp4");
    std::fs::write(&source_a, b"project-a").unwrap();
    std::fs::write(&source_b, b"project-b").unwrap();
    let alias = directory.path().join("selected-source");
    symlink(&source_a_dir, &alias).unwrap();
    let requested = alias.join("clip.mp4");

    let core = AppCore::new();
    core.save_project(Some(directory.path().join("Race.opentake")))
        .unwrap();
    core.import_media_file(&requested, "selected", &ProbedMedia::default())
        .unwrap();
    // Keep the rebound target referenced by the same current project, so
    // only identity binding (not manifest membership) can reject it.
    core.import_media_file(&source_b, "other", &ProbedMedia::default())
        .unwrap();
    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    scope.allow_directory(directory.path(), true).unwrap();
    let expected = non_project_asset_authority(
        app.handle(),
        &core,
        &asset_scope_snapshot(app.handle()),
        &requested,
    )
    .expect("the selected source is authorized");
    let request = external_request("ancestor-swap-token", &requested);

    // The helper opened A and the parent authorized that identity...
    let opened_a = open_helper_asset(&request);
    let opened = opened_metadata(&request, &opened_a);
    let opened_a = opened_a.unwrap();
    assert!(authorize_opened_asset(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened
    )
    .is_ok());
    // ...then the ancestor is rebound to B during the read.
    std::fs::remove_file(&alias).unwrap();
    symlink(&source_b_dir, &alias).unwrap();

    // Serving from the retained handle still yields A's bytes.
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        serve_helper_asset(&request, opened_a),
    );
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.body(), b"project-a");

    // A helper that answers with B after A was authorized is refused.
    let (_, served_b) = helper_exchange(&request);
    assert!(paths_equal_for_authority(
        Path::new(served_b.metadata.final_path.as_deref().unwrap()),
        &source_b
    ));
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        served_b,
    );
    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "the helper must not publish B bytes under A's authorized identity"
    );
}

#[cfg(unix)]
#[test]
fn response_for_request_rejects_an_exact_project_media_alias_rebound_before_authorization() {
    use std::os::unix::fs::symlink;
    use tauri::Manager;

    let selected = local_tempdir();
    let source_a = local_tempdir();
    let source_b = local_tempdir();
    let media_a = source_a.path().join("clip.mp4");
    let media_b = source_b.path().join("clip.mp4");
    std::fs::write(&media_a, b"project-a").unwrap();
    std::fs::write(&media_b, b"project-b").unwrap();
    let alias = selected.path().join("selected-source");
    symlink(source_a.path(), &alias).unwrap();
    let requested = alias.join("clip.mp4");

    let (app, core) = external_media_fixture(selected.path(), &requested);
    app.manage(core);
    let scope = app.handle().asset_protocol_scope();

    std::fs::remove_file(&alias).unwrap();
    symlink(source_b.path(), &alias).unwrap();
    let (_, final_b) = open_retained_regular_file(&requested).unwrap();
    assert!(paths_equal_for_authority(&final_b, &media_b));
    assert!(scope_allows_lexical_path(&scope, &requested));
    assert!(!scope_allows_lexical_path(&scope, &final_b));

    let pool = serving_pool();
    let response = multi_thread_runtime().block_on(response_for_request(
        app.handle(),
        get_request_for(&requested),
        &pool,
    ));

    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "an exact grant for A must not authorize B after the alias is rebound before the request"
    );
    assert!(response.body() != b"project-b");
}

#[cfg(unix)]
#[test]
fn stable_external_alias_remains_authorized_for_the_same_opened_file() {
    use std::os::unix::fs::symlink;

    let selected_directory = local_tempdir();
    let source_directory = local_tempdir();
    let source = source_directory.path().join("clip.mp4");
    std::fs::write(&source, b"stable-alias").unwrap();
    let alias = selected_directory.path().join("selected-source");
    symlink(source_directory.path(), &alias).unwrap();
    let requested = alias.join("clip.mp4");

    let (app, core) = external_media_fixture(selected_directory.path(), &requested);
    let scope = app.handle().asset_protocol_scope();
    assert!(scope_allows_lexical_path(&scope, &requested));
    assert!(scope_allows_lexical_path(&scope, &source));
    let expected = non_project_asset_authority(
        app.handle(),
        &core,
        &asset_scope_snapshot(app.handle()),
        &requested,
    )
    .expect("stable alias is initially authorized");
    let request = external_request("stable-alias-token", &requested);
    let (opened, isolated) = helper_exchange(&request);

    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        isolated,
    );

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "an unchanged alias to the retained file must remain valid"
    );
    assert_eq!(response.body(), b"stable-alias");
}

#[test]
fn external_authority_rejects_same_path_identity_replacement() {
    let directory = local_tempdir();
    let requested = directory.path().join("selected.mp4");
    let parked = directory.path().join("selected-original.mp4");
    std::fs::write(&requested, b"original-file").unwrap();

    let (app, core) = external_media_fixture(directory.path(), &requested);
    let expected = non_project_asset_authority(
        app.handle(),
        &core,
        &asset_scope_snapshot(app.handle()),
        &requested,
    )
    .expect("original path is initially authorized");
    let request = external_request("same-path-replacement-token", &requested);
    let (opened, _) = helper_exchange(&request);

    std::fs::rename(&requested, &parked).unwrap();
    std::fs::write(&requested, b"replacement-file").unwrap();
    let (_, replacement) = helper_exchange(&request);
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        replacement,
    );

    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "the same pathname must not publish a different file identity than the one authorized"
    );
}

#[test]
fn a_helper_answering_for_the_wrong_token_is_never_published() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    let (app, core) = external_media_fixture(directory.path(), &requested);
    let expected = non_project_asset_authority(
        app.handle(),
        &core,
        &asset_scope_snapshot(app.handle()),
        &requested,
    )
    .unwrap();
    let request = external_request("the-right-token", &requested);
    let (opened, isolated) = helper_exchange(&request);

    assert!(authorize_opened_asset(
        app.handle(),
        &core,
        None,
        Some(&expected),
        "another-request",
        &opened
    )
    .is_err());
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        "another-request",
        &opened,
        isolated,
    );
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[test]
fn authorization_revoked_between_request_and_response_is_rejected() {
    use opentake_core::ProbedMedia;

    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    let authorize = |app: &tauri::App<tauri::test::MockRuntime>, core: &AppCore| {
        non_project_asset_authority(
            app.handle(),
            core,
            &asset_scope_snapshot(app.handle()),
            &requested,
        )
        .expect("authorized at request time")
    };
    let request = external_request("revocation-token", &requested);
    // Each case retains its own bundle: Windows forbids replacing a bundle
    // that another live core still holds open.
    let case_directory = |name: &str| {
        let path = directory.path().join(name);
        std::fs::create_dir(&path).unwrap();
        path
    };

    // 1. The scope grant is revoked (deny precedence) during the read.
    let (app, core) = external_media_fixture(&case_directory("revoked-scope"), &requested);
    let expected = authorize(&app, &core);
    let (opened, isolated) = helper_exchange(&request);
    app.handle()
        .asset_protocol_scope()
        .forbid_file(&requested)
        .unwrap();
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        isolated,
    );
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    // 2. The media is removed from the manifest during the read.
    let (app, core) = external_media_fixture(&case_directory("removed-media"), &requested);
    let expected = authorize(&app, &core);
    let (opened, isolated) = helper_exchange(&request);
    let ids = core
        .media()
        .entries
        .iter()
        .map(|entry| entry.id.clone())
        .collect();
    core.apply(opentake_ops::command::EditCommand::DeleteMedia { asset_ids: ids })
        .unwrap();
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        isolated,
    );
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    // 3. Another project that references the same file is opened.
    let (app, core) = external_media_fixture(&case_directory("switched-project"), &requested);
    let expected = authorize(&app, &core);
    let (opened, isolated) = helper_exchange(&request);
    let other = AppCore::new();
    let other_bundle = directory.path().join("Other.opentake");
    other.save_project(Some(other_bundle.clone())).unwrap();
    other
        .import_media_file(&requested, "same", &ProbedMedia::default())
        .unwrap();
    other.save_project(None).unwrap();
    core.open_project(other_bundle).unwrap();
    let response = isolated_response_to_http(
        app.handle(),
        &core,
        None,
        Some(&expected),
        &request.token,
        &opened,
        isolated,
    );
    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "an authorization bound to the previous project epoch must not publish"
    );
}

#[test]
fn external_authorization_opens_nothing_and_reuses_one_index_per_revision() {
    let directory = local_tempdir();
    // The file does not exist: authorization is lexical, so it must still
    // succeed without touching the file system. The helper reports NotFound.
    let requested = directory.path().join("not-yet-on-disk.mp4");
    let (app, core) = external_media_fixture(directory.path(), &requested);
    let scope = asset_scope_snapshot(app.handle());

    let expected = non_project_asset_authority(app.handle(), &core, &scope, &requested)
        .expect("lexical authorization needs no file I/O");
    let first = external_media_index(&core);
    for _ in 0..100 {
        assert_eq!(
            non_project_asset_authority(app.handle(), &core, &scope, &requested).as_ref(),
            Some(&expected)
        );
        assert!(Arc::ptr_eq(&first, &external_media_index(&core)));
    }
    let request = external_request("missing-token", &requested);
    let (opened, _) = helper_exchange(&request);
    assert!(matches!(opened.error_kind, Some(WireIoErrorKind::NotFound)));
}

#[cfg(target_os = "windows")]
#[test]
fn external_authority_accepts_windows_case_equivalent_paths() {
    assert_eq!(
        authority_key(Path::new(r"C:\Media\Clip.mp4")),
        authority_key(Path::new(r"c:\media\CLIP.MP4"))
    );
    assert_eq!(
        authority_key(Path::new(r"\\?\C:\Media\Clip.mp4")),
        authority_key(Path::new(r"C:\Media\Clip.mp4"))
    );
    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    scope.allow_file(r"C:\Media\Clip.mp4").unwrap();
    assert!(asset_scope_snapshot(app.handle()).allows(Path::new(r"c:\media\CLIP.MP4")));
}

#[test]
fn pooled_helpers_are_reused_across_range_requests() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    let bytes = (0..=255_u8).cycle().take(64 * 1024).collect::<Vec<_>>();
    std::fs::write(&requested, &bytes).unwrap();
    let pool = serving_pool();

    multi_thread_runtime().block_on(async {
        for index in 0..100_usize {
            let start = (index * 512) % (bytes.len() - 512);
            let request = HelperRequest {
                range: Some(format!("bytes={start}-{}", start + 511)),
                ..external_request(&pool::random_token(), &requested)
            };
            let outcome = pool.exchange(&request, |_| Ok::<(), ()>(())).await.unwrap();
            let HelperOutcome::Served { response, .. } = outcome else {
                panic!("the helper must serve an authorized range");
            };
            assert_eq!(
                response.metadata.status,
                StatusCode::PARTIAL_CONTENT.as_u16()
            );
            assert_eq!(response.body, bytes[start..start + 512]);
        }
    });
    eprintln!(
        "helper spawns per 100 sequential Range requests: {}",
        pool.spawned()
    );
    assert!(pool.spawned() <= pool::HELPER_POOL_SIZE);
    assert_eq!(pool.spawned(), 1, "sequential requests reuse one helper");
}

#[test]
fn a_refused_identity_is_never_read_and_the_helper_stays_usable() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"secret").unwrap();
    let pool = serving_pool();

    multi_thread_runtime().block_on(async {
        let request = external_request(&pool::random_token(), &requested);
        let outcome = pool
            .exchange(&request, |opened| {
                assert!(opened.final_path.is_some() && opened.etag.is_some());
                Err("refused")
            })
            .await
            .unwrap();
        assert!(matches!(outcome, HelperOutcome::Refused("refused")));

        let missing = external_request(&pool::random_token(), &directory.path().join("gone.mp4"));
        let outcome = pool.exchange(&missing, |_| Ok::<(), ()>(())).await.unwrap();
        assert!(matches!(outcome, HelperOutcome::OpenFailed(_)));

        let request = external_request(&pool::random_token(), &requested);
        let outcome = pool.exchange(&request, |_| Ok::<(), ()>(())).await.unwrap();
        assert!(matches!(outcome, HelperOutcome::Served { .. }));
    });
    assert_eq!(pool.spawned(), 1);
}

#[test]
fn a_hung_helper_is_killed_and_replaced() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    let marker = directory.path().join("hung-once");
    let pool = test_pool(
        "hang-once",
        Duration::from_secs(2),
        &[(TEST_HELPER_MARKER_ENV, &marker)],
    );

    multi_thread_runtime().block_on(async {
        let request = external_request(&pool::random_token(), &requested);
        let error = pool
            .exchange(&request, |_| Ok::<(), ()>(()))
            .await
            .err()
            .expect("the hung helper must time out");
        assert!(matches!(error, IsolatedHelperError::TimedOut));
        assert!(marker.exists());
        // The stuck helper was killed and reaped, returning its slot.
        assert_eq!(pool.available_slots(), pool::HELPER_POOL_SIZE);

        let request = external_request(&pool::random_token(), &requested);
        let outcome = pool.exchange(&request, |_| Ok::<(), ()>(())).await.unwrap();
        let HelperOutcome::Served { response, .. } = outcome else {
            panic!("the replacement helper must serve");
        };
        assert_eq!(response.body, b"clip");
    });
    assert_eq!(pool.spawned(), 2);
}

#[test]
fn a_helper_answering_with_another_token_is_killed() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    let pool = test_pool("wrong-token", Duration::from_secs(30), &[]);

    multi_thread_runtime().block_on(async {
        let request = external_request(&pool::random_token(), &requested);
        let mut authorized = false;
        let error = pool
            .exchange(&request, |_| {
                authorized = true;
                Ok::<(), ()>(())
            })
            .await
            .err()
            .expect("a mismatched token is a protocol error");
        assert!(matches!(error, IsolatedHelperError::InvalidResponse));
        assert!(
            !authorized,
            "a forged identity must not reach authorization"
        );
        assert_eq!(pool.available_slots(), pool::HELPER_POOL_SIZE);
        assert!(pool.idle_helper_ids().is_empty());
    });
}

#[cfg(unix)]
#[test]
fn retiring_the_pool_reaps_every_helper_process() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    let pool = serving_pool();

    multi_thread_runtime().block_on(async {
        let requests = (0..pool::HELPER_POOL_SIZE)
            .map(|_| external_request(&pool::random_token(), &requested))
            .collect::<Vec<_>>();
        let exchanges = requests
            .iter()
            .map(|request| pool.exchange(request, |_| Ok::<(), ()>(())));
        for outcome in futures_util::future::join_all(exchanges).await {
            assert!(matches!(outcome, Ok(HelperOutcome::Served { .. })));
        }
        let helper_ids = pool.idle_helper_ids();
        assert!(!helper_ids.is_empty() && helper_ids.len() <= pool::HELPER_POOL_SIZE);

        pool.retire_idle_for_test().await;
        assert!(pool.idle_helper_ids().is_empty());
        assert_eq!(pool.available_slots(), pool::HELPER_POOL_SIZE);
        for process_id in helper_ids {
            // SAFETY: signal 0 only probes whether the reaped PID exists.
            assert_eq!(unsafe { libc::kill(process_id as i32, 0) }, -1);
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
        }
    });
}

#[test]
fn a_pool_without_slots_fails_fast_without_spawning() {
    let pool = Arc::new(HelperPool::with_slots(
        pool::HelperLauncher {
            program: Some(PathBuf::from("/nonexistent-helper")),
            args: Vec::new(),
            env: Vec::new(),
            max_preamble_bytes: 0,
        },
        Duration::from_secs(1),
        0,
    ));
    let request = external_request("degraded", Path::new("/nonexistent.mp4"));
    let outcome = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(pool.exchange(&request, |_| Ok::<(), ()>(())));
    assert!(matches!(outcome, Err(IsolatedHelperError::Degraded)));
    assert_eq!(pool.spawned(), 0);
}

#[test]
fn quarantined_slots_fail_fast_without_spawning() {
    // A deadline this long would fail the test if the pool waited for a slot.
    let pool = test_pool("serve", Duration::from_secs(3600), &[]);
    let _quarantined = pool.quarantine_free_slots_for_test();
    let request = external_request("degraded", Path::new("/nonexistent.mp4"));
    let outcome = multi_thread_runtime().block_on(pool.exchange(&request, |_| Ok::<(), ()>(())));
    assert!(matches!(outcome, Err(IsolatedHelperError::Degraded)));
    assert_eq!(pool.spawned(), 0);
}

#[test]
fn a_request_waits_for_a_slot_held_by_a_busy_helper() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    let pool = test_pool_with_slots("serve", Duration::from_secs(30), &[], 1);

    multi_thread_runtime().block_on(async {
        let busy = pool.hold_slot_for_test();
        let release = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            drop(busy);
        });
        let request = external_request(&pool::random_token(), &requested);
        let outcome = pool.exchange(&request, |_| Ok::<(), ()>(())).await;
        assert!(
            matches!(outcome, Ok(HelperOutcome::Served { .. })),
            "a busy slot is not a degraded pool"
        );
        release.await.unwrap();
    });
}

#[test]
fn helpers_in_flight_across_a_project_switch_are_retired_and_replaced() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    std::fs::write(&requested, b"clip").unwrap();
    // One slot: the replacement must wait for the retiring helper's slot.
    let pool = test_pool_with_slots("serve", Duration::from_secs(30), &[], 1);

    multi_thread_runtime().block_on(async {
        let request = external_request(&pool::random_token(), &requested);
        let outcome = pool
            .exchange(&request, |_| {
                // The project changes while this helper serves the request.
                pool.bump_generation_for_test();
                Ok::<(), ()>(())
            })
            .await;
        assert!(matches!(outcome, Ok(HelperOutcome::Served { .. })));
        assert!(
            pool.idle_helper_ids().is_empty(),
            "a helper from the previous project is not reused"
        );

        let request = external_request(&pool::random_token(), &requested);
        let outcome = pool.exchange(&request, |_| Ok::<(), ()>(())).await;
        let Ok(HelperOutcome::Served { response, .. }) = outcome else {
            panic!("the next request must get a fresh helper, not a 503");
        };
        assert_eq!(response.body, b"clip");
    });
    assert_eq!(pool.spawned(), 2);
}

#[test]
fn a_shut_down_pool_starts_no_helper() {
    let pool = serving_pool();
    pool.shutdown();
    let request = external_request("after-exit", Path::new("/nonexistent.mp4"));
    let outcome = multi_thread_runtime().block_on(pool.exchange(&request, |_| Ok::<(), ()>(())));
    assert!(matches!(outcome, Err(IsolatedHelperError::Degraded)));
    assert_eq!(pool.spawned(), 0);
}

#[test]
fn a_reply_larger_than_the_request_allows_is_rejected_before_allocation() {
    let directory = local_tempdir();
    let requested = directory.path().join("clip.mp4");
    let file = File::create(&requested).unwrap();
    file.set_len(64 * 1024).unwrap();
    let request = HelperRequest {
        head_only: true,
        ..external_request("head-token", &requested)
    };
    let (_, served) = helper_exchange(&request);
    let mut forged = serde_json::to_vec(&helper::HelperReply::Served(
        helper::HelperResponseMetadata {
            body_length: 64 * 1024,
            ..served.metadata
        },
    ))
    .unwrap();
    let mut framed = (forged.len() as u32).to_be_bytes().to_vec();
    framed.append(&mut forged);
    framed.extend(std::iter::repeat_n(0_u8, 64 * 1024));
    let result = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(helper::read_helper_reply(&mut framed.as_slice(), 1024));
    assert!(matches!(result, Err(IsolatedHelperError::InvalidResponse)));
}

/// A native final path reaches the parent without being replaced by its
/// Unicode alias. The helper waits for authorization and remains reusable.
#[cfg(target_os = "linux")]
#[test]
fn a_non_utf8_final_path_is_preserved_and_the_helper_serves_the_next_request() {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;

    let directory = local_tempdir();
    let target = directory
        .path()
        .join(std::ffi::OsStr::from_bytes(b"not-utf8-\xff"));
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("clip.mp4"), b"hidden").unwrap();
    symlink(&target, directory.path().join("alias")).unwrap();
    let requested = directory.path().join("alias/clip.mp4");
    let ordinary = directory.path().join("ordinary.mp4");
    std::fs::write(&ordinary, b"ordinary").unwrap();
    let pool = serving_pool();

    multi_thread_runtime().block_on(async {
        let request = external_request(&pool::random_token(), &requested);
        let mut authorized = false;
        let outcome = pool
            .exchange(&request, |opened| {
                let decoded =
                    opentake_domain::native_path::decode(opened.final_path.as_deref().unwrap())
                        .unwrap();
                let expected = target.join("clip.mp4");
                assert_eq!(decoded, expected);
                assert_ne!(decoded, PathBuf::from(expected.to_string_lossy().as_ref()));
                authorized = true;
                Ok::<(), ()>(())
            })
            .await;
        let Ok(HelperOutcome::Served { response, .. }) = outcome else {
            panic!("an authorized native final path must be served");
        };
        assert_eq!(response.body, b"hidden");
        assert!(authorized);

        let request = external_request(&pool::random_token(), &ordinary);
        let outcome = pool.exchange(&request, |_| Ok::<(), ()>(())).await;
        let Ok(HelperOutcome::Served { response, .. }) = outcome else {
            panic!("the same helper must still serve the next request");
        };
        assert_eq!(response.body, b"ordinary");
    });
    assert_eq!(pool.spawned(), 1);
}

#[test]
fn a_burst_of_64_thumbnail_requests_waits_instead_of_failing() {
    use tauri::Manager;

    let app = tauri::test::mock_app();
    app.manage(AppCore::new());
    let cache_root = app.path().app_cache_dir().unwrap();
    std::fs::create_dir_all(&cache_root).unwrap();
    let thumbnails = tempfile::Builder::new()
        .prefix("safe-asset-burst-")
        .tempdir_in(&cache_root)
        .unwrap();
    app.handle()
        .asset_protocol_scope()
        .allow_directory(thumbnails.path(), true)
        .unwrap();
    let paths = (0..64)
        .map(|index| {
            let path = thumbnails.path().join(format!("thumb-{index}.jpg"));
            std::fs::write(&path, format!("jpeg-{index}")).unwrap();
            path
        })
        .collect::<Vec<_>>();
    let protocol = SafeAssetProtocol::with_pool(serving_pool());
    let handle = app.handle().clone();

    let responses = multi_thread_runtime().block_on(async move {
        let requests = paths.iter().map(|path| {
            let protocol = protocol.clone();
            let handle = handle.clone();
            let request = get_request_for(path);
            tokio::spawn(async move { protocol.serve(&handle, request).await })
        });
        let mut responses = Vec::new();
        for response in futures_util::future::join_all(requests).await {
            responses.push(response.unwrap());
        }
        (responses, protocol.pool.spawned())
    });
    let (responses, spawned) = responses;
    for (index, response) in responses.iter().enumerate() {
        assert_eq!(response.status(), StatusCode::OK, "request {index}");
        assert_eq!(response.body(), format!("jpeg-{index}").as_bytes());
    }
    assert!(spawned <= pool::HELPER_POOL_SIZE);
}

#[test]
fn media_ranges_are_capped_at_the_streaming_budget() {
    let directory = local_tempdir();
    let path = directory.path().join("clip.mp4");
    let file = File::create(&path).unwrap();
    file.set_len(MAX_RANGE_BYTES * 3).unwrap();
    let open_ended = tauri::http::HeaderValue::from_static("bytes=0-");

    let response = serve_open_file(&path, None, false, Some(&open_ended)).unwrap();

    assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(response.body().len() as u64, MAX_RANGE_BYTES);
    assert_eq!(
        response.headers()[CONTENT_RANGE],
        format!("bytes 0-{}/{}", MAX_RANGE_BYTES - 1, MAX_RANGE_BYTES * 3)
    );
}

/// Measurement for the pull request (run with `--release --ignored`): the
/// parent's per-request authorization with 5000 external manifest entries and
/// 5000 exact scope grants, warm and after each kind of cache invalidation.
#[test]
#[ignore = "benchmark; run explicitly in release mode"]
fn benchmark_external_authorization_with_5000_entries() {
    use opentake_core::ProbedMedia;
    use tauri::Manager;

    let directory = local_tempdir();
    let core = AppCore::new();
    core.save_project(Some(directory.path().join("Bench.opentake")))
        .unwrap();
    let app = tauri::test::mock_app();
    let scope = app.handle().asset_protocol_scope();
    std::fs::create_dir_all(directory.path().join("media")).unwrap();
    let paths = (0..5000)
        .map(|index| directory.path().join(format!("media/clip-{index:05}.mp4")))
        .collect::<Vec<_>>();
    for path in &paths {
        core.import_media_file(path, "clip", &ProbedMedia::default())
            .unwrap();
        scope.allow_file(path).unwrap();
    }
    std::fs::write(&paths[0], b"clip").unwrap();
    let request = external_request("bench-token", &paths[0]);
    let (opened, _) = helper_exchange(&request);
    app.manage(core);
    let core = app.state::<AppCore>();

    // Everything the parent does for one request besides the helper I/O:
    // admission-time authorization, the pre-read identity check and the
    // final re-check (each takes its own scope snapshot).
    let request_authorization = |path: &Path| {
        let scope = asset_scope_snapshot(app.handle());
        let expected = non_project_asset_authority(app.handle(), &core, &scope, path).unwrap();
        for _ in 0..2 {
            assert!(authorize_opened_asset(
                app.handle(),
                &core,
                None,
                Some(&expected),
                &request.token,
                &opened
            )
            .is_ok());
        }
    };
    let time = |label: &str, run: &dyn Fn()| {
        let started = std::time::Instant::now();
        run();
        eprintln!("{label}: {:?}", started.elapsed());
    };

    time("first request (builds scope cache and index)", &|| {
        request_authorization(&paths[0])
    });
    let rounds = 10_000_u32;
    let started = std::time::Instant::now();
    for _ in 0..rounds {
        request_authorization(&paths[0]);
    }
    eprintln!("steady state: {:?} per request", started.elapsed() / rounds);
    scope
        .allow_file(directory.path().join("media/new-grant.mp4"))
        .unwrap();
    time(
        "first request after a scope grant (allowed set rebuilt)",
        &|| request_authorization(&paths[0]),
    );
    core.import_media_file(
        directory.path().join("media/new-import.mp4"),
        "clip",
        &ProbedMedia::default(),
    )
    .unwrap();
    time(
        "first request after an editor mutation (index rebuilt)",
        &|| request_authorization(&paths[0]),
    );
    let started = std::time::Instant::now();
    for _ in 0..100 {
        std::hint::black_box(core.runtime_snapshot());
    }
    eprintln!(
        "for comparison, one runtime_snapshot(): {:?}",
        started.elapsed() / 100
    );
}

#[cfg(unix)]
#[test]
fn timed_out_isolated_workers_are_killed_reaped_and_capacity_recovers() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let permits = Arc::new(Semaphore::new(MAX_CONCURRENT_READS));
        let process_slots = Arc::new(Semaphore::new(MAX_CONCURRENT_READS));
        let mut workers = Vec::new();
        for _ in 0..MAX_CONCURRENT_READS {
            let permits = permits.clone();
            let process_slots = process_slots.clone();
            workers.push(tokio::spawn(async move {
                let _permit = permits.acquire_owned().await.unwrap();
                let process_slot = process_slots.try_acquire_owned().unwrap();
                let mut child = Command::new("/bin/sleep")
                    .arg("30")
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .kill_on_drop(true)
                    .spawn()
                    .unwrap();
                let process_id = child.id().unwrap();
                assert!(
                    tokio::time::timeout(Duration::from_millis(100), child.wait())
                        .await
                        .is_err()
                );
                terminate_or_quarantine(
                    child,
                    None,
                    process_slot,
                    &Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                )
                .await;
                process_id
            }));
        }
        let mut process_ids = Vec::new();
        for worker in workers {
            process_ids.push(worker.await.unwrap());
        }
        assert_eq!(permits.available_permits(), MAX_CONCURRENT_READS);
        assert_eq!(process_slots.available_permits(), MAX_CONCURRENT_READS);

        for process_id in process_ids {
            // SAFETY: signal 0 only probes whether the already-reaped PID exists.
            assert_eq!(unsafe { libc::kill(process_id as i32, 0) }, -1);
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
        }

        let _permit = tokio::time::timeout(Duration::from_secs(1), permits.acquire())
            .await
            .expect("worker capacity recovers")
            .unwrap();
        let directory = local_tempdir();
        let path = directory.path().join("normal.jpg");
        std::fs::write(&path, b"normal").unwrap();
        let response = serve_open_file(&path, None, false, None).unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.body(), b"normal");
    });
}

#[test]
fn unreapable_wait_is_bounded_and_four_quarantines_fail_the_fifth_fast() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let started = std::time::Instant::now();
        assert!(
            !bounded_reap(
                std::future::pending::<std::io::Result<std::process::ExitStatus>>(),
                Duration::from_millis(25),
            )
            .await
        );
        assert!(started.elapsed() < Duration::from_secs(1));

        let slots = Arc::new(Semaphore::new(MAX_CONCURRENT_READS));
        let quarantined = (0..MAX_CONCURRENT_READS)
            .map(|_| slots.clone().try_acquire_owned().unwrap())
            .collect::<Vec<_>>();
        let spawned = std::sync::atomic::AtomicUsize::new(0);
        if slots.clone().try_acquire_owned().is_ok() {
            spawned.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        assert_eq!(spawned.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(slots.available_permits(), 0);
        drop(quarantined);
        assert_eq!(slots.available_permits(), MAX_CONCURRENT_READS);
    });
}
